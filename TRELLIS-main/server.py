import os
import sys
from typing import Tuple, List, Dict

# Force xformers backend
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import uuid
import json
import torch
import numpy as np
import imageio
from PIL import Image
from easydict import EasyDict as edict

# Import TRELLIS components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
HISTORY_FILE = os.path.join(OUTPUTS_DIR, "history.json")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Device configuration
use_cpu = "--cpu" in sys.argv
device = 'cpu' if use_cpu else 'cuda'

app = FastAPI(title="FormaAI TRELLIS 3D Creator")

# Mount directories
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global pipeline reference
pipeline = None
in_memory_states = {}  # Store states to extract GLB without re-running

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy().tolist(),
            '_features_dc': gs._features_dc.cpu().numpy().tolist(),
            '_scaling': gs._scaling.cpu().numpy().tolist(),
            '_rotation': gs._rotation.cpu().numpy().tolist(),
            '_opacity': gs._opacity.cpu().numpy().tolist(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy().tolist(),
            'faces': mesh.faces.cpu().numpy().tolist(),
        },
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device=device)
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device=device)
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device=device)
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device=device)
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device=device)
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device=device),
        faces=torch.tensor(state['mesh']['faces'], device=device),
    )
    return gs, mesh

@app.on_event("startup")
def startup_event():
    global pipeline
    if device == 'cuda':
        try:
            torch.zeros(1, device='cuda')
        except Exception as e:
            print("\n" + "="*80)
            print("ERROR: CUDA device is busy or unavailable. Your GPU VRAM is likely full!")
            print("Please free GPU memory or start with: ./venv_trellis/bin/python server.py --cpu")
            print("="*80 + "\n")
            sys.exit(1)
            
    print(f"Loading pipeline on {device}...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    if device == 'cuda':
        pipeline.cuda()
    else:
        pipeline.cpu()
    print("Pipeline loaded successfully.")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/preprocess")
async def api_preprocess(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_id = str(uuid.uuid4())
        input_filename = f"input_{temp_id}.png"
        input_path = os.path.join(OUTPUTS_DIR, input_filename)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Preprocess
        img = Image.open(input_path)
        processed_img = pipeline.preprocess_image(img)
        
        processed_filename = f"processed_{temp_id}.png"
        processed_path = os.path.join(OUTPUTS_DIR, processed_filename)
        processed_img.save(processed_path)
        
        return {
            "success": True,
            "originalUrl": f"/outputs/{input_filename}",
            "processedUrl": f"/outputs/{processed_filename}",
            "id": temp_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def api_generate(
    id: str = Form(...),
    seed: int = Form(0),
    randomize_seed: bool = Form(True),
    ss_guidance_strength: float = Form(7.5),
    ss_sampling_steps: int = Form(12),
    slat_guidance_strength: float = Form(3.0),
    slat_sampling_steps: int = Form(12)
):
    try:
        processed_filename = f"processed_{id}.png"
        processed_path = os.path.join(OUTPUTS_DIR, processed_filename)
        
        if not os.path.exists(processed_path):
            raise HTTPException(status_code=400, detail="Preprocessed image not found. Please upload again.")
            
        # Resolve seed
        if randomize_seed:
            seed = int(np.random.randint(0, np.iinfo(np.int32).max))
            
        # Load image
        img = Image.open(processed_path)
        
        # Run generation
        outputs = pipeline.run(
            img,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        
        # Save state in memory for subsequent GLB extraction (faster than serializing to disk/JSON)
        gs_repr = outputs['gaussian'][0]
        mesh_repr = outputs['mesh'][0]
        in_memory_states[id] = (gs_repr, mesh_repr)
        
        # Render previews
        print("Rendering preview videos...")
        video_color = render_utils.render_video(gs_repr, num_frames=120)['color']
        video_normal = render_utils.render_video(mesh_repr, num_frames=120)['normal']
        combined_video = [np.concatenate([video_color[i], video_normal[i]], axis=1) for i in range(len(video_color))]
        
        video_filename = f"preview_{id}.mp4"
        video_path = os.path.join(OUTPUTS_DIR, video_filename)
        imageio.mimsave(video_path, combined_video, fps=15)
        
        # Save PLY
        ply_filename = f"model_{id}.ply"
        ply_path = os.path.join(OUTPUTS_DIR, ply_filename)
        gs_repr.save_ply(ply_path)
        
        # Clear VRAM cache
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Update history
        history = load_history()
        new_entry = {
            "id": id,
            "seed": seed,
            "ss_guidance_strength": ss_guidance_strength,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_guidance_strength": slat_guidance_strength,
            "slat_sampling_steps": slat_sampling_steps,
            "originalUrl": f"/outputs/input_{id}.png",
            "processedUrl": f"/outputs/processed_{id}.png",
            "videoUrl": f"/outputs/{video_filename}",
            "plyUrl": f"/outputs/{ply_filename}",
            "glbUrl": None,  # Not extracted yet
            "timestamp": torch.cuda.initial_seed() if device == 'cuda' else 0 # Dummy placeholder
        }
        
        # Add to history
        history.insert(0, new_entry)
        save_history(history)
        
        return {
            "success": True,
            "id": id,
            "seed": seed,
            "videoUrl": f"/outputs/{video_filename}",
            "plyUrl": f"/outputs/{ply_filename}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract_glb")
async def api_extract_glb(
    id: str = Form(...),
    mesh_simplify: float = Form(0.95),
    texture_size: int = Form(1024)
):
    try:
        # Check if state is in memory
        if id not in in_memory_states:
            raise HTTPException(status_code=400, detail="Session state not found. Regenerate or restart session.")
            
        gs, mesh = in_memory_states[id]
        
        print("Extracting GLB mesh...")
        glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
        
        glb_filename = f"model_{id}.glb"
        glb_path = os.path.join(OUTPUTS_DIR, glb_filename)
        glb.export(glb_path)
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Update history with GLB path
        history = load_history()
        for entry in history:
            if entry["id"] == id:
                entry["glbUrl"] = f"/outputs/{glb_filename}"
                break
        save_history(history)
        
        return {
            "success": True,
            "glbUrl": f"/outputs/{glb_filename}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def api_history():
    return load_history()

@app.post("/api/delete")
async def api_delete(id: str = Form(...)):
    try:
        history = load_history()
        history = [entry for entry in history if entry["id"] != id]
        save_history(history)
        
        # Clean up files from disk
        for ext in ["png", "mp4", "ply", "glb"]:
            for prefix in ["input_", "processed_", "preview_", "model_"]:
                filepath = os.path.join(OUTPUTS_DIR, f"{prefix}{id}.{ext}")
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("server.py:app" if os.path.basename(__file__) == "server.py" else app, host="127.0.0.1", port=port, reload=True)
