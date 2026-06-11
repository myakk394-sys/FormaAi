import os
import sys
import uuid
import torch
import numpy as np
import asyncio
from PIL import Image
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add roots to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
TRELLIS_DIR = os.path.join(BASE_DIR, "TRELLIS-main")
if TRELLIS_DIR not in sys.path:
    sys.path.append(TRELLIS_DIR)

# Force xformers backend
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from formaai import FormaAi

app = FastAPI(title="FormaAi Custom API Server")

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FormaAi model
print("[Backend] Initializing FormaAi pipeline on GPU/CPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FormaAi(device=device)

# Directory structure
OUTPUT_DIR = os.path.join(BASE_DIR, "static/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory task database
tasks: Dict[str, Dict[str, Any]] = {}

def run_pipeline_task(task_id: str, images: List[Image.Image], params: Dict[str, Any]):
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10
        tasks[task_id]["stage"] = "Вырезание фона и центрирование (Rembg)..."
        
        task_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # 1. Preprocess input images
        processed_images = [model.preprocess(img) for img in images]
        
        # Save first preprocessed image as primary preview
        preprocessed_path = os.path.join(task_dir, "preprocessed.png")
        processed_images[0].save(preprocessed_path)
        
        # Save all preprocessed images for reference
        for idx, p_img in enumerate(processed_images):
            p_img.save(os.path.join(task_dir, f"preprocessed_{idx}.png"))
        
        tasks[task_id]["progress"] = 30
        tasks[task_id]["stage"] = "Запуск 3D-генерации (Stage 1 & Stage 2)..."
        
        # 2. Forward pass (refinement is only allowed/supported for single image)
        run_refine = params["refine_gs"] if len(processed_images) == 1 else False
        
        outputs = model(
            processed_images,
            seed=int(params["seed"]),
            ss_steps=int(params["ss_steps"]),
            ss_cfg=float(params["ss_cfg"]),
            slat_steps=int(params["slat_steps"]),
            slat_cfg=float(params["slat_cfg"]),
            formats=['mesh', 'gaussian'],
            preprocess=False, # Already preprocessed
            refine_gs=run_refine,
            refine_steps=int(params["refine_steps"])
        )
        
        tasks[task_id]["progress"] = 70
        tasks[task_id]["stage"] = "Рендеринг превью 3D-гауссианов..."
        
        # 3. Render refined preview
        from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
        from trellis.renderers.gaussian_render import GaussianRenderer
        
        renderer = GaussianRenderer({
            "resolution": 512,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0, 0, 0),
        })
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([0.0], [0.0], 2.0, 40.0)
        
        with torch.no_grad():
            rendered = renderer.render(outputs['gaussian'], extrinsics[0], intrinsics[0])['color']
            
        rendered_np = (rendered.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rendered_img = Image.fromarray(rendered_np)
        
        refined_render_path = os.path.join(task_dir, "refined_render.png")
        rendered_img.save(refined_render_path)
        
        tasks[task_id]["progress"] = 85
        tasks[task_id]["stage"] = "Экспорт полигональной сетки и запекание текстур (GLB)..."
        
        # 4. Save aligned, textured mesh (GLB) + Gaussians (PLY) + metadata
        model.save_hybrid_asset(
            outputs,
            task_dir,
            prefix="forma_web",
            upscale_factor=int(params["upscale_factor"]),
            upscale_device=params["upscale_device"]
        )
        
        # Paths for result serving
        tasks[task_id]["progress"] = 100
        tasks[task_id]["stage"] = "Генерация успешно завершена!"
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {
            "glb_url": f"/static/outputs/{task_id}/forma_web.glb",
            "ply_url": f"/static/outputs/{task_id}/forma_web.ply",
            "obj_url": f"/static/outputs/{task_id}/forma_web_obj.zip",
            "meta_url": f"/static/outputs/{task_id}/forma_web_metadata.json",
            "mask_url": f"/static/outputs/{task_id}/preprocessed.png",
            "render_url": f"/static/outputs/{task_id}/refined_render.png"
        }
        print(f"[Backend] Task {task_id} completed successfully.")
        
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[Backend ERROR] Task {task_id} failed: {err}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["stage"] = "Ошибка генерации"
        tasks[task_id]["error"] = str(e)

@app.post("/api/generate")
async def generate_3d(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    seed: int = Form(42),
    ss_steps: int = Form(12),
    ss_cfg: float = Form(7.5),
    slat_steps: int = Form(12),
    slat_cfg: float = Form(3.0),
    refine_gs: bool = Form(False),
    refine_steps: int = Form(100),
    upscale_factor: int = Form(2),
    upscale_device: str = Form("Auto-Fallback")
):
    try:
        # Load and validate images
        img_list = []
        for image_file in images:
            contents = await image_file.read()
            import io
            img_list.append(Image.open(io.BytesIO(contents)))
        if not img_list:
            raise ValueError("No images provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file(s): {str(e)}")
        
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "stage": "Инициализация задачи в очереди...",
        "error": None,
        "result": None
    }
    
    params = {
        "seed": seed,
        "ss_steps": ss_steps,
        "ss_cfg": ss_cfg,
        "slat_steps": slat_steps,
        "slat_cfg": slat_cfg,
        "refine_gs": refine_gs,
        "refine_steps": refine_steps,
        "upscale_factor": upscale_factor,
        "upscale_device": upscale_device
    }
    
    # Launch pipeline run in background task thread
    background_tasks.add_task(run_pipeline_task, task_id, img_list, params)
    
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    return tasks[task_id]

# Root endpoint redirect to index.html
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "static/index.html"))

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Start FormaAi Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
