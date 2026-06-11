import os
import sys
import math
import json
import time
from typing import List, Dict, Tuple, Any, Literal, Union
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from easydict import EasyDict as edict

# Add TRELLIS-main to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRELLIS_DIR = os.path.join(BASE_DIR, "TRELLIS-main")
if TRELLIS_DIR not in sys.path:
    sys.path.append(TRELLIS_DIR)


# Force xformers attention backend for GPU stability
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.representations import Gaussian, MeshExtractResult
    from trellis.utils import render_utils, postprocessing_utils
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"WARNING: TRELLIS packages could not be imported: {e}. Please run within venv_trellis.")
    class Gaussian:
        pass
    class MeshExtractResult:
        pass

class FormaAi(nn.Module):
    """
    FormaAi: A unified hybrid 3D neural network class.
    Consolidates fast generation, discrete Gaussians, and continuous NeRF fields into a single model.
    """
    def __init__(self, pretrained_path: str = "microsoft/TRELLIS-image-large", device: str = "cuda"):
        super().__init__()
        self.device = device
        
        # Load the TRELLIS pipeline internally
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(pretrained_path)
        
        # Override the pipeline device property to always return self.device (cuda)
        # to prevent model offloading from confusing the internal device detection.
        self.pipeline.__class__.device = property(lambda s: torch.device(self.device))
        
        self.pipeline.to("cpu")
        
        # Expose individual sub-networks as nn.Module attributes/layers
        self.image_encoder = self.pipeline.models['image_cond_model']
        self.sparse_flow_model = self.pipeline.models['sparse_structure_flow_model']
        self.sparse_decoder = self.pipeline.models['sparse_structure_decoder']
        self.slat_flow_model = self.pipeline.models['slat_flow_model']
        
        self.decoder_mesh = self.pipeline.models['slat_decoder_mesh']
        self.decoder_gs = self.pipeline.models['slat_decoder_gs']
        self.decoder_rf = self.pipeline.models['slat_decoder_rf']
        
        # Force CPU offloading state initially
        for model in self.pipeline.models.values():
            model.to("cpu")
        torch.cuda.empty_cache()
        
        print("[FormaAi] All model components loaded and ready.")

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocesses the input image (masking background, centering, resizing to 518x518).
        """
        return self.pipeline.preprocess_image(image)

    @torch.no_grad()
    def forward(
        self,
        image: Union[Image.Image, List[Image.Image]],
        seed: int = 42,
        ss_steps: int = 12,
        ss_cfg: float = 7.5,
        slat_steps: int = 12,
        slat_cfg: float = 3.0,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess: bool = True,
        refine_gs: bool = False,
        refine_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Unified forward pass of the hybrid model supporting single or multi-image.
        
        Args:
            image (Union[Image.Image, List[Image.Image]]): The input image or list of images.
            seed (int): The random seed.
            ss_steps (int): Sampling steps for Stage 1 (Sparse Structure).
            ss_cfg (float): Guidance strength for Stage 1.
            slat_steps (int): Sampling steps for Stage 2 (Structured Latent).
            slat_cfg (float): Guidance strength for Stage 2.
            formats (List[str]): Desired output formats.
            preprocess (bool): Whether to run background removal.
            refine_gs (bool): Whether to run differentiable Gaussian optimization.
            refine_steps (int): Optimization steps for Gaussian refinement.
            
        Returns:
            Dict[str, Any]: Aligned representations (Mesh, Gaussian, Radiance Field).
        """
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if isinstance(image, list):
            images = image
        else:
            images = [image]

        if preprocess:
            images = [self.preprocess(img) for img in images]
            
        with torch.no_grad():
            # 1. Condition generation
            self.image_encoder.to(self.device)
            cond = self.pipeline.get_cond(images)
            cond['neg_cond'] = cond['neg_cond'][:1]
            torch.manual_seed(seed)
            
            # 2. Sparse structure generation (Stage 1 flow matching pass)
            self.sparse_flow_model.to(self.device)
            self.sparse_decoder.to(self.device)
            
            if len(images) > 1:
                with self.pipeline.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode='stochastic'):
                    coords = self.pipeline.sample_sparse_structure(
                        cond,
                        num_samples=1,
                        sampler_params={"steps": ss_steps, "cfg_strength": ss_cfg}
                    )
            else:
                coords = self.pipeline.sample_sparse_structure(
                    cond,
                    num_samples=1,
                    sampler_params={"steps": ss_steps, "cfg_strength": ss_cfg}
                )
            
            # Offload Stage 1 models to save memory
            self.image_encoder.to("cpu")
            self.sparse_flow_model.to("cpu")
            self.sparse_decoder.to("cpu")
            torch.cuda.empty_cache()
            
            # 3. Structured latent generation (Stage 2 flow matching pass)
            self.slat_flow_model.to(self.device)
            
            if len(images) > 1:
                with self.pipeline.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode='stochastic'):
                    slat = self.pipeline.sample_slat(
                        cond,
                        coords,
                        sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg}
                    )
            else:
                slat = self.pipeline.sample_slat(
                    cond,
                    coords,
                    sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg}
                )
            
            # Offload Stage 2 model
            self.slat_flow_model.to("cpu")
            torch.cuda.empty_cache()
            
            # 4. Decoding structured latent into the representations
            raw_outputs = self.decode_slat_offloaded(slat, formats=formats)
        
        # 4b. Differentiable Gaussian Refinement
        if refine_gs and 'gaussian' in raw_outputs:
            gs = raw_outputs['gaussian'][0]
            refined_gs = self.refine_gaussians(gs, images[0], steps=refine_steps)
            raw_outputs['gaussian'] = [refined_gs]
            
        with torch.no_grad():
            # 5. Hybrid coordinate alignment
            aligned_outputs = self.align_coordinates(raw_outputs)
        
        return aligned_outputs

    def align_coordinates(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keeps Mesh, Gaussian Splatting, and NeRF coordinates in their native coordinate systems
        to ensure perfect geometric alignment and texture projection.
        Calculates bounding box center and scale for JSON metadata only.
        """
        aligned = {}
        center = torch.zeros(3, device=self.device)
        scale = 1.0
        
        # Determine center and scale from Mesh (most stable geometric bounding box)
        if 'mesh' in outputs:
            mesh = outputs['mesh'][0]
            min_coords = mesh.vertices.min(dim=0).values
            max_coords = mesh.vertices.max(dim=0).values
            center = (min_coords + max_coords) / 2.0
            
            extents = max_coords - min_coords
            max_extent = extents.max().item()
            scale = 1.0 / max_extent if max_extent > 0 else 1.0
            
            aligned['mesh'] = edict(
                vertices=mesh.vertices, # Keep original coordinates
                faces=mesh.faces,
                raw=mesh
            )
        
        aligned['center'] = center
        aligned['scale'] = scale

        if 'gaussian' in outputs:
            aligned['gaussian'] = outputs['gaussian'][0] # Keep original coordinates

        if 'radiance_field' in outputs:
            aligned['radiance_field'] = outputs['radiance_field'][0]

        return aligned

    def decode_slat_offloaded(self, slat: Any, formats: List[str]) -> Dict[str, Any]:
        """
        Decodes the structured latent by sequentially moving each decoder to the GPU and offloading it,
        keeping GPU VRAM usage at a minimum.
        """
        ret = {}
        # Ensure all decoders are initially on CPU
        self.decoder_mesh.to("cpu")
        self.decoder_gs.to("cpu")
        self.decoder_rf.to("cpu")
        torch.cuda.empty_cache()
        
        if 'mesh' in formats:
            print("[FormaAi] Offload: Decoding Mesh on GPU...")
            self.decoder_mesh.to(self.device)
            ret['mesh'] = self.decoder_mesh(slat)
            self.decoder_mesh.to("cpu")
            torch.cuda.empty_cache()
            
        if 'gaussian' in formats:
            print("[FormaAi] Offload: Decoding Gaussian Splatting on GPU...")
            self.decoder_gs.to(self.device)
            ret['gaussian'] = self.decoder_gs(slat)
            self.decoder_gs.to("cpu")
            torch.cuda.empty_cache()
            
        if 'radiance_field' in formats:
            print("[FormaAi] Offload: Decoding Radiance Field on GPU...")
            self.decoder_rf.to(self.device)
            ret['radiance_field'] = self.decoder_rf(slat)
            self.decoder_rf.to("cpu")
            torch.cuda.empty_cache()
            
        return ret

    def refine_gaussians(self, gs: Gaussian, target_image: Image.Image, steps: int = 100) -> Gaussian:
        """
        Runs a quick optimization loop on the GPU to refine the generated Gaussian Splatting
        representation against the input image.
        """
        with torch.enable_grad():
            print(f"[FormaAi] Starting Differentiable Gaussian Refinement ({steps} steps)...")
            from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
            from trellis.renderers.gaussian_render import GaussianRenderer
            import torch.nn.functional as F
            
            start_time = time.time()
            
            # Prepare target image tensor
            target_img_resized = target_image.resize((512, 512), Image.Resampling.LANCZOS)
            target_tensor = torch.from_numpy(np.array(target_img_resized).astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)[:3]
            
            # Enable gradients on the Gaussian parameters
            gs._xyz = torch.nn.Parameter(gs._xyz.clone().detach().requires_grad_(True))
            gs._rotation = torch.nn.Parameter(gs._rotation.clone().detach().requires_grad_(True))
            gs._scaling = torch.nn.Parameter(gs._scaling.clone().detach().requires_grad_(True))
            gs._opacity = torch.nn.Parameter(gs._opacity.clone().detach().requires_grad_(True))
            
            # Setup optimizer
            optimizer = torch.optim.Adam([
                {"params": gs._xyz, "lr": 1e-4},
                {"params": gs._rotation, "lr": 1e-3},
                {"params": gs._scaling, "lr": 5e-3},
                {"params": gs._opacity, "lr": 0.025},
            ], lr=1e-4)
            
            # Setup renderer
            renderer = GaussianRenderer({
                "resolution": 512,
                "near": 0.8,
                "far": 1.6,
                "ssaa": 1,
                "bg_color": (0, 0, 0),
            })
            
            # Camera pose corresponding to input image: yaw=0, pitch=0, radius=2.0, fov=40
            extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([0.0], [0.0], 2.0, 40.0)
            extr, intr = extrinsics[0], intrinsics[0]
            
            initial_loss = None
            for step in range(steps + 1):
                rendered = renderer.render(gs, extr, intr)['color']
                loss = F.l1_loss(rendered, target_tensor)
                
                if step == 0:
                    initial_loss = loss.item()
                    
                if step < steps:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if step % 20 == 0 and step > 0:
                        print(f"  [Refinement] Step {step}/{steps}: Loss = {loss.item():.5f}")
                        
            final_loss = loss.item()
            elapsed = time.time() - start_time
            print(f"[FormaAi] Refinement finished. L1 Loss: {initial_loss:.5f} -> {final_loss:.5f} ({elapsed:.2f} seconds)")
            
            # Detach parameters to save memory/prevent graph retention
            gs._xyz = gs._xyz.data
            gs._rotation = gs._rotation.data
            gs._scaling = gs._scaling.data
            gs._opacity = gs._opacity.data
            
            return gs

    def upscale_texture(self, texture_image: Image.Image, scale_factor: int = 2, upscale_device: str = "auto") -> Image.Image:
        """
        Upscales the texture image using Swin2SR.
        Temporarily offloads TRELLIS models to CPU to free GPU VRAM, runs upscaling,
        and then restores TRELLIS models to GPU.
        Supports GPU, CPU, or Auto-Fallback mode to prevent OOM.
        """
        print(f"[FormaAi] Offloading TRELLIS models to CPU to free VRAM...")
        for model in self.pipeline.models.values():
            model.to("cpu")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
        
        if scale_factor == 2:
            model_id = "caidas/swin2SR-lightweight-x2-64"
        elif scale_factor == 4:
            model_id = "caidas/swin2SR-classical-sr-x4-64"
        else:
            raise ValueError(f"Unsupported upscale factor: {scale_factor}. Choose 2 or 4.")
            
        processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Determine initial run device
        dev_str = upscale_device.lower()
        if "cpu" in dev_str:
            run_device = "cpu"
        else:
            run_device = self.device  # default to cuda if available
            
        orig_w, orig_h = texture_image.size
        has_alpha = texture_image.mode == "RGBA"
        
        if has_alpha:
            rgb_image = texture_image.convert("RGB")
            alpha_image = texture_image.split()[3]
        else:
            rgb_image = texture_image
            
        print(f"[FormaAi] Upscaling mesh texture using Swin2SR x{scale_factor} on {run_device}...")
        
        try:
            model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(run_device)
            inputs = processor(rgb_image, return_tensors="pt").to(run_device)
            with torch.no_grad():
                outputs = model(**inputs)
        except torch.OutOfMemoryError as e:
            if "auto" in dev_str or dev_str == "auto-fallback":
                print(f"[FormaAi WARNING] CUDA Out of Memory on {run_device}. Falling back to CPU...")
                # Clear GPU memory
                if 'model' in locals():
                    del model
                if 'inputs' in locals():
                    del inputs
                torch.cuda.empty_cache()
                gc.collect()
                
                run_device = "cpu"
                print(f"[FormaAi] Retrying Swin2SR upscaling on CPU...")
                model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(run_device)
                inputs = processor(rgb_image, return_tensors="pt").to(run_device)
                with torch.no_grad():
                    outputs = model(**inputs)
            else:
                raise e
                
        output_pixel_values = outputs.reconstruction
        output_pixel_values = output_pixel_values.squeeze().cpu().permute(1, 2, 0).numpy()
        output_pixel_values = np.clip(output_pixel_values * 255.0, 0, 255).astype(np.uint8)
        upscaled_rgb = Image.fromarray(output_pixel_values)
        
        target_w, target_h = orig_w * scale_factor, orig_h * scale_factor
        upscaled_rgb = upscaled_rgb.crop((0, 0, target_w, target_h))
        
        if has_alpha:
            upscaled_alpha = alpha_image.resize((target_w, target_h), Image.Resampling.BICUBIC)
            upscaled_rgb.putalpha(upscaled_alpha)
            
        del model
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[FormaAi] Restoring TRELLIS models to GPU ({self.device})...")
        for model in self.pipeline.models.values():
            model.to(self.device)
            
        print(f"[FormaAi] Texture upscaled from {orig_w}x{orig_h} to {target_w}x{target_h}")
        return upscaled_rgb

    def save_hybrid_asset(self, aligned_outputs: Dict[str, Any], output_dir: str, prefix: str = "hybrid", upscale_factor: int = 2, upscale_device: str = "auto"):
        """
        Saves all aligned 3D representations to a single hybrid asset directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Mesh as GLB
        if 'mesh' in aligned_outputs:
            print(f"[FormaAi] Saving aligned mesh to GLB...")
            glb = postprocessing_utils.to_glb(
                aligned_outputs['gaussian'],
                aligned_outputs['mesh']['raw'], # Use original for to_glb as it processes internally
                simplify=0.95,
                texture_size=1024
            )
            
            # AI Texture Upscaling
            if upscale_factor > 1 and hasattr(glb.visual, 'material') and hasattr(glb.visual.material, 'baseColorTexture'):
                orig_texture = glb.visual.material.baseColorTexture
                if orig_texture is not None:
                    upscaled_texture = self.upscale_texture(orig_texture, scale_factor=upscale_factor, upscale_device=upscale_device)
                    glb.visual.material.baseColorTexture = upscaled_texture
                    
            glb_path = os.path.join(output_dir, f"{prefix}.glb")
            glb.export(glb_path)
            print(f"Mesh saved: {glb_path}")
            
            # Export OBJ + MTL + Textures
            print(f"[FormaAi] Saving aligned mesh to OBJ/MTL...")
            files_before = set(os.listdir(output_dir))
            
            obj_path = os.path.join(output_dir, f"{prefix}.obj")
            glb.export(obj_path)
            
            files_after = set(os.listdir(output_dir))
            new_files = files_after - files_before
            
            # Create a ZIP file of the OBJ model + material + textures
            import zipfile
            zip_path = os.path.join(output_dir, f"{prefix}_obj.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_name in new_files:
                    file_path = os.path.join(output_dir, file_name)
                    zipf.write(file_path, arcname=file_name)
            print(f"OBJ asset package saved: {zip_path}")

        # Save Gaussian Splatting as PLY
        if 'gaussian' in aligned_outputs:
            print(f"[FormaAi] Saving aligned 3D Gaussian splat...")
            ply_path = os.path.join(output_dir, f"{prefix}.ply")
            aligned_outputs['gaussian'].save_ply(ply_path)
            print(f"Gaussians saved: {ply_path}")

        # Save Alignment metadata JSON
        meta_path = os.path.join(output_dir, f"{prefix}_metadata.json")
        metadata = {
            "center": aligned_outputs['center'].cpu().numpy().tolist(),
            "scale": float(aligned_outputs['scale'])
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Alignment metadata saved: {meta_path}")
        print("[FormaAi] All hybrid asset formats saved successfully.")
