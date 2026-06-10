import os
import sys
from PIL import Image
import torch

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formaai import FormaAi

def main():
    print("=" * 80)
    print("FormaAi Unified Hybrid 3D Neural Network Test")
    print("=" * 80)
    
    # 1. Initialize hybrid model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FormaAi(device=device)
    
    # 2. Load test image
    image_path = "TRELLIS-main/assets/example_image/typical_misc_lantern.png"
    full_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
    
    if not os.path.exists(full_image_path):
        print(f"Error: Test image not found at {full_image_path}")
        return
        
    print(f"Loading test image: {full_image_path}")
    image = Image.open(full_image_path).convert("RGBA")
    
    # 3. Preprocess
    print("Preprocessing image (removing background, centering)...")
    processed_image = model.preprocess(image)
    
    # Save processed image to check
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch/formaai_test")
    os.makedirs(output_dir, exist_ok=True)
    processed_image.save(os.path.join(output_dir, "preprocessed.png"))
    print(f"Saved preprocessed image to: {output_dir}/preprocessed.png")
    
    # 4. Run unified forward pass
    print("Running unified forward pass (Sparse Structure + Latent Flow + Multi-decoder inference)...")
    aligned_outputs = model(
        processed_image,
        seed=42,
        ss_steps=12,
        ss_cfg=7.5,
        slat_steps=12,
        slat_cfg=3.0,
        formats=['mesh', 'gaussian', 'radiance_field'],
        preprocess=False, # Already preprocessed
        refine_gs=False, # Disabled by default
        refine_steps=100
    )
    
    # 5. Print alignment metadata
    center = aligned_outputs['center']
    scale = aligned_outputs['scale']
    print(f"\nCoordinate Alignment Metadata:")
    print(f" - Center offset translated to origin: {center.cpu().numpy().tolist()}")
    print(f" - Scale multiplier: {scale}")
    
    # 6. Save hybrid assets
    print("\nSaving aligned hybrid assets (GLB + PLY + Metadata)...")
    model.save_hybrid_asset(aligned_outputs, output_dir, prefix="forma_hybrid", upscale_factor=2)
    
    print("\n" + "=" * 80)
    print("FormaAi verification completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
