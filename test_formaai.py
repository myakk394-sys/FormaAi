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
    
    # 4. Run unified forward pass (single image)
    print("Running unified forward pass for SINGLE image...")
    aligned_outputs = model(
        processed_image,
        seed=42,
        ss_steps=12,
        ss_cfg=7.5,
        slat_steps=12,
        slat_cfg=3.0,
        formats=['mesh', 'gaussian'],
        preprocess=False, # Already preprocessed
        refine_gs=False, # Disabled by default
        refine_steps=100
    )
    
    print("\nSaving single image aligned assets...")
    model.save_hybrid_asset(aligned_outputs, output_dir, prefix="forma_single", upscale_factor=2)

    # 5. Run multi-image forward pass
    print("\n" + "=" * 40)
    print("Testing MULTI-IMAGE mode...")
    print("=" * 40)
    
    multi_img_paths = [
        "TRELLIS-main/assets/example_multi_image/character_1.png",
        "TRELLIS-main/assets/example_multi_image/character_2.png",
        "TRELLIS-main/assets/example_multi_image/character_3.png"
    ]
    multi_images = []
    for p in multi_img_paths:
        full_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
        if os.path.exists(full_p):
            multi_images.append(Image.open(full_p).convert("RGBA"))
            
    if len(multi_images) > 0:
        print(f"Loaded {len(multi_images)} images for multi-view generation.")
        print("Running unified forward pass for MULTI-IMAGE...")
        multi_outputs = model(
            multi_images,
            seed=42,
            ss_steps=12,
            ss_cfg=7.5,
            slat_steps=12,
            slat_cfg=3.0,
            formats=['mesh', 'gaussian'],
            preprocess=True,
            refine_gs=False
        )
        print("\nSaving multi-image aligned assets...")
        model.save_hybrid_asset(multi_outputs, output_dir, prefix="forma_multi", upscale_factor=2)
    else:
        print("Warning: Multi-image assets not found, skipping multi-image test.")

    print("\n" + "=" * 80)
    print("FormaAi verification completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
