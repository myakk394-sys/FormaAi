import os
import argparse
import urllib.request
from PIL import Image
import torch
import imageio
import numpy as np

# Set environment variables for TRELLIS
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
except ImportError:
    print("WARNING: TRELLIS packages could not be imported. Please make sure you are in the correct Conda environment.")

def download_image(url, save_path):
    print(f"Downloading image from: {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Download completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="TRELLIS CLI Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to local image or direct HTTP/HTTPS URL")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save generated outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--simplify", type=float, default=0.95, help="Ratio of triangles to remove in mesh simplification (0.0 to 1.0)")
    parser.add_argument("--texture_size", type=int, default=1024, help="Texture resolution for GLB export")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve image path
    image_src = args.image
    if image_src.startswith("http://") or image_src.startswith("https://"):
        temp_image_path = os.path.join(args.output_dir, "temp_input.png")
        try:
            download_image(image_src, temp_image_path)
            image_src = temp_image_path
        except Exception as e:
            print(f"Error downloading image: {e}")
            return
    elif not os.path.exists(image_src):
        print(f"Error: Local file '{image_src}' does not exist.")
        return

    # Load input image
    try:
        image = Image.open(image_src).convert("RGBA")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Loading TRELLIS Pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.to(args.device)

    print(f"Running TRELLIS model with seed {args.seed}...")
    outputs = pipeline.run(
        image,
        seed=args.seed,
        formats=["gaussian", "mesh", "radiance_field"]
    )

    print("Rendering preview videos...")
    # Render Gaussian preview video
    try:
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(os.path.join(args.output_dir, "sample_gs.mp4"), video_gs, fps=30)
        print("Saved sample_gs.mp4")
    except Exception as e:
        print(f"Could not render Gaussian video: {e}")

    # Render Radiance Field preview video
    try:
        video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
        imageio.mimsave(os.path.join(args.output_dir, "sample_rf.mp4"), video_rf, fps=30)
        print("Saved sample_rf.mp4")
    except Exception as e:
        print(f"Could not render Radiance Field video: {e}")

    # Render Mesh preview video
    try:
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(os.path.join(args.output_dir, "sample_mesh.mp4"), video_mesh, fps=30)
        print("Saved sample_mesh.mp4")
    except Exception as e:
        print(f"Could not render Mesh video: {e}")

    # Export to GLB
    print("Extracting GLB mesh...")
    try:
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=args.simplify,
            texture_size=args.texture_size
        )
        glb.export(os.path.join(args.output_dir, "sample.glb"))
        print(f"Exported mesh to {os.path.join(args.output_dir, 'sample.glb')}")
    except Exception as e:
        print(f"Error exporting GLB: {e}")

    # Export to PLY (Gaussians)
    print("Saving 3D Gaussian splatting PLY file...")
    try:
        outputs['gaussian'][0].save_ply(os.path.join(args.output_dir, "sample.ply"))
        print(f"Exported PLY to {os.path.join(args.output_dir, 'sample.ply')}")
    except Exception as e:
        print(f"Error exporting PLY: {e}")

    # Cleanup temporary download file
    if args.image.startswith("http://") or args.image.startswith("https://"):
        try:
            os.remove(temp_image_path)
        except:
            pass

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
