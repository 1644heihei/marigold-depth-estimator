"""
Simple runner to perform depth inference with MarigoldDepthPipeline.

Usage example (PowerShell):
  python .\scripts\run_depth_example.py --image .\examples\input.jpg --out .\out\depth.png

The script loads a pretrained model from Hugging Face by default (prs-eth/marigold-depth-v1-1).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# Add repo root to sys.path so we can import marigold from anywhere
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import local marigold package (the package in this repo)
from marigold import MarigoldDepthPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="prs-eth/marigold-depth-v1-1",
        help="Hugging Face repo id or local path",
    )
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument(
        "--out", default="out_depth.png", help="Output image path (colored depth)"
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    p.add_argument(
        "--denoising_steps",
        type=int,
        default=None,
        help="Number of denoising steps (None = use model default)",
    )
    p.add_argument("--ensemble_size", type=int, default=1, help="Ensemble size")
    p.add_argument(
        "--processing_res",
        type=int,
        default=512,
        help="Processing resolution (0 = original)",
    )
    p.add_argument(
        "--no_color", action="store_true", help="Skip saving colorized depth image"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # prepare device and dtype
    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading model {args.model} to {device} (dtype={dtype})...")
    try:
        pipe = MarigoldDepthPipeline.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except Exception as e:
        print("Failed to load model via from_pretrained:", e)
        print("Try running with `--device cpu` or ensure internet connectivity.")
        raise

    pipe = pipe.to(device)

    # load image
    img = Image.open(args.image).convert("RGB")

    print("Running inference... this may take some time depending on model/device")
    out = pipe(
        img,
        denoising_steps=args.denoising_steps,
        ensemble_size=args.ensemble_size,
        processing_res=args.processing_res,
        match_input_res=True,
        show_progress_bar=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save colorized depth if available
    if (not args.no_color) and out.depth_colored is not None:
        print(f"Saving colorized depth to {out_path}")
        out.depth_colored.save(out_path)
    else:
        # fallback: save grayscale depth and numpy
        depth_np = out.depth_np
        depth_img = (depth_np * 255.0).astype("uint8")
        if depth_img.ndim == 2:
            img_pil = Image.fromarray(depth_img)
        else:
            # if single-channel shaped as [H,W] or [C,H,W]
            if depth_img.ndim == 3 and depth_img.shape[0] == 1:
                depth_img = depth_img.squeeze(0)
            img_pil = Image.fromarray(depth_img)
        print(f"Saving grayscale depth to {out_path}")
        img_pil.save(out_path)

    # also save numpy file next to the image
    np_path = out_path.with_suffix(out_path.suffix + ".npy")
    try:
        import numpy as _np

        _np.save(str(np_path), out.depth_np)
        print(f"Saved depth numpy to {np_path}")
    except Exception:
        print("Could not save numpy file (numpy may be missing).")

    print("Done.")


if __name__ == "__main__":
    main()
