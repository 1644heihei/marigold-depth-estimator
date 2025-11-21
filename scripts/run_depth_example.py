"""
Simple runner to perform depth inference with MarigoldDepthPipeline.

Usage example (PowerShell):
  python .\scripts\run_depth_example.py --image .\examples\00000.png --out .\out\00000_depth.png

The script loads a pretrained model from Hugging Face by default (prs-eth/marigold-depth-v1-1).
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

# Add repo root to sys.path so we can import marigold from anywhere
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import local marigold package (the package in this repo)
from marigold import MarigoldDepthPipeline


def get_config():
    """Load configuration from environment variables with sensible defaults.

    Environment variables (examples):
      MARIGOLD_MODEL
      MARIGOLD_EXAMPLES_DIR
      MARIGOLD_PATTERN
      MARIGOLD_OUT_DIR
      MARIGOLD_DEVICE
      MARIGOLD_DENOISING_STEPS
      MARIGOLD_ENSEMBLE_SIZE
      MARIGOLD_PROCESSING_RES
      MARIGOLD_NO_COLOR (set to '1' or 'true' to enable)
    """

    env = os.environ

    def env_int(key, default):
        v = env.get(key)
        if v is None or v == "":
            return default
        try:
            return int(v)
        except Exception:
            return default

    def env_bool(key):
        v = env.get(key)
        if v is None:
            return False
        return str(v).lower() in ("1", "true", "yes")

    model = env.get("MARIGOLD_MODEL", "prs-eth/marigold-depth-v1-1")
    examples_dir = env.get("MARIGOLD_EXAMPLES_DIR", "examples")
    pattern = env.get("MARIGOLD_PATTERN", "*.png")
    out_dir = env.get("MARIGOLD_OUT_DIR", "out")
    device = env.get("MARIGOLD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    denoising_steps = env.get("MARIGOLD_DENOISING_STEPS", "")
    denoising_steps = None if denoising_steps == "" else int(denoising_steps)
    ensemble_size = env_int("MARIGOLD_ENSEMBLE_SIZE", 1)
    processing_res = env_int("MARIGOLD_PROCESSING_RES", 512)
    no_color = env_bool("MARIGOLD_NO_COLOR")

    return SimpleNamespace(
        model=model,
        examples_dir=examples_dir,
        pattern=pattern,
        out_dir=out_dir,
        device=device,
        denoising_steps=denoising_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        no_color=no_color,
    )


def main():
    args = get_config()

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

    # gather input images
    examples_dir = Path(args.examples_dir)
    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return

    # collect files using glob (non-recursive); use rglob if you want recursive
    files = sorted(examples_dir.glob(args.pattern))
    if len(files) == 0:
        print(f"No files found in {examples_dir} matching pattern {args.pattern}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(files)} files... this may take a while.")
    for inp in files:
        try:
            img = Image.open(inp).convert("RGB")
        except Exception as e:
            print(f"Skipping {inp} (could not open): {e}")
            continue

        print(f"Processing {inp} ...")
        out = pipe(
            img,
            denoising_steps=args.denoising_steps,
            ensemble_size=args.ensemble_size,
            processing_res=args.processing_res,
            match_input_res=True,
            show_progress_bar=False,
        )

        # build output path using input stem
        out_path = out_dir / f"{inp.stem}_depth.png"

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

    print("All done.")


if __name__ == "__main__":
    main()
