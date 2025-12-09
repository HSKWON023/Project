# sample.py
import argparse
from pathlib import Path

import torch

from src.model import UNet
from src.diffusion import Diffusion, DiffusionConfig
from src.utils import set_seed, save_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Sampling method: ddpm (original) or ddim (fast sampling)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of DDIM steps (ignored when method=ddpm)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cfg = DiffusionConfig()
    diffusion = Diffusion(cfg)

    model = UNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    # 샘플 생성
    if args.method == "ddpm":
        # DDPM full-step sampling with the original method
        shape = (args.num_samples, 1, 28, 28)  # (N, C, H, W)
        samples = diffusion.sample(model, shape)
        out_path = Path("samples/generated_ddpm_full.png")
    else:
        # DDIM fast sampling
        shape = (args.num_samples, 1, 28, 28)
        samples = diffusion.ddim_sample(
            model,
            shape,
            num_steps=args.num_steps,
            device=device,
        )
        out_path = Path(f"samples/generated_ddim_{args.num_steps}.png")

    # [-1, 1] → [0, 1] 스케일
    samples = (samples.clamp(-1, 1) + 1) / 2.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(samples, out_path)
    print(f"Saved samples to: {out_path}")



if __name__ == "__main__":
    main()
