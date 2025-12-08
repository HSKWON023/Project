# sample.py
import argparse
import torch
from pathlib import Path

from src.model import UNet
from src.diffusion import Diffusion, DiffusionConfig
from src.utils import save_image_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    diff_cfg = DiffusionConfig()
    diffusion = Diffusion(diff_cfg)

    with torch.no_grad():
        samples = diffusion.sample(model, (args.num_samples, 1, 28, 28))
        Path("samples").mkdir(exist_ok=True, parents=True)
        save_image_grid(samples, "samples/generated.png", nrow=8)


if __name__ == "__main__":
    main()
