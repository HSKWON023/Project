# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path

from src.model import UNet
from src.diffusion import Diffusion, DiffusionConfig
from src.utils import set_seed, save_image_grid


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),                # [0,1]
        transforms.Lambda(lambda x: x * 2 - 1),  # [-1,1]
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,  # 윈도우: 멀티프로세싱 끄기
        pin_memory=False  # CPU만 쓸 거면 굳이 필요 없음
    )

    # 2. Model & Diffusion
    model = UNet().to(device)
    diff_cfg = DiffusionConfig()
    diffusion = Diffusion(diff_cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    epochs = 20
    T = diff_cfg.timesteps

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        total_loss = 0.0
        for x, _ in pbar:
            x = x.to(device)

            # random t for each sample
            t = torch.randint(0, T, (x.size(0),), device=device).long()

            loss = diffusion.p_losses(model, x, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")

        # save checkpoint
        torch.save(model.state_dict(), ckpt_dir / "model_latest.pth")

        # generate a small sample grid for monitoring
        with torch.no_grad():
            samples = diffusion.sample(model, (32, 1, 28, 28))
            save_image_grid(samples, f"samples/epoch_{epoch:03d}.png", nrow=8)


if __name__ == "__main__":
    main()

