# src/utils.py
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_grid(tensor, path, nrow=8):
    """
    tensor: (N, 1, H, W) in [-1, 1]
    """
    tensor = (tensor.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    tensor = tensor.cpu()

    N = tensor.size(0)
    rows = (N + nrow - 1) // nrow
    fig, axes = plt.subplots(rows, nrow, figsize=(nrow, rows))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < N:
            axes[i].imshow(tensor[i, 0].numpy(), cmap="gray")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
