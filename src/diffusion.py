# src/diffusion.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    timesteps: int = 300
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Diffusion:
    def __init__(self, config: DiffusionConfig):
        self.config = config
        T = config.timesteps
        device = config.device

        self.betas = torch.linspace(config.beta_start, config.beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """
        x0: (B, C, H, W)
        t: (B,) with values in [0, T-1]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alphas_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alphas_bar_t) * x0 + torch.sqrt(1 - alphas_bar_t) * noise

    def p_losses(self, model, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred_noise = model(x_noisy, t)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        One reverse step: p(x_{t-1} | x_t)
        """
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)

        # predict noise
        eps_theta = model(x_t, t)

        # equation from DDPM
        mean = sqrt_recip_alpha_t * (x_t - betas_t / sqrt_one_minus_alpha_bar_t * eps_theta)

        if t[0] == 0:
            return mean

        noise = torch.randn_like(x_t)
        sigma_t = torch.sqrt(betas_t)
        return mean + sigma_t * noise

    @torch.no_grad()
    def sample(self, model, shape):
        device = self.config.device
        model.eval()
        x = torch.randn(shape, device=device)
        T = self.config.timesteps

        for step in reversed(range(T)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x
