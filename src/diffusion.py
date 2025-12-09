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

    @torch.no_grad()
    def ddim_sample(
            self,
            model,
            shape,
            num_steps: int = 50,
            device: str = "cpu",
    ):
        """
        Deterministic DDIM sampling (eta = 0).

        - model: noise-prediction network ε_θ(x_t, t)
        - shape: (N, C, H, W)
        - num_steps: number of DDIM steps (e.g., 50, 20)
        """

        device = torch.device(device)

        # --- 여기에서 전체 alpha_bar (ᾱ_t) 를 self.betas 로부터 직접 계산 ---
        # betas: (T,)
        betas = self.betas.to(device)
        alphas = 1.0 - betas  # α_t
        alpha_bar = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏_{s≤t} α_s, shape: (T,)
        T = alpha_bar.shape[0]  # 전체 step 수 (예: 300)
        # -----------------------------------------------------------------------

        N = shape[0]

        # DDIM에서 사용할 타임스텝 서브시퀀스 (0 ~ T-1 중 num_steps개 균일하게 뽑기)
        step_indices = torch.linspace(
            0,
            T - 1,
            steps=num_steps,
            dtype=torch.long,
            device=device,
        )

        # x_T ~ N(0, I)
        x_t = torch.randn(shape, device=device)

        # reverse: t_i -> t_{i-1}
        for i in reversed(range(num_steps)):
            t = step_indices[i]

            if i == 0:
                # t_{-1}는 x_0을 의미 → ᾱ_{-1} = 1
                t_prev = -1
            else:
                t_prev = step_indices[i - 1]

            # scalar ᾱ_t, ᾱ_{t_prev}
            alpha_bar_t = alpha_bar[t]  # ()
            if t_prev >= 0:
                alpha_bar_prev = alpha_bar[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # 배치에 맞는 timestep 텐서
            t_batch = torch.full((N,), t.item(), device=device, dtype=torch.long)

            # 모델이 noise ε_θ(x_t, t) 예측
            eps_theta = model(x_t, t_batch)

            # x_0 추정
            x0_pred = (
                              x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_theta
                      ) / torch.sqrt(alpha_bar_t)

            # eta = 0 인 deterministic DDIM 업데이트
            x_t = (
                    torch.sqrt(alpha_bar_prev) * x0_pred
                    + torch.sqrt(1.0 - alpha_bar_prev) * eps_theta
            )

        # 마지막에는 x_0 근사값
        return x_t
