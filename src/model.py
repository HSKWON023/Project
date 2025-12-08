# src/model.py
import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    """timestep embedding (1D)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, img_ch=1, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # down
        self.conv_in = nn.Conv2d(img_ch, base_ch, 3, padding=1)
        self.res1 = ResidualBlock(base_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)

        self.res2 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)

        # bottleneck
        self.res3 = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # up
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.res4 = ResidualBlock(base_ch * 4, base_ch * 2, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.res5 = ResidualBlock(base_ch * 2, base_ch, time_emb_dim)

        self.conv_out = nn.Conv2d(base_ch, img_ch, 1)

    def forward(self, x, t):
        # t: (B,)
        t_emb = self.time_mlp(t)

        x = self.conv_in(x)
        h1 = self.res1(x, t_emb)
        x = self.down1(h1)

        h2 = self.res2(x, t_emb)
        x = self.down2(h2)

        x = self.res3(x, t_emb)

        x = self.up1(x)
        x = torch.cat([x, h2], dim=1)
        x = self.res4(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, h1], dim=1)
        x = self.res5(x, t_emb)

        return self.conv_out(x)
