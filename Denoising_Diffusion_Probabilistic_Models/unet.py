import math
import torch
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb



class TimeMLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out),
            nn.SiLU(),
            nn.Linear(out, out),
        )

    def forward(self, t):
        return self.net(t)


class TimeConditionedBlock(nn.Module):
    def __init__(self, cin, cout, time_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

        self.to_scale = nn.Linear(time_dim, cout)
        self.to_shift = nn.Linear(time_dim, cout)

    def forward(self, x, t_emb):
        h = self.block(x)
        scale = self.to_scale(t_emb).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(t_emb).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + scale) + shift


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, base=64, time_embed_dim=128, timesteps=1000):
        super().__init__()

        self.timesteps = timesteps
        self.time_mlp = TimeMLP(time_embed_dim, time_embed_dim)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = TimeConditionedBlock(base, base * 2, time_embed_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = TimeConditionedBlock(base * 2, base * 4, time_embed_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = TimeConditionedBlock(base * 4, base * 4, time_embed_dim)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up_block1 = TimeConditionedBlock(base * 6, base * 2, time_embed_dim)

        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.up_block2 = TimeConditionedBlock(base * 3, base, time_embed_dim)

        self.out_conv = nn.Conv2d(base, in_ch, 1)

    def forward(self, x, t):
        t = t.clamp(0, self.timesteps - 1)
        t_emb = sinusoidal_embedding(t, self.time_mlp.net[0].in_features)
        t_emb = self.time_mlp(t_emb)

        x1 = self.in_conv(x)

        d1 = self.down1(x1, t_emb)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        m = self.mid(p2, t_emb)

        u1 = self.up1(m)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_block1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_block2(u2, t_emb)

        return self.out_conv(u2)
