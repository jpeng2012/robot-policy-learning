import math
import torch
import torch.nn as nn

def make_beta_schedule(
        num_steps=50,
        bata_start=1e-4,
        beta_end=0.02,
):

    return torch.linspace(bata_start, beta_end, num_steps, dytype=torch.float32)

class DiffusionSchedule:
    def __init__(
            self,
            num_steps=50,
            device="cuda",
    ):
        self.num_steps = num_steps
        self.beta = make_beta_schedule(
            num_steps=self.num_steps,
        )

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

