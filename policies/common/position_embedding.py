import math
import torch
import torch.nn as nn


class PositionEmbeddingSine2D(nn.Module):
    def __init__(
      self,
      hidden_dim=512,
      temperature=10000,      
    ):
        super().__init__()
        assert hidden_dim%2==0

        self.num_pos_feats = hidden_dim//2
        self.temperature = temperature

    def forward(self, x):
        """
        x: B,C,H,W

        return:
            B,hidden_dim,H,W
        """

        B, _, H, W = x.shape
        device = x.device
        dtype = x.dtype

        y_embed = torch.arange(
            H, 
            device=device,
            dtype=dtype,
        ).unsqueeze(1).expand(H, W)

        x_embed = torch.arange(
            W,
            device=device,
            dtype=dtype,
        ).unsqueeze(0).expand(H, W)

        # normalize approximately to [0, 2pi]

        eps = 1.0e-6
        y_embed = y_embed / (H - 1 + eps) * 2 * math.pi
        x_embed = x_embed / (W - 1 + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)

        dim_t = self.temperature ** (
            2 * torch.div( dim_t, 2, rounding_mode="floor") / self.num_pos_feats

        )

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t


        pos_x = torch.stack(
            (
                pos_x[:, :, 0::2].sin(),
                pos_x[:, :, 1::2].cos(),
            ), 
            dim=3
        ).flatten(2)

        pos_y = torch.stack(
            (
                pos_y[:, :, 0::2].sin(),
                pos_y[:, :, 1::2].cos(),
            ), 
            dim=3
        ).flatten(2)

        # H,W,hidden_dim
        pos = torch.cat(
            [pos_y, pos_x], dim=2
        )

        # B,hidden_dim,H,W
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        return pos