from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)

class SpatialViTEncoder(nn.Module):
    """
    Pretrained ViT-B/16 visual encoder.

    Input:
        images:
            [B, C, H, W]

    Output:
        tokens:
            [B, num_latent_tokens, latent_dim]

    For 224x224 images:

        ViT-B/16 produces:

            14 x 14 = 196 patch tokens

    We reshape those tokens back to a spatial grid,
    then pool them to:

            latent_grid_size x latent_grid_size

    Example:

        latent_grid_size = 4

    gives:

        4 x 4 = 16 latent tokens
    """

    def __init__(
            self,
            latent_dim: int = 384,
            latent_grid_size: int = 4,
            pretrained: bool = True,
            freeze_backbone: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_grid_size = latent_grid_size

        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
        else:
            weights = None

        self.vit = vit_b_16(weights=weights)

        self.vit_dim = self.vit.hidden_dim

        # ====================================================
        # Projection into world-model latent dimension
        # ====================================================

        self.projection = nn.Linear(
            self.vit_dim,
            self.latent_dim,
        )

        self.output_norm = nn.LayerNorm(latent_dim)

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
    

    def forward_backbone(
            self, 
            images: torch.Tensor,
    ):
        """
        Run ViT and return patch tokens before
        classification head.

        Output:

            [B, 196, 768]

        for 224x224 input.
        """

        # ----------------------------------------------------
        # Convert image to patch embeddings
        #
        # torchvision ViT:
        #
        #     [B, 3, 224, 224]
        #
        # becomes:
        #
        #     [B, 196, 768]
        # ----------------------------------------------------

        x = self.vit._process_input(images)

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # Add CLS token
        # ----------------------------------------------------
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # ----------------------------------------------------
        # ViT Transformer encoder
        # ----------------------------------------------------
        x = self.vit.encoder(x)

         # ----------------------------------------------------
        # Remove CLS token
        #
        # Keep spatial patch tokens only.
        # ----------------------------------------------------
        patch_tokens = x[:, 1:, :]

        return patch_tokens

    def forward(
            self, 
            images: torch.Tensor,
    ):
        patch_tokens = self.forward_backbone(images)

        B, N, D = patch_tokens.shape

        grid_size = int(N ** 0.5)

        if grid_size * grid_size != N:
            raise ValueError(
                "Patch token count is not "
                f"a square number: {N}"
            )

        # ----------------------------------------------------
        # [B, N, D]
        #
        # ->
        #
        # [B, D, H, W]
        # ----------------------------------------------------
        x = patch_tokens.reshape(
            B,
            grid_size,
            grid_size, 
            D,
        )

        x = x.permute(0, 3, 1, 2)

        # ====================================================
        # Spatial compression
        #
        # 14 x 14
        #
        # ->
        #
        # 4 x 4
        # ====================================================

        x = F.adaptive_avg_pool2d(
            x,
            output_size = (self.latent_grid_size, self.latent_grid_size),
        )

        # ----------------------------------------------------
        # Return token sequence
        #
        # [B, D, 4, 4]
        #
        # ->
        #
        # [B, 16, D]
        # ----------------------------------------------------

        x = x.flatten(2)
        x = x.transpose(1, 2)

        # ====================================================
        # Project 768 -> latent_dim
        # ====================================================

        x = self.projection(x)
        x = self.output_norm(x)

        return x