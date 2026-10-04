from __future__ import annotations

import torch
import torch.nn as nn


class CubePositionHead(nn.Module):
    """
    Decode future cube xyz from the world model's predicted latent state.

    Input:
        future_latent: [B, H, N, D]

    Token layout:
        0:V       agent-view visual tokens
        V:2V      wrist-view visual tokens
        2V        task-state token
        2V + 1    robot-config token

    We preserve the two camera streams separately instead of averaging
    all state tokens together.

    Decoder input:
        [agent_pool, wrist_pool, task_token, config_token]

    Output:
        normalized future cube position [B, H, 3]
    """

    def __init__(
        self,
        latent_dim=384,
        hidden_dim=512,
        visual_tokens_per_camera=16,
    ):
        super().__init__()

        self.visual_tokens_per_camera = visual_tokens_per_camera

        self.mlp = nn.Sequential(
            nn.Linear(
                latent_dim * 4,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                3,
            ),
        )

    def forward(
        self,
        future_latent,
    ):
        """
        future_latent:
            [B, H, N, D]
        """

        V = self.visual_tokens_per_camera

        # Agent-view visual tokens:
        # [B, H, V, D] -> [B, H, D]
        agent_tokens = future_latent[
            :,
            :,
            :V,
            :,
        ]

        agent_pool = agent_tokens.mean(
            dim=2
        )

        # Wrist-view visual tokens:
        # [B, H, V, D] -> [B, H, D]
        wrist_tokens = future_latent[
            :,
            :,
            V : 2 * V,
            :,
        ]

        wrist_pool = wrist_tokens.mean(
            dim=2
        )

        # Keep low-dimensional state tokens directly.
        task_token = future_latent[
            :,
            :,
            2 * V,
            :,
        ]

        config_token = future_latent[
            :,
            :,
            2 * V + 1,
            :,
        ]

        # [B, H, D] x 4 -> [B, H, 4D]
        x = torch.cat(
            [
                agent_pool,
                wrist_pool,
                task_token,
                config_token,
            ],
            dim=-1,
        )

        # [B, H, 4D] -> [B, H, 3]
        cube_pos = self.mlp(x)

        return cube_pos