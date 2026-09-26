from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observation_encoder import ObservationEncoder
from .dynamics_transformer import DynamicsTransformer


class LatentWorldModel(nn.Module):
    """
    JEPA-style action-conditioned latent world model.

    Main flow:

        current observation
            |
            v
        online encoder
            |
            v
        z_t
            |
            + action chunk
            |
            v
        dynamics transformer
            |
            v
        predicted future latents


    Future observations are encoded by an EMA target encoder:

        future observation
            |
            v
        target encoder
            |
            v
        target future latents


    The target encoder receives no gradients.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        latent_grid_size: int = 4,
        task_state_dim: int = 9,
        robot_config_dim: int = 7,
        action_dim: int = 7,
        horizon: int = 16,
        num_dynamics_layers: int = 6,
        num_dynamics_heads: int = 6,
        dynamics_ff_dim: int = 1536,
        dropout: float = 0.1,
        ema_momentum: float = 0.996,
        freeze_vision: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_grid_size = latent_grid_size
        self.horizon = horizon
        self.ema_momentum = ema_momentum

        # ====================================================
        # Number of latent tokens
        # ====================================================
        #
        # Each camera:
        #
        #     latent_grid_size^2
        #
        # With 4x4:
        #
        #     16 tokens per camera
        #
        # plus:
        #
        #     1 task-state token
        #     1 robot-config token
        #
        # ====================================================

        self.num_visual_tokens_per_camera = latent_grid_size * latent_grid_size
        self.num_state_tokens = 2 * self.num_visual_tokens_per_camera + 2

        self.online_encoder = ObservationEncoder(
               latent_dim=latent_dim,
               latent_grid_size=latent_grid_size,
               task_state_dim=task_state_dim,
               robot_config_dim=robot_config_dim,
               pretrained_vision=True,
               freeze_vision=freeze_vision,
        )

        # ====================================================
        # Target encoder
        # ====================================================
        #
        # Start as an exact copy of online encoder.
        #
        # It will NOT receive gradients.
        #
        # Instead:
        #
        #     target <- EMA(online)
        #
        # ====================================================

        self.target_encoder = copy.deepcopy(
               self.online_encoder,
        )

        for param in self.target_encoder.parameters():
               param.requires_grad = False

        # ====================================================
        # Dynamics predictor
        # ====================================================

        self.dynamics = DynamicsTransformer(
               latent_dim=latent_dim,
               action_dim=action_dim,
               num_state_tokens=self.num_state_tokens,
               horizon=horizon,
               num_layers=num_dynamics_layers,
               num_heads=num_dynamics_heads,
               ff_dim=dynamics_ff_dim,
               dropout=dropout,
        )

    
    @torch.no_grad()
    def update_target_encoder(
            self,
    ):
        """
        EMA update:

            target =
                m * target
                + (1-m) * online
        """

        momentum = self.ema_momentum

        for online_param, target_param in zip(
             self.online_encoder.parameters(),
             self.target_encoder.parameters(),
        ):
            target_param.data.mul_(momentum)
            target_param.data.add_(
                 online_param,
                 alpha=1.0 - momentum,
            )


    def encode_current(
              self,
              agent_images: torch.Tensor,
              wrist_images: torch.Tensor,
              task_state: torch.Tensor,
              robot_config: torch.Tensor,
    ):
        """
        Encode the current observation.

        Expected input:

            agent_images:
                [B, 3, H, W]

            wrist_images:
                [B, 3, H, W]

            task_state:
                [B, 9]

            robot_config:
                [B, 7]

        Output:

            [B, N, D]
        """

        return self.online_encoder(
             agent_images=agent_images,
             wrist_images=wrist_images,
             task_state=task_state,
             robot_config=robot_config,
        )

    @torch.no_grad()
    def encode_future_targets(
        self,
        future_agent_images: torch.Tensor,
        future_wrist_images: torch.Tensor,
        future_task_state: torch.Tensor,
        future_robot_config: torch.Tensor,
    ):
        """
        Encode H future observations using the target encoder.

        Inputs:

            future_agent_images:
                [B, H, 3, image_h, image_w]

            future_wrist_images:
                [B, H, 3, image_h, image_w]

            future_task_state:
                [B, H, 9]

            future_robot_config:
                [B, H, 7]

        Output:

            target_latents:
                [B, H, N, D]
        """

        B, H = future_agent_images.shape[:2]

        agent = future_agent_images.flatten(0, 1)
        wrist = future_wrist_images.flatten(0, 1)
        task_state = future_task_state.flatten(0, 1)
        robot_config = future_robot_config.flatten(0, 1)

        target = self.target_encoder(
            agent_images=agent,
            wrist_images=wrist,
            task_state=task_state,
            robot_config=robot_config,
        )

        target = target.reshape(
            B,
            H,
            self.num_state_tokens,
            self.latent_dim,
        )

        return target

    def predict_future(
            self,
            current_latent: torch.Tensor,
            actions: torch.Tensor,
    ):
        """
        Predict the whole future latent chunk.

        Inputs:

            current_latent:
                [B, N, D]

            actions:
                [B, H, action_dim]

        Output:

            [B, H, N, D]
        """

        return self.dynamics(
            state_tokens=current_latent,
            actions=actions,
        )

    def forward(
            self,
            agent_images: torch.Tensor,
            wrist_images: torch.Tensor,
            task_state: torch.Tensor,
            robot_config: torch.Tensor,
            actions: torch.Tensor,
            future_agent_images: torch.Tensor,
            future_wrist_images: torch.Tensor,
            future_task_state: torch.Tensor,
            future_robot_config: torch.Tensor,
    ):
        # ====================================================
        # Current world state
        # ====================================================

        current_latent = self.encode_current(
            agent_images=agent_images,
            wrist_images=wrist_images,
            task_state=task_state,
            robot_config=robot_config,
        )

        predicted_future = self.predict_future(
            current_latent=current_latent,
            actions=actions,
        )

        target_future = self.encode_future_targets(
            future_agent_images=future_agent_images,
            future_wrist_images=future_wrist_images,
            future_task_state=future_task_state,
            future_robot_config=future_robot_config,
        )

        return {
            "current_latent":
                current_latent,

            "predicted_future":
                predicted_future,

            "target_future":
                target_future,
        }


    def compute_latent_loss(
            self,
            predicted_future: torch.Tensor,
            target_future: torch.Tensor,
    ):
        """
        Compute group-balanced latent prediction loss.

        Token layout:

            agent visual tokens
            wrist visual tokens
            task-state token
            robot-config token

        We calculate each group independently so the 32 visual
        tokens do not numerically overwhelm the two robot-state
        tokens.
        """

        V = self.num_visual_tokens_per_camera

        agent_pred = predicted_future[:, :, :V, :]
        agent_target = target_future[:, :, :V, :]

        wrist_pred = predicted_future[:, :, V:2*V, :]
        wrist_target = target_future[:, :, V:2*V, :]

        task_pred = predicted_future[:, :, 2*V:2*V+1, :]
        task_target = target_future[:, :, 2*V:2*V+1, :]

        config_pred = predicted_future[:, :, 2*V+1:, :]
        config_target = target_future[:, :, 2*V+1:, :]

        agent_pred = F.normalize(
            agent_pred,
            dim=-1,
        )

        agent_target = F.normalize(
            agent_target,
            dim=-1,
        )

        wrist_pred = F.normalize(
            wrist_pred,
            dim=-1,
        )

        wrist_target = F.normalize(
            wrist_target,
            dim=-1,
        )

        task_pred = F.normalize(
            task_pred,
            dim=-1,
        )

        task_target = F.normalize(
            task_target,
            dim=-1,
        )

        config_pred = F.normalize(
            config_pred,
            dim=-1,
        )

        config_target = F.normalize(
            config_target,
            dim=-1,
        )

        # ====================================================
        # Loss per semantic group
        # ====================================================

        agent_loss = F.mse_loss(
            agent_pred,
            agent_target,
        )

        wrist_loss = F.mse_loss(
            wrist_pred,
            wrist_target,
        )

        task_loss = F.mse_loss(
            task_pred,
            task_target,
        )

        config_loss = F.mse_loss(
            config_pred,
            config_target,
        )

        # ====================================================
        # Equal weighting between semantic groups
        # ====================================================

        total_loss = (
            agent_loss
            + wrist_loss
            + task_loss
            + config_loss
        ) / 4.0

        return {
            "loss":
                total_loss,

            "agent_loss":
                agent_loss.detach(),

            "wrist_loss":
                wrist_loss.detach(),

            "task_loss":
                task_loss.detach(),

            "config_loss":
                config_loss.detach(),
        }



