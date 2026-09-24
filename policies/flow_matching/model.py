import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.common.position_embedding import PositionEmbeddingSine2D

import math

# ============================================================
# Continuous flow-time embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        t: (B,) continuous values in [0, 1]

        return:
            (B, dim)
        """

        half_dim = self.dim // 2

        # t in [0,1] is too small for ordinary transformer-style
        # sinusoidal PE, so scale it to a diffusion-like range.
        t = t * 1000.0

        freq = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                half_dim, device=t.device, dtype=t.dtype
            ) / half_dim
        )

        angles = t[:, None] * freq[None,:]

        emb = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1
        )

        if self.dim % 2:
            emb = F.pad(emb, (0, 1))

        return emb


class FlowMatchingDecoder(nn.Module):
    def __init__(
            self,
             agent_feat_dim=512,
            wrist_feat_dim=512,
            proprio_feat_dim=48,
            action_dim=7,
            action_horizon=16,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            dim_feedforward=2048,
            dropout=0.1,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim

        # --------------------------------------------------
        # Image projections
        # --------------------------------------------------

        self.agent_proj = nn.Conv2d(
            agent_feat_dim,
            hidden_dim,
            kernel_size=1,
        )

        self.wrist_proj = nn.Conv2d(
            wrist_feat_dim,
            hidden_dim,
            kernel_size=1,
        )

        self.image_pos_embedding = PositionEmbeddingSine2D(
            hidden_dim=hidden_dim,
        )

        # identifies agent vs wrist camera
        self.camera_embedding = nn.Embedding(
            2,
            hidden_dim,
        )

        # -----------------------------------------
        # noisy action → token
        # -----------------------------------------

        self.action_proj = nn.Linear(
            action_dim-1,
            hidden_dim,
        )

        self.action_pos_embedding = nn.Parameter(
            torch.zeros(
                1, action_horizon, hidden_dim)
        )

        nn.init.normal_(
            self.action_pos_embedding,
            std=0.02,
        )

        # -----------------------------------------
        # Flow time t
        # -----------------------------------------

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -----------------------------------------
        # proprio
        # -----------------------------------------

        self.proprio_proj = nn.Linear(
            proprio_feat_dim,
            hidden_dim,
        )

        # -----------------------------------------
        # Transformer
        # -----------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.velocity_head = nn.Linear(
            hidden_dim,
            action_dim-1,
        )

        self.gripper_head = nn.Linear(
            hidden_dim,
            1,
        )

    def _image_tokens(
        self,
        feat,
        projection,
        camera_id,
    ):
        """
        feat:
            B,T,C,H,W

        returns:
            B,(T*H*W),hidden_dim
        """

        B, T, C, H, W = feat.shape

        feat = feat.reshape(
            B * T,
            C,
            H,
            W,
        )

        feat = projection(feat)

        # B*T,D,H,W
        pos = self.image_pos_embedding(feat)

        feat = feat + pos

        # Camera identity
        cam_id = torch.tensor(
            camera_id,
            device=feat.device,
        )

        cam_emb = self.camera_embedding(cam_id)

        feat = feat + cam_emb[None, :, None, None]

        D = feat.shape[1]

        feat = (
            feat.flatten(2)
            .transpose(1, 2)
        )
        # B*T, H*W, D

        feat = feat.reshape(
            B,
            T * H * W,
            D,
        )

        return feat

    
    def forward(
            self, 
            agent_feat,
            wrist_feat,
            proprio_feat,
            noisy_actions,
            t,
        ):

        """
        agent_feat:
            B,T,512,7,7

        wrist_feat:
            B,T,512,7,7

        proprio_feat:
            B,48

        noisy_actions:
            B,H,7

        t:
            B,
        """

        B, H, _ = noisy_actions.shape

        assert H == self.action_horizon

        # -----------------------------------------
        # observation memory
        # -----------------------------------------

        agent_tokens = self._image_tokens(
            agent_feat,
            self.agent_proj,
            camera_id=0,
        )

        wrist_tokens = self._image_tokens(
            wrist_feat,
            self.wrist_proj,
            camera_id=1,
        )


        proprio_token = self.proprio_proj(proprio_feat).unsqueeze(1)

        memory = torch.cat(
            [
                agent_tokens,
                wrist_tokens,
                proprio_token,
            ], 
            dim=1,
        )

        # ==================================================
        # Current flow state x_t
        # ==================================================

        action_tokens = self.action_proj(noisy_actions)

        action_tokens = action_tokens + self.action_pos_embedding[:, :H]

        # -----------------------------------------
        # time conditioning
        # -----------------------------------------
        time_emb = self.time_embedding(t)

        # same t applies to every action token
        action_tokens = action_tokens + time_emb.unsqueeze(1)

        # -----------------------------------------
        # observation-conditioned flow
        # -----------------------------------------

        h = self.transformer(
            tgt=action_tokens,
            memory=memory,
        )

        velocity = self.velocity_head(h)
        gripper_logits = self.gripper_head(h)

        return velocity, gripper_logits.squeeze(-1)


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        observation_encoder,
        agent_feat_dim=512,
        wrist_feat_dim=512,
        proprio_feat_dim=48,
        action_dim=7,
        action_horizon=16,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.observation_encoder = observation_encoder

        self.action_dim = action_dim
        self.action_horizon = action_horizon

        self.flow_decoder = FlowMatchingDecoder(
            agent_feat_dim=agent_feat_dim,
            wrist_feat_dim=wrist_feat_dim,
            proprio_feat_dim=proprio_feat_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def encode_observation(
        self,
        agent,
        wrist,
        proprio,
    ):
        (
            condition,
            agent_feat,
            wrist_feat,
            proprio_feat,
        ) = self.observation_encoder(
            agent,
            wrist,
            proprio,
            return_features=True,
        )

        return (
            agent_feat,
            wrist_feat,
            proprio_feat,
        )

    def forward(
        self,
        agent,
        wrist,
        proprio,
        noisy_actions,
        t,
    ):
        (
            agent_feat,
            wrist_feat,
            proprio_feat,
        ) = self.encode_observation(
            agent,
            wrist,
            proprio,
        )

        velocity, gripper_logits = self.flow_decoder(
            agent_feat=agent_feat,
            wrist_feat=wrist_feat,
            proprio_feat=proprio_feat,
            noisy_actions=noisy_actions,
            t=t,
        )

        return velocity, gripper_logits


    @torch.no_grad()
    def sample_actions(
        self,
        agent,
        wrist,
        proprio,
        num_steps=10,
    ):
        self.eval()

        (
            agent_feat,
            wrist_feat,
            proprio_feat,
        ) = self.encode_observation(
            agent,
            wrist,
            proprio,
        )

        B = agent.shape[0]

        # t = 0
        # start with Gaussian noise
        x = torch.randn(
            B,
            self.action_horizon,
            self.action_dim-1,
            device=agent.device,
            dtype=agent.dtype,
        )

        dt = 1.0 / num_steps

        for i in range(num_steps):

            t_value = i / num_steps

            t = torch.full(
                (B,),
                t_value,
                device=x.device,
                dtype=x.dtype,
            )

            velocity, _ = self.flow_decoder(
                agent_feat=agent_feat,
                wrist_feat=wrist_feat,
                proprio_feat=proprio_feat,
                noisy_actions=x,
                t=t,
            )

            # Euler integration
            x = x + dt * velocity

        # --------------------------------------------------
        # Convert generated trajectory into valid robot action
        # --------------------------------------------------

        x = torch.clamp(
            x,
            -1.0,
            1.0,
        )

        _, gripper_logits = self.flow_decoder(
            agent_feat=agent_feat,
            wrist_feat=wrist_feat,
            proprio_feat=proprio_feat,
            noisy_actions=x,
            t=torch.ones(B, device=x.device),
        )

        gripper = torch.where(
            gripper_logits > 0,
            1.0,
            -1.0,
        )

        actions = torch.cat(
            [
                x,
                gripper.unsqueeze(-1),
            ],
            dim=-1,
        )
        return actions        


    @torch.no_grad()
    def sample_actions_with_x(
        self,
        agent,
        wrist,
        proprio,
        x,
        num_steps=10,
    ):
        self.eval()

        (
            agent_feat,
            wrist_feat,
            proprio_feat,
        ) = self.encode_observation(
            agent,
            wrist,
            proprio,
        )

        B = agent.shape[0]

        dt = 1.0 / num_steps

        for i in range(num_steps):

            t_value = i / num_steps

            t = torch.full(
                (B,),
                t_value,
                device=x.device,
                dtype=x.dtype,
            )

            velocity, _ = self.flow_decoder(
                agent_feat=agent_feat,
                wrist_feat=wrist_feat,
                proprio_feat=proprio_feat,
                noisy_actions=x,
                t=t,
            )

            # Euler integration
            x = x + dt * velocity

        # --------------------------------------------------
        # Convert generated trajectory into valid robot action
        # --------------------------------------------------

        x = torch.clamp(
            x,
            -1.0,
            1.0,
        )

        _, gripper_logits = self.flow_decoder(
            agent_feat=agent_feat,
            wrist_feat=wrist_feat,
            proprio_feat=proprio_feat,
            noisy_actions=x,
            t=torch.ones(B, device=x.device),
        )

        gripper = torch.where(
            gripper_logits > 0,
            1.0,
            -1.0,
        )

        actions = torch.cat(
                [
                    x,
                    gripper.unsqueeze(-1),
                ],
                dim=-1,
        )
        return actions                                                                     