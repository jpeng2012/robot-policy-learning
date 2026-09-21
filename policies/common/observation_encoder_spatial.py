import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEncoder(nn.Module):
    def __init__(
            self,
            agent_encoder,
            wrist_encoder,
            proprio_dim,
            vision_history_len=1,
            proprio_history_len=4,
            condition_dim=256,
    ):
        super().__init__()
        self.agent_encoder = agent_encoder
        self.wrist_encoder = wrist_encoder
        
        self.vision_history_len = vision_history_len
        self.proprio_history_len = proprio_history_len
        self.condition_dim = condition_dim

        self.visual_dim = 512 * vision_history_len
        self.proprio_flat_dim = proprio_dim * proprio_history_len

        input_dim = 2*self.visual_dim + self.proprio_flat_dim

        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, condition_dim),
            nn.ReLU(),
        )

    def _forward_resnet_spatial(self, model, x):
        # x: B,C,H,W
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        # B,512,7,7 for 224x224 input
        return x

    def forward(self, agent, wrist, proprio, return_features=False,):
        # agent: (B, T, C, H, W)
        B, T, C, H, W = agent.shape
        agent = agent.reshape(-1, *agent.shape[2:])

        # wrist: (B, T, C, H, W)
        wrist = wrist.reshape(-1, *wrist.shape[2:])

        # -------------------------------------------------
        # Keep spatial features
        # -------------------------------------------------
        agent_map = self._forward_resnet_spatial(
            self.agent_encoder,
            agent,
        )
        wrist_map = self._forward_resnet_spatial(
            self.wrist_encoder,
            wrist,
        )

        # B*T,512,7,7
        _, D, Hf, Wf = agent_map.shape

        # Preserve time dimension.
        agent_map = agent_map.reshape(
            B, T, D, Hf, Wf
        )

        wrist_map = wrist_map.reshape(
            B, T, D, Hf, Wf
        )

        # -------------------------------------------------
        # Global features are still useful for CVAE condition
        # -------------------------------------------------
        agent_global = F.adaptive_avg_pool2d(
            agent_map.reshape(B*T, D, Hf, Wf),
            1,
        ).flatten(1)

        wrist_global = F.adaptive_avg_pool2d(
            wrist_map.reshape(B*T, D, Hf, Wf),
            1,
        ).flatten(1)

        agent_global = agent_global.reshape(B, T * D)
        wrist_global = wrist_global.reshape(B, T * D)

        proprio_feat = proprio.reshape(B, -1)

        x = torch.cat(
            [
                agent_global,
                wrist_global,
                proprio_feat,
            ],
            dim=1,
        )

        condition = self.projector(x)

        if return_features:
            return (
                condition,
                agent_map,
                wrist_map,
                proprio_feat,
            )

        return condition
