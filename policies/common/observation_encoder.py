import torch
import torch.nn as nn

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

    def forward(self, agent, wrist, proprio, return_features=False,):
        # agent: (B, T, C, H, W)
        B, T, C, H, W = agent.shape
        agent = agent.reshape(-1, *agent.shape[2:])

        # wrist: (B, T, C, H, W)
        wrist = wrist.reshape(-1, *wrist.shape[2:])

        agent_feat = self.agent_encoder(agent).reshape(-1, T*512)
        wrist_feat = self.wrist_encoder(wrist).reshape(-1, T*512)

        proprio_feat = proprio.reshape(B, -1)

        x = torch.cat([agent_feat, wrist_feat, proprio_feat], dim=1)

        condition = self.projector(x)

        if return_features:
            return (
                condition,
                agent_feat,
                wrist_feat,
                proprio_feat,
            )

        return condition