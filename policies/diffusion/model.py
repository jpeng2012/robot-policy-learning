import torch
import torch.nn as nn

from policies.diffusion.diffusion_utils import SinusoidalPosEmb

class ActionDenoiser(nn.Module):
    def __init__(
            self,
            condition_dim,
            action_horizon = 16,
            action_dim = 7,
            time_dim = 64,
            hidden_dim = 512,
    ):
        super().__init__()

        self.action_horizon = action_horizon
        self.action_dim = action_dim   

        self.action_flat_dim = action_horizon * action_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        input_dim = self.action_flat_dim + condition_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_flat_dim),
        )

    def forward(self, noisy_action, timestep, condition):
        B = noisy_action.shape[0]
        noisy_action = noisy_action.reshape(B, -1)
        time_emb = self.time_embedding(timestep)
        x = torch.cat([noisy_action, condition, time_emb], dim=1)

        eps_pred = self.net(x)
        
        return eps_pred.reshape(B, self.action_horizon, self.action_dim)


