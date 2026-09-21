import torch
import torch.nn as nn


class ACTLatentEncoderMLP(nn.Module):
    def __init__(
            self,
            condition_dim=256,
            action_horizon=16,
            action_dim=16,
            latent_dim=32,
    ):
        super().__init__()

        action_flat_dim = action_dim * action_horizon

        self.net = nn.Sequential(
            nn.Linear(
                condition_dim + action_flat_dim,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                256,
            ),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(
            256,
            latent_dim,
        )

        self.logvar_head = nn.Linear(
            256,
            latent_dim,
        )

    def forward(
            self,
            condition,
            action_chunk,
    ):
        B = action_chunk.shape[0]
        action_flat = action_chunk.reshape(B, -1)

        x = torch.cat(
            [
                condition,
                action_flat
            ], dim=1
        )

        h = self.net(x)

        mu =  self.mu_head(h)

        logvar = self.logvar_head(h)

        return mu, logvar

class ACTLatentEncoder(nn.Module):
    def __init__(
            self,
            condition_dim=256,
            action_dim=7,
            action_horizon=16,
            hidden_dim=256,
            latent_dim=32,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
    ):
        super().__init__()

        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim

        # observation / condition -> one token
        self.condition_proj = nn.Linear(
            condition_dim,
            hidden_dim,
        )

        # each action -> one token
        self.action_proj = nn.Linear(
            action_dim,
            hidden_dim,
        )

        # 1 condition token + H action tokens
        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1, 
                action_horizon+1, 
                hidden_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.mu_head = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.logvar_head = nn.Linear(
            hidden_dim,
            latent_dim,
        )

    def forward(
            self,
            condtion,
            action_chunk,
    ):

        # condition:
        # (B, condition_dim)

        # action_chunk:
        # (B, H, action_dim)
    
        B, H, _ = action_chunk.shape
        assert H == self.action_horizon

        # -----------------------------------------
        # condition token
        # -----------------------------------------

        condition_token = self.condition_proj(
            condtion
        ).unsqueeze(1)
        # (B, 1, hidden_dim)

        # -----------------------------------------
        # action tokens
        # -----------------------------------------
        action_tokens = self.action_proj(action_chunk)
        # (B, H, hidden_dim)

        # -----------------------------------------
        # build token sequence
        # -----------------------------------------
        tokens = torch.cat(
            [
                condition_token,
                action_tokens,
            ],
            dim=1
        )
        # (B, H+1, hidden_dim)

        tokens = (
            tokens
            + self.pos_embedding[:, :H+1]
        )
        
        encoded = self.transformer_encoder(tokens) # (B,17,256)

        # use transformed condition/CLS-like token:
        latent_feature = encoded[:, 0]

        mu = self.mu_head(
            latent_feature
        )

        logvar = self.logvar_head(
            latent_feature
        )
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mu, logvar




def reparameterize(
        mu,
        logvar,
):
    std = torch.exp(0.5*logvar)

    eps = torch.randn_like(std)

    z = mu + std*eps

    return z


class ACTDecoder(nn.Module):
    def __init__(
            self,
            agent_feat_dim,
            wrist_feat_dim,
            proprio_feat_dim,
            latent_dim=32,
            hidden_dim=512,
            action_horizon=16,
            action_dim=7,
            num_layers=4,
            num_heads=4,
            dim_feedforward=3200,
            dropout=0.1,
    ):
        super().__init__()

        self.action_horizon=action_horizon
        self.action_dim=action_dim
        self.hidden_dim=hidden_dim

        # condition -> Transformer hidden dimension
        self.agent_proj = nn.Linear(
            agent_feat_dim,
            hidden_dim,
        )

        self.wrist_proj = nn.Linear(
            wrist_feat_dim,
            hidden_dim,
        )

        self.proprio_proj = nn.Linear(
            proprio_feat_dim,
            hidden_dim,
        )

        # latent z -> Transformer hidden dimension
        self.latent_proj = nn.Linear(
            latent_dim,
            hidden_dim,
        )

        # learned query for each predicted action timestep
        self.action_queries = nn.Embedding(
            action_horizon,
            hidden_dim,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.motion_head = nn.Linear(
            hidden_dim,
            action_dim-1
        )

        self.gripper_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
            self,
            agent_feat,
            wrist_feat,
            proprio_feat,
            z,
    ):
        B = agent_feat.shape[0]

        agent_token = self.agent_proj(
            agent_feat
        ).unsqueeze(1)

        wrist_token = self.wrist_proj(
            wrist_feat
        ).unsqueeze(1)

        proprio_token = self.proprio_proj(
            proprio_feat
        ).unsqueeze(1)
        latent_token = self.latent_proj(z).unsqueeze(1)

        # Memory tokens
        # (B, 2, hidden_dim)
        memory = torch.cat(
            [
                agent_token,
                wrist_token,
                proprio_token,
                latent_token,
            ],
            dim=1
        )

        # learned action queries:
        # (H, hidden_dim)
        # learnable table stored inside nn.Embedding.
        queries = self.action_queries.weight

        queries = queries.unsqueeze(0).expand(
            B, -1, -1
        )

        # decoder output:
        # (B, H, hidden_dim)
        h = self.transformer_decoder(
            tgt=queries,
            memory=memory,
        )

        motion = torch.tanh(
            self.motion_head(h)
        )
        gripper = self.gripper_head(h)

        return torch.cat(
            [
                motion,
                gripper,
            ],
            dim=-1
        )


class ACTPolicy(nn.Module):
    def __init__(
        self,
        observation_encoder,
        conditon_dim=256,
        agent_feat_dim=512,
        wrist_feat_dim=512,
        proprio_feat_dim=48,   # 12 * 4
        action_dim=7,
        action_horizon=16,
        hidden_dim=512,
        latent_dim=32,
        num_encoder_layer=4,
        num_decoder_layer=4,
        num_heads=8,
        dim_feedforward=3200,
        dropout=0.1,    
    ):
        super().__init__()

        self.observation_encoder= observation_encoder
        self.action_dim= action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim

        self.latent_encoder = ACTLatentEncoder(
            condition_dim=conditon_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layer,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.decoder = ACTDecoder(
             agent_feat_dim=agent_feat_dim,
            wrist_feat_dim=wrist_feat_dim,
            proprio_feat_dim=proprio_feat_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_horizon=action_horizon,
            action_dim=action_dim,
            num_layers=num_decoder_layer,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
            self,
            agent,
            wrist,
            proprio,
            action_chunk=None,
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

        # ========================================================
        # Training
        # ========================================================
        if action_chunk is not None:
            mu, logvar = self.latent_encoder(condition, action_chunk)

            z = reparameterize(mu, logvar)

        # ========================================================
        # Inference
        # ========================================================
        else:
            B = condition.shape[0]

            z = torch.zeros(
                B,
                self.latent_dim,
                device=condition.device,
                dtype=condition.dtype,
            )

            mu = None, 
            logvar = None

        pred_chunk = self.decoder(
            agent_feat,
            wrist_feat,
            proprio_feat,
            z,
        )


        return (pred_chunk, mu, logvar)