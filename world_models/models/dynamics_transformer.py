from __future__ import annotations

import torch
import torch.nn as nn

import math

def sinusoidal_embedding(
    positions: torch.Tensor,
    dim: int,
):
    """
    positions:
        [T]

    returns:
        [T, dim]
    """

    half_dim = dim // 2

    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(
            half_dim,
            device=positions.device,
            dtype=torch.float32,
        )
        / half_dim
    )

    angles = (
        positions.float().unsqueeze(1)
        * frequencies.unsqueeze(0)
    )

    embedding = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=1,
    )

    if dim % 2 == 1:
        embedding = torch.cat(
            [
                embedding,
                torch.zeros(
                    embedding.shape[0],
                    1,
                    device=embedding.device,
                ),
            ],
            dim=1,
        )

    return embedding

class DynamicsTransformer(nn.Module):
    """
    Action-conditioned latent world-model predictor.

    Inputs:

        state_tokens:
            [B, N, D]

        actions:
            [B, H, action_dim]

    Output:

        predicted_future:
            [B, H, N, D]

    where:

        predicted_future[:, 0]
            predicts z_{t+1}

        predicted_future[:, 1]
            predicts z_{t+2}

        ...

        predicted_future[:, H-1]
            predicts z_{t+H}


    Important causal constraint:

        prediction of z_{t+k}

    can use:

        z_t
        a_t ... a_{t+k-1}

    but cannot use later actions.
    """

    def __init__(
            self,
            latent_dim: int = 384,
            action_dim: int = 7,
            num_state_tokens: int = 34, 
            horizon: int = 16,
            num_layers: int = 6,
            num_heads: int = 6,
            ff_dim: int = 1536,
            dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_state_tokens = num_state_tokens
        self.horizon = horizon

        # ====================================================
        # Action encoder
        # ====================================================
        self.action_projection = nn.Sequential(
            nn.Linear(
                action_dim,  #7
                latent_dim,
            ),
            nn.GELU(),
            nn.Linear(
                latent_dim,
                latent_dim,
            ),
            nn.LayerNorm(
                latent_dim,
            ),
        )

        # ====================================================
        # Future prediction queries
        # ====================================================
        #
        # For every future timestep we need N query tokens.
        #
        # Example:
        #
        #     step t+1 -> 34 queries
        #     step t+2 -> 34 queries
        #     ...
        #
        # query_slot_embedding tells the model WHICH state
        # token this query is predicting.
        #
        # ====================================================
        
        self.query_slot_embedding = nn.Parameter(
            torch.randn(
                num_state_tokens,
                latent_dim,
            ) * 0.02
        )

        # ====================================================
        # Token type embeddings
        # ====================================================
        #
        # 0 = current state token
        # 1 = action token
        # 2 = future query token
        #
        # ====================================================

        self.token_type_embedding = nn.Embedding(
            3,
            latent_dim,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead = num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Sequential(
            nn.Linear(
                latent_dim,
                latent_dim,
            ),
            nn.LayerNorm(
                latent_dim,
            ),
        )


    def _build_attention_mask(
            self,
            device: torch.device,
    ):
        """
        Build causal attention mask.

        Sequence layout:

            [state tokens]

            [a_0]
            [a_1]
            ...
            [a_H-1]

            [queries for future step 1]
            [queries for future step 2]
            ...
            [queries for future step H]


        PyTorch bool attention mask:

            False = attention allowed
            True  = attention blocked
        """

        N = self.num_state_tokens
        H = self.horizon

        # Number of tokens:
        #
        #     N state tokens
        #   + H action tokens
        #   + H*N query tokens

        total_tokens = N + H + N * H

        mask = torch.ones(
            total_tokens,
            total_tokens,
            dtype=torch.bool,
            device=device,
        )
        
        state_start = 0
        state_end = N

        action_start = N
        action_end = N + H

        query_start = N + H

         # ====================================================
        # 1. State tokens
        #
        # Current-state tokens only attend to current state.
        #
        # They do NOT need future actions or future queries.
        # ====================================================
        mask[
            state_start:state_end,
            state_start:state_end,
        ] = False

        # ====================================================
        # 2. Action tokens
        #
        # Action i can attend:
        #
        #     all current state tokens
        #
        # and:
        #
        #     action 0 ... action i
        #
        # This makes the action sequence causal too.
        # ====================================================

        for i in range(H):
            row = action_start + i

            # Current state
            mask[
                row,
                state_start:state_end,
            ] = False

            # Current + previous actions

            mask[
                row,
                action_start: action_start + i + 1,
            ] = False

        # ====================================================
        # 3. Future query tokens
        #
        # Future step k predicts:
        #
        #     z_{t+k+1}
        #
        # Therefore it may use:
        #
        #     a_0 ... a_k
        #
        # but NOT:
        #
        #     a_{k+1} ...
        #
        # ====================================================

        for k in range(H):
            q_start = query_start + k * N
            q_end = q_start + N

            # -----------------------------------------------
            # Every future query can see current state.
            # -----------------------------------------------

            mask[
                q_start:q_end,
                state_start:state_end
            ] = False

            # -----------------------------------------------
            # Future state t+k+1 sees actions:
            #
            #     a_0 ... a_k
            #
            # -----------------------------------------------

            mask[
                q_start:q_end,
                action_start:action_start + k + 1,
            ] = False

            # -----------------------------------------------
            # Allow queries to communicate with queries
            # from previous and current future steps.
            #
            # But NOT future query steps.
            #
            # -----------------------------------------------

            mask[
                q_start:q_end,
                query_start:q_end,
            ] = False

        return mask


    def _build_future_queries(
            self,
            batch_size: int,
            device: torch.device,
    ):
        
        """
        Construct future query tokens.

        Returns:

            [B, H, N, D]
        """

        H = self.horizon
        N = self.num_state_tokens
        D = self.latent_dim

        # ====================================================
        # State-slot identity
        # ====================================================
        #
        # Shape:
        #
        #     [1, 1, N, D]
        #
        # ====================================================

        slot = self.query_slot_embedding.view(1, 1, N, D)

        # ====================================================
        # Future timestep identity
        # ====================================================

        time_ids = torch.arange(1, H+1, device=device)

        time = sinusoidal_embedding(
                time_ids,
                self.latent_dim,
        )

        time = time.view(1, H, 1, D)

        queries = slot + time
        queries = queries.expand(batch_size, -1, -1, -1)

        return queries


    def forward(
            self,
            state_tokens: torch.Tensor,
            actions: torch.Tensor,
    ):
        """
        Inputs:

            state_tokens:
                [B, N, D]

            actions:
                [B, H, action_dim]

        Returns:

            predicted_future:
                [B, H, N, D]
        """

        B, N, D = state_tokens.shape

        # ====================================================
        # Validate input shapes
        # ====================================================

        if N != self.num_state_tokens:
            raise ValueError(
                f"Expected "
                f"{self.num_state_tokens} "
                f"state tokens, got {N}"
            )

        if D != self.latent_dim:
            raise ValueError(
                f"Expected latent dim "
                f"{self.latent_dim}, "
                f"got {D}"
            )

        if actions.shape[1] != self.horizon:
            raise ValueError(
                f"Expected action horizon "
                f"{self.horizon}, "
                f"got {actions.shape[1]}"
            )

        # ====================================================
        # Encode action sequence
        # ====================================================

        action_tokens = self.action_projection(actions)

        action_time_ids = torch.arange(
            self.horizon,
            device=actions.device,
        )

        action_time = sinusoidal_embedding(
                action_time_ids,
                self.latent_dim,
        )

        action_tokens = action_tokens + action_time.unsqueeze(0)

        future_queries_flat = self._build_future_queries(
            batch_size=B,
            device=state_tokens.device,
        )

        # [B, H, N, D]
        # ->
        # [B, H*N, D]

        future_queries_flat = future_queries_flat.reshape(
            B, -1, D
        )

        state_tokens = (
            state_tokens
            + self.token_type_embedding.weight[
                0
            ]
        )

        action_tokens = (
            action_tokens
            + self.token_type_embedding.weight[
                1
            ]
        )

        future_queries_flat = (
            future_queries_flat
            + self.token_type_embedding.weight[
                2
            ]
        )

        sequence = torch.cat(
            [
                state_tokens,
                action_tokens,
                future_queries_flat,
            ],
            dim=1,
        )

        attention_mask = self._build_attention_mask(
            device=sequence.device,
        )

        output = self.transformer(
            sequence,
            mask=attention_mask,
        )

        query_start = self.num_state_tokens + self.horizon

        prediction = output[
            :,
            query_start:,
            :
        ]

        prediction = self.output_head(prediction)

        prediction = prediction.reshape(
            B,
            self.horizon,
            self.num_state_tokens,
            self.latent_dim,
        )

        return prediction


