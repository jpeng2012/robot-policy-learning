from __future__ import annotations

import torch
import torch.nn as nn

from .visual_encoder import SpatialViTEncoder

class ObservationEncoder(nn.Module):
    """
    Encode one robot observation into a set of latent tokens.

    Observation:

        agent camera
        wrist camera
        task-space robot state
        joint configuration

    Output:

        latent tokens:

            [B, N, latent_dim]

    Default configuration:

        agent visual tokens:    16
        wrist visual tokens:    16
        task-state token:        1
        robot-config token:      1

        total:                   34
    """

    def __init__(
            self,
            latent_dim: int = 384,
            latent_grid_size: int = 4,
            task_state_dim: int = 9,
            robot_config_dim: int = 7,
            pretrained_vision: bool = True,
            freeze_vision = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # ====================================================
        # Visual encoder
        # ====================================================
        self.visual_encoder = SpatialViTEncoder(
            latent_dim=latent_dim,
            latent_grid_size=latent_grid_size,
            pretrained=pretrained_vision,
            freeze_backbone=freeze_vision,
        )

        # ====================================================
        # Robot state projections
        # ====================================================
        self.task_state_projection = nn.Sequential(
            nn.Linear(
                task_state_dim,
                latent_dim,
            ),
            nn.GELU(),
            nn.Linear(
                latent_dim,
                latent_dim,
            ),
            nn.LayerNorm(
                latent_dim
            ),
        )

        self.robot_config_projection = nn.Sequential(
            nn.Linear(
                robot_config_dim,
                latent_dim,
            ),
            nn.GELU(),
            nn.Linear(
                latent_dim,
                latent_dim,
            ),
            nn.LayerNorm(
                latent_dim
            ),
        )

        # ====================================================
        # Token-type embeddings
        # ====================================================
        #
        # Tell the Transformer where each token came from:
        #
        #     0 -> agent camera
        #     1 -> wrist camera
        #     2 -> task state
        #     3 -> robot configuration
        #
        # ====================================================

        self.token_type_embedding = nn.Embedding(
            4,
            latent_dim,
        )


    def forward(
            self,
            agent_images: torch.Tensor,
            wrist_images: torch.Tensor,
            task_state: torch.Tensor,
            robot_config: torch.Tensor,
    ):
        """
        Inputs:

            agent_images:
                [B, 3, H, W]

            wrist_images:
                [B, 3, H, W]

            task_state:
                [B, 9]

            robot_config:
                [B, 7]

        Returns:

            tokens:
                [B, N, latent_dim]
        """

        agent_token = self.visual_encoder(agent_images)
        wrist_token = self.visual_encoder(wrist_images)
        # [B, 16, 384]

        task_token = self.task_state_projection(task_state).unsqueeze(dim=1)
        config_token = self.robot_config_projection(robot_config).unsqueeze(dim=1)

        # ====================================================
        # Add token-type information
        # ====================================================

        agent_type = self.token_type_embedding.weight[0]
        wrist_type = self.token_type_embedding.weight[1]
        task_type = self.token_type_embedding.weight[2]
        config_type = self.token_type_embedding.weight[3]

        agent_token = agent_token + agent_type
        wrist_token = wrist_token + wrist_type
        task_token = task_token + task_type
        config_token = config_token + config_type

        tokens = torch.cat(
            [
                agent_token,
                wrist_token,
                task_token,
                config_token,
            ], 
            dim=1,
        )

        return tokens



    def forward_features(
            self,
            agent_features: torch.Tensor,
            wrist_features: torch.Tensor,
            task_state: torch.Tensor,
            robot_config: torch.Tensor,
        ):
            
            """
            Encode precomputed frozen ViT features.
    
            Inputs:
    
                agent_features:
                    [B, 16, 768]
    
                wrist_features:
                    [B, 16, 768]
    
                task_state:
                    [B, 9]
    
                robot_config:
                    [B, 7]
    
            Output:
    
                tokens:
                    [B, 34, latent_dim]
            """
            agent_token = self.visual_encoder.project_features(agent_features)
            wrist_token = self.visual_encoder.project_features(wrist_features)
            # [B, 16, 384]

            task_token = self.task_state_projection(task_state).unsqueeze(dim=1)
            config_token = self.robot_config_projection(robot_config).unsqueeze(dim=1)
    
            # ====================================================
            # Add token-type information
            # ====================================================
            agent_type = self.token_type_embedding.weight[0]
            wrist_type = self.token_type_embedding.weight[1]
            task_type = self.token_type_embedding.weight[2]
            config_type = self.token_type_embedding.weight[3]
    
            agent_token = agent_token + agent_type
            wrist_token = wrist_token + wrist_type
            task_token = task_token + task_type
            config_token = config_token + config_type
    
            tokens = torch.cat(
                [
                    agent_token,
                    wrist_token,
                    task_token,
                    config_token,
                ], 
                dim=1,
            )
    
            return tokens


class ObservationEncoderVjepa(nn.Module):
    """
    Encode one robot observation into a set of latent tokens.

    Observation:

        agent camera
        wrist camera
        task-space robot state
        joint configuration

    Output:

        latent tokens:

            [B, N, latent_dim]

    Default configuration:

        agent visual tokens:    16
        wrist visual tokens:    16
        task-state token:        1
        robot-config token:      1

        total:                   34
    """

    def __init__(
            self,
            visual_feature_dim: int = 768,
            latent_dim: int = 384,
            task_state_dim: int = 9,
            robot_config_dim: int = 7,
    ):
        super().__init__()

        self.visual_feature_dim = visual_feature_dim
        self.latent_dim = latent_dim

        # ====================================================
        # V-JEPA feature adapter
        #
        # V-JEPA representation itself is frozen.
        #
        # This projection exists only on the INPUT side of
        # the dynamics model.
        #
        # It does NOT define the prediction target.
        # ====================================================

        self.visual_projection = nn.Sequential(
             nn.Linear(
                  visual_feature_dim,
                  latent_dim
             ),
             nn.LayerNorm(latent_dim),
        )

        # ====================================================
        # Robot state projections
        # ====================================================
        
        self.task_state_projection = nn.Sequential(
            nn.Linear(
                task_state_dim,
                latent_dim,
            ),
            nn.GELU(),
            nn.Linear(
                latent_dim,
                latent_dim,
            ),
            nn.LayerNorm(
                latent_dim
            ),
        )

        self.robot_config_projection = nn.Sequential(
            nn.Linear(
                robot_config_dim,
                latent_dim,
            ),
            nn.GELU(),
            nn.Linear(
                latent_dim,
                latent_dim,
            ),
            nn.LayerNorm(
                latent_dim
            ),
        )

        # ====================================================
        # Token-type embeddings
        # ====================================================
        #
        # Tell the Transformer where each token came from:
        #
        #     0 -> agent camera
        #     1 -> wrist camera
        #     2 -> task state
        #     3 -> robot configuration
        #
        # ====================================================
        self.token_type_embedding = nn.Embedding(
            4,
            latent_dim,
        )

    def forward_features(
                self,
                agent_features: torch.Tensor,
                wrist_features: torch.Tensor,
                task_state: torch.Tensor,
                robot_config: torch.Tensor,
            ):
                
                """
                Encode precomputed frozen ViT features.
        
                Inputs:
        
                    agent_features:
                        [B, 16, 768]
        
                    wrist_features:
                        [B, 16, 768]
        
                    task_state:
                        [B, 9]
        
                    robot_config:
                        [B, 7]
        
                Output:
        
                    tokens:
                        [B, 34, latent_dim]
                """

                """
                Build current state tokens.

                V-JEPA features remain fixed inputs.

                The projection into Transformer dimension is trainable.
                """

                agent_token = self.visual_projection(agent_features)
                wrist_token = self.visual_projection(wrist_features)
                # [B, 16, 384]
    
                task_token = self.task_state_projection(task_state).unsqueeze(dim=1)
                config_token = self.robot_config_projection(robot_config).unsqueeze(dim=1)
        
                # ====================================================
                # Add token-type information
                # ====================================================
                agent_type = self.token_type_embedding.weight[0]
                wrist_type = self.token_type_embedding.weight[1]
                task_type = self.token_type_embedding.weight[2]
                config_type = self.token_type_embedding.weight[3]
        
                agent_token = agent_token + agent_type
                wrist_token = wrist_token + wrist_type
                task_token = task_token + task_type
                config_token = config_token + config_type
        
                tokens = torch.cat(
                    [
                        agent_token,
                        wrist_token,
                        task_token,
                        config_token,
                    ], 
                    dim=1,
                )
        
                return tokens