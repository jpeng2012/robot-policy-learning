import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from policies.vision_bc.dataset import VisionChunkBCDataset
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
import torch.nn.functional as F

from policies.common.observation_encoder_spatial import ObservationEncoder
from policies.flow_matching.model import FlowMatchingPolicy

# ============================================================
# Config
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 20

vision_history_len = 1
proprio_history_len = 4

proprio_dim = 12      # joint 7 + eef 3 + gripper 2

action_horizon = 16
action_dim = 7

condition_dim = 256
hidden_dim = 512


num_encoder_layer=4
num_decoder_layer=4
num_heads=8

dim_feedforward=2048

num_workers = 8
kl_weight = 1.0


encoder_lr = 2e-5
projector_lr = 1e-4
decoder_lr = 1e-4

agent_feat_dim = 512 # ? 
wrist_feat_dim = 512
proprio_feat_dim = proprio_dim * proprio_history_len

dataset_path = "data/level3_hist_1_4_dataset.npz"
save_path0 = "policies/flow_matching/flowmatching_1_4"

STATE_TO_ID = {
    "APPROACH": 0,
    "DESCEND": 1,
    "GRASP": 2,
    "LIFT": 3,
    "TRANSPORT": 4,
    "LOWER": 5,
    "RELEASE": 6,
}

STATE_WEIGHT = torch.tensor([
    1.0,   # APPROACH
    1.0,   # DESCEND
    1.0,   # GRASP
    1.0,   # LIFT
    0.5,   # TRANSPORT
    1.0,   # LOWER
    2.0,   # RELEASE
], dtype=torch.float32, device=device)

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

data = np.load(dataset_path)

proprio_train = data["proprio_train"].astype(np.float32)
proprio_mean = np.mean(proprio_train, axis=(0, 1))
proprio_std = np.std(proprio_train, axis=(0, 1))+1e-6

train_dataset = VisionChunkBCDataset(
    dataset_npz=dataset_path,
    data_root="data/level3",
    split="train",
    transform=transform,
    proprio_mean=proprio_mean,
    proprio_std=proprio_std,
)

val_dataset = VisionChunkBCDataset(
    dataset_npz=dataset_path,
    data_root="data/level3",
    split="val",
    transform=transform,
    proprio_mean=proprio_mean,
    proprio_std=proprio_std,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

agent_encoder = resnet18(weights=weights)

wrist_encoder = resnet18(weights=weights)

agent_encoder.fc = nn.Identity()
wrist_encoder.fc = nn.Identity()

for p in agent_encoder.parameters():
    p.requires_grad = False

for p in wrist_encoder.parameters():
    p.requires_grad = False

# for p in agent_encoder.layer3.parameters():
#     p.requires_grad = True

# for p in wrist_encoder.layer3.parameters():
#     p.requires_grad = True

for p in agent_encoder.layer4.parameters():
    p.requires_grad = True

for p in wrist_encoder.layer4.parameters():
    p.requires_grad = True

obs_encoder = ObservationEncoder(
    agent_encoder=agent_encoder,
    wrist_encoder=wrist_encoder,
    proprio_dim=proprio_dim,
    vision_history_len=vision_history_len,
    proprio_history_len=proprio_history_len,
    condition_dim=condition_dim,
).to(device)


policy = FlowMatchingPolicy(
    observation_encoder=obs_encoder,
    agent_feat_dim=agent_feat_dim,
    wrist_feat_dim=wrist_feat_dim,
    proprio_feat_dim=proprio_feat_dim,
    action_dim=action_dim,
    action_horizon=action_horizon,
    hidden_dim=hidden_dim,
    num_layers=num_decoder_layer,
    num_heads=num_heads,
    dim_feedforward=dim_feedforward,
).to(device)

# fine-tune the ResNets, but with a lower LR on the encoders than on the MLP head:
optimizer = torch.optim.AdamW([
    {
        "params": obs_encoder.agent_encoder.layer4.parameters(),
        "lr": encoder_lr,
    },
    {
        "params": obs_encoder.wrist_encoder.layer4.parameters(),
        "lr": encoder_lr,
    },
    {
        "params": obs_encoder.projector.parameters(),
        "lr": projector_lr,
    },
    {
        "params": policy.flow_decoder.parameters(),
        "lr": decoder_lr,
    }
])

best_eval_loss = float('inf')


for epoch in range(num_epochs):

    torch.cuda.reset_peak_memory_stats()

    policy.train()

    # Keep frozen ResNet parts' BatchNorm stats fixed
    obs_encoder.agent_encoder.eval()
    obs_encoder.wrist_encoder.eval()

    # Fine-tuned final stage
    # obs_encoder.agent_encoder.layer4.train()
    # obs_encoder.wrist_encoder.layer4.train()

    train_loss = 0.0

    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()

        agent = batch["agent"].to(device)
        wrist = batch["wrist"].to(device)
        proprio = batch["proprio"].to(device)
        target_chunk = batch["action"].to(device)
        state_id = batch["state"].to(device)

        weights = STATE_WEIGHT[state_id]

        B = target_chunk.shape[0]

        # ------------------------------------------------------
        # x1 = expert action trajectory
        # ------------------------------------------------------

        x1 = target_chunk[..., :action_dim-1]

        gripper_target = (
            target_chunk[..., action_dim-1] > 0
        ).float()
        # print(gripper_target.shape)

        # ------------------------------------------------------
        # x0 = random Gaussian action trajectory
        # ------------------------------------------------------

        x0 = torch.randn_like(x1)

        # ------------------------------------------------------
        # random flow time
        # ------------------------------------------------------

        t = torch.rand(
            B,
            device=x1.device,
        )

        t_expand = t[:, None, None]

        # ------------------------------------------------------
        # interpolation
        #
        # t=0 -> noise
        # t=1 -> expert
        # ------------------------------------------------------

        xt = (
            (1.0 - t_expand) * x0
            + t_expand * x1
        )

        # ------------------------------------------------------
        # velocity target
        # ------------------------------------------------------

        target_velocity = x1 - x0

        # ------------------------------------------------------
        # model
        # ------------------------------------------------------

        pred_velocity, gripper_logits = policy(
            agent,
            wrist,
            proprio,
            noisy_actions=xt,
            t=t,
        )

        # ------------------------------------------------------
        # basic FM loss
        # ------------------------------------------------------

        motion_loss = F.mse_loss(
            pred_velocity,
            target_velocity,
            reduction="none",
        ).mean(dim=(1, 2))

        gripper_loss = F.binary_cross_entropy_with_logits(
            gripper_logits,
            gripper_target,
            reduction="none",
        ).mean(1)
        
        loss_per_sample = motion_loss + 1.1*gripper_loss

        loss = (loss_per_sample * weights).mean()
       
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * agent.size(0)

        err = (
            pred_velocity - target_velocity
        ).pow(2)

        xyz_loss = err[..., :3].mean()
        rot_loss = err[..., 3:6].mean()

        x_loss = err[..., 0].mean()
        y_loss = err[..., 1].mean()
        z_loss = err[..., 2].mean()

        if batch_idx % 100 == 0:
            print(
                f"epoch={epoch+1} "
                f"batch={batch_idx}/{len(train_loader)} "
                f"loss={loss.item():.5f} "
                f"motion={motion_loss.mean().item():.5f} "
                f"gripper={gripper_loss.mean().item():.5f} "
                f"xyz={xyz_loss.item():.5f} "
                f"rot={rot_loss.item():.5f} "
                f"x={x_loss.item():.5f} "
                f"y={y_loss.item():.5f} "
                f"z={z_loss.item():.5f} "
                f"batch_time={time.time()-batch_start:.2f}s")

    train_loss /= len(train_dataset)

    # Validation
    policy.eval()
    obs_encoder.eval()
    val_loss = 0.0

    generator = torch.Generator(
        device=device
    ).manual_seed(1234)

    with torch.no_grad():
        for batch in val_loader:
            agent = batch["agent"].to(device, non_blocking=True,)
            wrist = batch["wrist"].to(device, non_blocking=True,)
            proprio = batch["proprio"].to(device, non_blocking=True,)
            target_chunk = batch["action"].to(device, non_blocking=True,)
            state_id = batch["state"].to(device, non_blocking=True,)

            weights = STATE_WEIGHT[state_id]

            B = target_chunk.shape[0]
            
            # ------------------------------------------------------
            # x1 = expert action trajectory
            # ------------------------------------------------------
    
            x1 = target_chunk[..., :action_dim-1]
            
            gripper_target = (
                target_chunk[..., action_dim-1] > 0
            ).float()
    
            # ------------------------------------------------------
            # x0 = random Gaussian action trajectory
            # ------------------------------------------------------
    
            

            x0 = torch.randn(
                x1.shape,
                device=x1.device,
                generator=generator,
            )
    
            # ------------------------------------------------------
            # random flow time
            # ------------------------------------------------------
    
            t = torch.rand(
                B,
                device=x1.device,
                generator=generator,
            )
    
            t_expand = t[:, None, None]
    
            # ------------------------------------------------------
            # interpolation
            #
            # t=0 -> noise
            # t=1 -> expert
            # ------------------------------------------------------
    
            xt = (
                (1.0 - t_expand) * x0
                + t_expand * x1
            )
    
            # ------------------------------------------------------
            # velocity target
            # ------------------------------------------------------
    
            target_velocity = x1 - x0
    
            # ------------------------------------------------------
            # model
            # ------------------------------------------------------
    
            pred_velocity, gripper_logits = policy(
                agent,
                wrist,
                proprio,
                noisy_actions=xt,
                t=t,
            )
    
            # ------------------------------------------------------
            # basic FM loss
            # ------------------------------------------------------
    
            motion_loss = F.mse_loss(
                pred_velocity,
                target_velocity,
                reduction="none",
            ).mean(dim=(1, 2))
    
            gripper_loss = F.binary_cross_entropy_with_logits(
                gripper_logits,
                gripper_target,
                reduction="none",
            ).mean(1)
            
            loss_per_sample = motion_loss + 1.1*gripper_loss
    
            loss = (loss_per_sample * weights).mean()
            val_loss += loss.item() * agent.size(0)

    val_loss /= len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} time="
        f"{time.time()-t0:.1f}s")

    print(
        torch.cuda.max_memory_allocated() / 1024**3,
        "GB allocated"
    )

    print(
        torch.cuda.max_memory_reserved() / 1024**3,
        "GB reserved"
    )

    # if val_loss < best_eval_loss:
    if ((epoch+1) % 4) == 0:
            save_path = f"{save_path0}_ep{epoch:03d}.pth"
    
            best_eval_loss = val_loss
    
            torch.save(
                {    
                    "policy_state_dict":
                        policy.state_dict(),
    
                    "proprio_mean":
                        proprio_mean,
    
                    "proprio_std":
                        proprio_std,
    
                    "vision_history_len":
                        vision_history_len,
    
                    "proprio_history_len":
                        proprio_history_len,
    
                    "proprio_dim":
                        proprio_dim,
    
                    "action_horizon":
                        action_horizon,
    
                    "action_dim":
                        action_dim,

                    "condition_dim":
                        condition_dim,
    
                    "hidden_dim":
                        hidden_dim,

                    "num_encoder_layer":
                        num_encoder_layer,

                    "num_decoder_layer":
                        num_decoder_layer,

                    "num_heads":
                        num_heads,

                    "dim_feedforward":
                        dim_feedforward,
    
                    "val_loss":
                        val_loss,
                },
                save_path,
            )
    
            print(
                "saved best:",
                save_path,
            )

# torch.save(
#     {
#         "model_state_dict": policy.state_dict(),
#         "proprio_mean": proprio_mean,
#         "proprio_std": proprio_std,
#         "vision_history_len": vision_history_len,
#         "proprio_history_len": proprio_history_len,
#     }, save_path
# )

# print(f"Model saved to {save_path}")