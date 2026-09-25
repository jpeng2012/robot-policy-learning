import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from policies.vision_bc.dataset import VisionChunkBCDataset
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from policies.common.observation_encoder_spatial import ObservationEncoder
from policies.diffusion.model import DiffusionPolicy
from policies.diffusion.diffusion_utils import DiffusionSchedule

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

num_decoder_layer = 4
num_heads = 8

dim_feedforward=2048

diffusion_steps = 50

schedule_type = "cosine"

num_workers = 8

encoder_lr = 2e-5
projector_lr = 1e-4
denoiser_lr = 1e-4
agent_feat_dim = 512 # ? 
wrist_feat_dim = 512
proprio_feat_dim = proprio_dim * proprio_history_len

dataset_path = "data/level3_hist_1_4_dataset.npz"
save_path0 = "policies/diffusion/diffusion_transformer_1_4"

STATE_TO_ID = {
    "APPROACH": 0,
    "DESCEND": 1,
    "GRASP": 2,
    "LIFT": 3,
    "TRANSPORT": 4,
    "LOWER": 5,
    "RELEASE": 6,
}

STATE_WEIGHT = torch.tensor(
    [
        1.0,   # APPROACH
        1.0,   # DESCEND
        1.0,   # GRASP
        1.0,   # LIFT
        0.5,   # TRANSPORT
        1.0,   # LOWER
        2.0,   # RELEASE
    ], dtype=torch.float32, device=device,
)

# ============================================================
# Dataset
# ============================================================

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
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

# ============================================================
# Model
# ============================================================

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

policy = DiffusionPolicy(
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

schedule = DiffusionSchedule(num_steps=diffusion_steps, schedule_type=schedule_type, device=device,)

optimizer = torch.optim.AdamW(
    [
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
            "params": policy.diffusion_decoder.parameters(),
            "lr": denoiser_lr,
        },
    ]
)

# ============================================================
# Loss
# ============================================================

def diffusion_loss(batch):
    agent = batch["agent"].to(device, non_blocking=True)
    wrist = batch["wrist"].to(device, non_blocking=True)
    proprio = batch["proprio"].to(device, non_blocking=True)
    target_chunk = batch["action"].to(device, non_blocking=True)
    state_id = batch["state"].to(device, non_blocking=True)

    weights = STATE_WEIGHT[state_id]

    B = target_chunk.shape[0]

    # ------------------------------------------------------
    # x0 = expert motion trajectory
    # ------------------------------------------------------

    x0 = target_chunk[..., :action_dim-1]

    gripper_target = (target_chunk[..., action_dim-1] > 0).float()

    # ------------------------------------------------------
    # random diffusion timestep
    # ------------------------------------------------------

    k = torch.randint(
        low=0, high=schedule.num_steps,
        size=(B,),
        device=device,
    )

    # ------------------------------------------------------
    # Gaussian noise
    # ------------------------------------------------------

    noise = torch.randn_like(x0)

    # ------------------------------------------------------
    # q(x_k | x_0)
    # ------------------------------------------------------

    sqrt_alpha_bar = schedule.sqrt_alpha_bar[k].view(B, 1, 1)

    sqrt_one_minus_alpha_bar = (
        schedule.sqrt_one_minus_alpha_bar[k].view(B, 1, 1)
    )

    noisy_actions = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    # ------------------------------------------------------
    # Predict epsilon
    # ------------------------------------------------------

    noise_pred, gripper_logits = policy(
        agent, wrist, proprio,noisy_actions=noisy_actions,
        t=k.float(),
    )

    # ------------------------------------------------------
    # Diffusion + gripper loss
    # ------------------------------------------------------

    motion_loss = F.mse_loss(
        noise_pred, noise, reduction="none",
    ).mean(dim=(1, 2))

    gripper_loss = (
        F.binary_cross_entropy_with_logits(
            gripper_logits, gripper_target, reduction="none",
        ).mean(dim=1)
    )

    loss_per_sample = motion_loss + 1.1 * gripper_loss

    loss = (loss_per_sample * weights).mean()

    return loss, motion_loss.mean(), gripper_loss.mean()

# ============================================================
# Training
# ============================================================

best_eval_loss = float("inf")

for epoch in range(num_epochs):
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    policy.train()

    # Keep frozen ResNet blocks' BatchNorm stats fixed
    obs_encoder.agent_encoder.eval()
    obs_encoder.wrist_encoder.eval()

    # Fine-tuned final stage
    # obs_encoder.agent_encoder.layer4.train()
    # obs_encoder.wrist_encoder.layer4.train()

    train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()

        loss, motion_loss, gripper_loss = diffusion_loss(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * batch["agent"].size(0)
        
        if batch_idx % 100 == 0:
            print(
                f"epoch={epoch+1} "
                f"batch={batch_idx}/{len(train_loader)} "
                f"loss={loss.item():.5f} "
                f"motion={motion_loss.item():.5f} "
                f"gripper={gripper_loss.item():.5f} "
                f"batch_time={time.time()-batch_start:.2f}s"
            )

    train_loss /= len(train_dataset)

    # ========================================================
    # Validation
    # ========================================================

    policy.eval()
    obs_encoder.eval()

    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            loss, _, _ = diffusion_loss(batch)

            val_loss += loss.item() * batch["agent"].size(0)

    val_loss /= len(val_dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs} - "
        f"Train Loss: {train_loss:.6f}, "
        f"Val Loss: {val_loss:.6f} "
        f"time={time.time()-t0:.1f}s"
    )

    print(
        torch.cuda.max_memory_allocated() / 1024**3,
        "GB allocated",
    )

    print(
        torch.cuda.max_memory_reserved() / 1024**3,
        "GB reserved",
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

                "num_decoder_layer":
                    num_decoder_layer,

                "num_heads":
                    num_heads,

                "dim_feedforward":
                    dim_feedforward,

                "diffusion_steps":
                    diffusion_steps,

                "schedule_type":
                    schedule_type,

                "val_loss":
                    val_loss,
            },
            save_path,
        )

        print(
            "saved best:",
            save_path,
        )

