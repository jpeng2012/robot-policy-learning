import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from policies.vision_bc.dataset import VisionChunkBCDataset
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from policies.vision_bc.dataset import VisionChunkBCDataset

from policies.common.observation_encoder import ObservationEncoder

from policies.diffusion.model import (
    ActionDenoiser,
)

from policies.diffusion.diffusion_utils import (
    DiffusionSchedule,
)


# ============================================================
# Config
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 30

vision_history_len = 1
proprio_history_len = 4

proprio_dim = 12      # joint 7 + eef 3 + gripper 2

action_horizon = 16
action_dim = 7

condition_dim = 256
time_dim = 64
hidden_dim = 512

diffusion_steps = 50

batch_size = 64
num_epochs = 20

num_workers = 8

encoder_lr = 1e-5
projector_lr = 1e-3
denoiser_lr = 1e-3

dataset_path = "data/level3_hist_1_4_dataset.npz"
save_path = "policies/diffusion/vision_diffusion_hist_1_4.pth"

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

agent_encoder = resnet18(weights=weights)

wrist_encoder = resnet18(weights=weights)

agent_encoder.fc = nn.Identity()
wrist_encoder.fc = nn.Identity()

for p in agent_encoder.parameters():
    p.requires_grad = False

for p in wrist_encoder.parameters():
    p.requires_grad = False

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

denoiser = ActionDenoiser(
    action_dim=action_dim,
    action_horizon=action_horizon,
    condition_dim=condition_dim,
    time_dim=time_dim,
    hidden_dim=hidden_dim,
).to(device)

schedule = DiffusionSchedule(
    num_steps=diffusion_steps,
    device=device,
)

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
        "params": denoiser.parameters(),
        "lr": denoiser_lr,
    },
], 
weight_decay=1e-4)

def diffusion_loss(
        batch,
        obs_encoder,
        denoiser,
        schedule,
):
    agent = batch["agent"].to(device, non_blocking=True)
    wrist = batch["wrist"].to(device, non_blocking=True)

    proprio = batch["proprio"].to(device, non_blocking=True)

    action_chunk = batch["action"].to(device, non_blocking=True)

    B = action_chunk.shape[0]

    # --------------------------------------------------------
    # Condition
    # --------------------------------------------------------
    condition = obs_encoder(agent, wrist, proprio)

    # --------------------------------------------------------
    # Random diffusion timestep
    # --------------------------------------------------------

    k = torch.randint(
        low=0, 
        high=schedule.num_steps,
        size= (B, ),
        device=device
    )

    # --------------------------------------------------------
    # Gaussian noise
    # --------------------------------------------------------
    noise = torch.randn_like(action_chunk)

    # --------------------------------------------------------
    # q(A_k | A_0)
    # --------------------------------------------------------
    sqrt_alpha_bar = (
        schedule.sqart_alpha_bar[k].view(B, 1, 1)
    )

    sqrt_one_minus_alpha_bar = (
        schedule.sqrt_one_minus_alpha_bar[k].view(B, 1, 1)
    )

    noisy_action = (
        sqrt_alpha_bar * action_chunk + sqrt_one_minus_alpha_bar*noise
    )

    # --------------------------------------------------------
    # Predict epsilon
    # --------------------------------------------------------

    noise_pred = denoiser(noisy_action, k, condition,)

    # --------------------------------------------------------
    # Noise prediction loss
    # --------------------------------------------------------
    loss = ((noise_pred-noise)**2).mean()

    return loss

best_eval_loss = float('inf')

for epoch in range(num_epochs):
    epoch_start = time()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    obs_encoder.train()
    denoiser.train()

     # Keep frozen ResNet blocks in eval mode
    obs_encoder.agent_encoder.eval()
    obs_encoder.wrist_encoder.eval()

    obs_encoder.agent_encoder.layer4.train()
    obs_encoder.wrist_encoder.layer4.train()

    train_loss = 0.0
    train_count = 0

    for batch_idx, batch in enumerate(
        train_loader
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        loss = diffusion_loss(
            batch, 
            obs_encoder, 
            denoiser,
            schedule)

        loss.backward()
        optimizer.step()

        batch_size_actual = batch[
            "action_chunk"
        ].shape[0]

        train_loss += (
            loss.item()
            * batch_size_actual
        )

        train_count += (
            batch_size_actual
        )

        if batch_idx % 25 == 0:

            print(
                f"epoch={epoch+1:02d} "
                f"batch="
                f"{batch_idx:04d}/"
                f"{len(train_loader):04d} "
                f"loss={loss.item():.5f}"
            )

    train_loss /= train_count


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    obs_encoder.eval()
    denoiser.eval()

    val_loss = 0.0
    val_count = 0

    with torch.no_grad():

        for batch in val_loader:

            loss = diffusion_loss(
                batch,
                obs_encoder,
                denoiser,
                schedule,
            )

            batch_size_actual = batch[
                "action_chunk"
            ].shape[0]

            val_loss += (
                loss.item()
                * batch_size_actual
            )

            val_count += (
                batch_size_actual
            )

    val_loss /= val_count


    epoch_time = (
        time.time()
        - epoch_start
    )


    print(
        f"\nEpoch {epoch+1:02d} | "
        f"train={train_loss:.6f} | "
        f"val={val_loss:.6f} | "
        f"time={epoch_time:.1f}s\n"
    )


    # --------------------------------------------------------
    # Save best
    # --------------------------------------------------------

    if val_loss < best_eval_loss:

        best_eval_loss = val_loss

        torch.save(
            {
                "obs_encoder_state_dict":
                    obs_encoder.state_dict(),

                "denoiser_state_dict":
                    denoiser.state_dict(),

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

                "time_dim":
                    time_dim,

                "hidden_dim":
                    hidden_dim,

                "diffusion_steps":
                    diffusion_steps,

                "val_loss":
                    val_loss,
            },
            save_path,
        )

        print(
            "saved best:",
            save_path,
        )





