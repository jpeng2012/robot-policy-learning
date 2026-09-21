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
from policies.act.model_spatial import ACTPolicy

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

latent_dim = 16

num_encoder_layer=4
num_decoder_layer=4
num_heads=8

dim_feedforward=3200

num_workers = 8
kl_weight = 1.0


encoder_lr = 5e-5
projector_lr = 3e-4
latent_encoder_lr = 1e-5
decoder_lr = 1e-5

agent_feat_dim = 512 * vision_history_len
wrist_feat_dim = 512 * vision_history_len
proprio_feat_dim = proprio_dim * proprio_history_len

dataset_path = "data/level3_hist_1_4_dataset.npz"
save_path = "policies/act/act_1_4_3.pth"

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


policy = ACTPolicy(
    observation_encoder=obs_encoder,
    conditon_dim=condition_dim,
    agent_feat_dim=agent_feat_dim,
    wrist_feat_dim=wrist_feat_dim,
    proprio_feat_dim=proprio_feat_dim,
    action_dim=action_dim,
    action_horizon=action_horizon,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    num_encoder_layer=num_encoder_layer,
    num_decoder_layer=num_decoder_layer,
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
        "params": policy.latent_encoder.parameters(),
        "lr": latent_encoder_lr,
    },
    {
        "params": policy.decoder.parameters(),
        "lr": decoder_lr,
    }
])

best_eval_loss = float('inf')
target_kl_weight = 3
warmup_epochs = 3


for epoch in range(num_epochs):

    kl_weight = target_kl_weight * min(
        1.0,
        (epoch + 1) / warmup_epochs,
    )

    torch.cuda.reset_peak_memory_stats()

    policy.train()

    # Keep frozen ResNet parts' BatchNorm stats fixed
    obs_encoder.eval()

    # Fine-tuned final stage
    obs_encoder.agent_encoder.layer4.train()
    obs_encoder.wrist_encoder.layer4.train()

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

        pred_chunk, mu, logvar = policy(
            agent,
            wrist,
            proprio,
            action_chunk=target_chunk,
        )

        pred_motion = pred_chunk[..., :6]
        pred_gripper = pred_chunk[..., 6]

        target_motion = target_chunk[..., :6]
        target_gripper = (
            target_chunk[..., 6] > 0
        ).float()

        motion_loss = F.l1_loss(
                pred_motion,
                target_motion,
                reduction="none",
            ).mean(dim=(1, 2))
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred_gripper,
            target_gripper,
            reduction="none",
        ).mean(1)
        # print(motion_loss.shape)
        # print(gripper_loss.shape)
        action_loss_per_sample = motion_loss + 1.1*gripper_loss        
        action_loss = (action_loss_per_sample * weights).mean()
        kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        kl_loss = kl_per_sample.mean()
        loss = action_loss + kl_weight * kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * agent.size(0)

        if batch_idx % 100 == 0:
            print(
                f"epoch={epoch+1} "
                f"batch={batch_idx}/{len(train_loader)} "
                f"loss={loss.item():.5f} "
                f"motion={motion_loss.mean().item():.5f} "
                f"gripper={gripper_loss.mean().item():.5f} "
                f"action={action_loss.item():.5f} "
                f"kl={kl_loss.item():.5f} "
                f"weighted_kl={(kl_weight * kl_loss).item():.5f}"
                f"batch_time={time.time()-batch_start:.2f}s")

    train_loss /= len(train_dataset)

    # Validation
    policy.eval()
    obs_encoder.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            agent = batch["agent"].to(device, non_blocking=True,)
            wrist = batch["wrist"].to(device, non_blocking=True,)
            proprio = batch["proprio"].to(device, non_blocking=True,)
            target_chunk = batch["action"].to(device, non_blocking=True,)
            state_id = batch["state"].to(device, non_blocking=True,)

            weights = STATE_WEIGHT[state_id]

            pred_chunk, mu, logvar = policy(
                agent,
                wrist,
                proprio,
                action_chunk=target_chunk,
            )
           
            
            pred_motion = pred_chunk[..., :6]
            pred_gripper = pred_chunk[..., 6]
    
            target_motion = target_chunk[..., :6]
            target_gripper = (
                target_chunk[..., 6] > 0
            ).float()
    
            motion_loss = F.l1_loss(
                pred_motion,
                target_motion,
                reduction="none",
            ).mean(dim=(1, 2))

            gripper_loss = F.binary_cross_entropy_with_logits(
                pred_gripper,
                target_gripper,
                reduction="none",
            ).mean(dim=1)

            action_loss_per_sample = motion_loss + 1.1*gripper_loss        
            action_loss = (action_loss_per_sample * weights).mean()
            kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            kl_loss = kl_per_sample.mean()
            loss = action_loss + kl_weight * kl_loss
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

    if val_loss < best_eval_loss:
    
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

                    "latent_dim":
                        latent_dim,
    
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