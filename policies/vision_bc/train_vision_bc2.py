import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from policies.vision_bc.dataset import VisionBCDataset
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

class VisionBCModel(nn.Module):
    def __init__(
            self,
            vision_history_len=1,
            proprio_history_len=4,
            proprio_dim=9,
            action_dim=7,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.agent_encoder = resnet18(weights=weights)
        self.wrist_encoder = resnet18(weights=weights)

        self.agent_encoder.fc = nn.Identity()
        self.wrist_encoder.fc = nn.Identity()

        self.vision_history_len = vision_history_len
        self.proprio_history_len = proprio_history_len
        input_dim = 512 * 2 * vision_history_len + proprio_dim * proprio_history_len

        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, agent, wrist, proprio):
        # agent: (B, T, C, H, W)
        B, T, C, H, W = agent.shape
        agent = agent.reshape(-1, *agent.shape[2:])

        # wrist: (B, T, C, H, W)
        wrist = wrist.reshape(-1, *wrist.shape[2:])

        agent_feat = self.agent_encoder(agent).reshape(-1, T*512)
        wrist_feat = self.wrist_encoder(wrist).reshape(-1, T*512)

        proprio = proprio.reshape(B, -1)

        x = torch.cat([agent_feat, wrist_feat, proprio], dim=1)

        return self.head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 30
vision_history_len = 1
proprio_history_len = 4

dataset_path = "data/level3_hist_1_4_dataset.npz"
save_path = "policies/vision_bc/vision_bc_level3_hist_1_4.pth"

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
    0.6,   # TRANSPORT
    1.0,   # LOWER
    1.5,   # RELEASE
], dtype=torch.float32, device=device)

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

data = np.load(dataset_path)

proprio_train = data["proprio_train"].astype(np.float32)
proprio_mean = np.mean(proprio_train, axis=(0, 1))
proprio_std = np.std(proprio_train, axis=(0, 1))+1e-6

train_dataset = VisionBCDataset(
    dataset_npz=dataset_path,
    data_root="data/level3",
    split="train",
    transform=transform,
    proprio_mean=proprio_mean,
    proprio_std=proprio_std,
)

val_dataset = VisionBCDataset(
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

batch = next(iter(train_loader))

print(batch["agent"].shape)
print(batch["wrist"].shape)
print(batch["proprio"].shape)
print(batch["action"].shape)

policy = VisionBCModel(
    vision_history_len=vision_history_len,
    proprio_history_len=proprio_history_len,
    proprio_dim=9,
    action_dim=7,
).to(device)

for p in policy.agent_encoder.parameters():
    p.requires_grad = False

for p in policy.wrist_encoder.parameters():
    p.requires_grad = False

for p in policy.agent_encoder.layer4.parameters():
    p.requires_grad = True

for p in policy.wrist_encoder.layer4.parameters():
    p.requires_grad = True


# fine-tune the ResNets, but with a lower LR on the encoders than on the MLP head:
optimizer = torch.optim.AdamW([
    {
        "params": policy.agent_encoder.layer4.parameters(),
        "lr": 1e-5,
    },
    {
        "params": policy.wrist_encoder.layer4.parameters(),
        "lr": 1e-5,
    },
    {
        "params": policy.head.parameters(),
        "lr": 1e-3,
    },
])

criterion = nn.MSELoss()

for epoch in range(num_epochs):

    torch.cuda.reset_peak_memory_stats()

    policy.train()
    train_loss = 0.0

    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()

        agent = batch["agent"].to(device)
        wrist = batch["wrist"].to(device)
        proprio = batch["proprio"].to(device)
        action = batch["action"].to(device)
        state_id = batch["state"].to(device)

        weights = STATE_WEIGHT[state_id]

        pred_action = policy(agent, wrist, proprio)

        loss_per_sample = ((pred_action - action) ** 2).mean(dim=1)

        optimizer.zero_grad(set_to_none=True)
        loss = (loss_per_sample * weights).mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * agent.size(0)

        if batch_idx % 100 == 0:
            print(
                f"epoch={epoch+1} "
                f"batch={batch_idx}/{len(train_loader)} "
                f"loss={loss.item():.5f} "
                f"batch_time={time.time()-batch_start:.2f}s")

    train_loss /= len(train_dataset)

    # Validation
    policy.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            agent = batch["agent"].to(device, non_blocking=True,)
            wrist = batch["wrist"].to(device, non_blocking=True,)
            proprio = batch["proprio"].to(device, non_blocking=True,)
            action = batch["action"].to(device, non_blocking=True,)
            state_id = batch["state"].to(device, non_blocking=True,)

            weights = STATE_WEIGHT[state_id]

            pred_action = policy(agent, wrist, proprio)

            loss_per_sample = ((pred_action - action) ** 2).mean(dim=1)
            val_loss += (loss_per_sample * weights).mean().item() * agent.size(0)

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

torch.save(
    {
        "model_state_dict": policy.state_dict(),
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "vision_history_len": vision_history_len,
        "proprio_history_len": proprio_history_len,
    }, save_path
)

print(f"Model saved to {save_path}")