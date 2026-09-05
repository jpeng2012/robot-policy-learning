import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
num_epochs = 30
lr = 1e-3

dataset_path = "data/level2_dataset.npz"
save_path = "policies/bc/bc_level2.pth"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

data = np.load(dataset_path)

X_train = data["X_train"].astype(np.float32)
Y_train = data["Y_one_train"].astype(np.float32)

X_val = data["X_val"].astype(np.float32)
Y_val = data["Y_one_val"].astype(np.float32)

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)


# ---------------------------------------------------------
# Normalize input
# ---------------------------------------------------------

x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0) + 1e-6

X_train = (X_train - x_mean) / x_std
X_val = (X_val - x_mean) / x_std


train_set = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(Y_train),
)

val_set = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(Y_val),
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
)


# ---------------------------------------------------------
# Policy
# ---------------------------------------------------------

class BCPolicy(nn.Module):
    def __init__(self, input_dim=96, action_dim=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


policy = BCPolicy(
    input_dim=X_train.shape[1],
    action_dim=Y_train.shape[1],
).to(device)


optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=lr,
    weight_decay=1e-4,
)

criterion = nn.MSELoss()


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(num_epochs):

    policy.train()

    train_loss = 0.0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        pred = policy(x)

        loss = criterion(
            pred,
            y,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() * x.size(0)
        )

    train_loss /= len(train_set)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    policy.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            pred = policy(x)

            loss = criterion(
                pred,
                y,
            )

            val_loss += (
                loss.item() * x.size(0)
            )

    val_loss /= len(val_set)

    print(
        f"Epoch {epoch+1:02d} | "
        f"train={train_loss:.6f} | "
        f"val={val_loss:.6f}"
    )


# ---------------------------------------------------------
# Save model + normalization
# ---------------------------------------------------------

torch.save(
    {
        "model_state_dict": policy.state_dict(),

        "x_mean": torch.from_numpy(x_mean),,
        "x_std": torch.from_numpy(x_std),

        "input_dim": X_train.shape[1],
        "action_dim": Y_train.shape[1],
    },
    save_path,
)

print("saved:", save_path)