# train_var_gpu_multistep.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Config
# -----------------------------
JSON_FILE = "full_timeseries.json"   # input JSON
MODEL_FILE = "var_model.pt"          # saved model
N_LAGS = 12                          # number of months to look back
HORIZON = 12                         # forecast horizon (# of months ahead)
N_EPOCHS = 5000                      # max epochs
LR = 0.001                           # learning rate
PATIENCE = 1000                       # early stopping patience
BATCH_SIZE = 64                      # mini-batch size
VAL_SPLIT = 0.2                      # validation ratio

# -----------------------------
# Step 1: Load JSON file
# -----------------------------
df = pd.read_json(JSON_FILE, orient="records")

# Convert period to month number and create datetime
df['month'] = df['period'].str.extract('M(\d+)').astype(int)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.sort_values('date')

# Pivot so each series_id becomes a column
var_df = df.pivot(index='date', columns='series_id', values='value')
var_df = var_df.fillna(method='ffill').fillna(method='bfill')

# -----------------------------
# Step 2: Scale data
# -----------------------------
scaler = StandardScaler()
var_scaled = scaler.fit_transform(var_df.values)

# -----------------------------
# Step 3: Create lagged features for multi-step prediction
# -----------------------------
def create_lagged_matrix(data, n_lags, horizon):
    """
    X: past n_lags windows
    Y: next horizon windows
    """
    X_list, Y_list = [], []
    for t in range(n_lags, len(data) - horizon + 1):
        X_list.append(data[t-n_lags:t].flatten())
        Y_list.append(data[t:t+horizon])  # sequence of next horizon steps
    X = torch.tensor(X_list, dtype=torch.float32)
    Y = torch.tensor(Y_list, dtype=torch.float32)  # shape: (samples, horizon, n_series)
    return X, Y

X, Y = create_lagged_matrix(var_scaled, N_LAGS, HORIZON)

# Train/val split
split_idx = int(len(X) * (1 - VAL_SPLIT))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_val, Y_val = X_val.to(device), Y_val.to(device)

# -----------------------------
# Step 4: Define model
# -----------------------------
n_series = var_df.shape[1]
input_dim = N_LAGS * n_series
output_dim = HORIZON * n_series  # predict full horizon

class MultiStepModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(-1, HORIZON, n_series)

model = MultiStepModel(input_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()


#Training loop with early stopping

best_val_loss = float("inf")
counter = 0

for epoch in range(1, N_EPOCHS + 1):
    # ---- Training ----
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        X_batch, Y_batch = X_train[indices], Y_train[indices]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= X_train.size(0)

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, Y_val).item()

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{N_EPOCHS} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_lags': N_LAGS,
            'horizon': HORIZON,
            'series_columns': var_df.columns.tolist(),
            'scaler': scaler
        }, MODEL_FILE)
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.4f}")
            break

print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to {MODEL_FILE}")
