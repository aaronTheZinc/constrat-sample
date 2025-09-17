# inference.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Config
MODEL_FILE = "var_model.pt"
JSON_FILE = "full_timeseries.json"
N_LAGS = 12    # months to look back
N_STEPS = 12   # forecast horizon
TARGET_SERIES = "WPU081"  # set to series_id you want to forecast; None = first series

#Load JSON and pivot
df = pd.read_json(JSON_FILE, orient="records")
df['month'] = df['period'].str.extract('M(\d+)').astype(int)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.sort_values('date')


 #Load checkpoint (model + scaler)
checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
scaler = checkpoint['scaler']  # load scaler from checkpoint
series_columns = checkpoint['series_columns']
n_lags = checkpoint['n_lags']
horizon = checkpoint['horizon']  # Get the horizon from checkpoint

var_df = df.pivot(index='date', columns='series_id', values='value')
var_df = var_df[series_columns]
var_df = var_df.fillna(method='ffill').fillna(method='bfill')
var_scaled = scaler.transform(var_df.values)

n_series = var_scaled.shape[1]
input_dim = n_series * n_lags
output_dim = horizon * n_series  # This should match training

class MultiStepModel(nn.Module):
    def __init__(self, input_dim, output_dim, horizon, n_series):
        super().__init__()
        self.horizon = horizon
        self.n_series = n_series
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(-1, self.horizon, self.n_series)  # Same as training

print(f"Input dim: {input_dim}, Output dim: {output_dim}")
print(f"n_series: {n_series}, horizon: {horizon}")

model = MultiStepModel(input_dim, output_dim, horizon, n_series)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Predict N steps using N Lags
def forecast_series(data, series_id, n_lags=12, n_steps=6):
    """
    Forecast using the trained multi-step model
    """
    history = data[-n_lags:]  # shape (12, n_series)
    history_tensor = torch.tensor(history.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    
    print(f"History tensor shape: {history_tensor.shape}")
    
    # Check for NaN in input
    if torch.isnan(history_tensor).any():
        print("WARNING: NaN detected in input history!")
        return None
    
    forecast = []
    
    with torch.no_grad():  # Important for inference
        for step in range(n_steps):
            # Get prediction
            y_pred = model(history_tensor)  # shape: (1, horizon, n_series)
            
            # Check for NaN in prediction
            if torch.isnan(y_pred).any():
                print(f"NaN detected in prediction at step {step}")
                print(f"Model parameters check:")
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"  NaN found in parameter: {name}")
                break
            
            # Convert to numpy
            y_pred_np = y_pred.detach().cpu().numpy()  # shape: (1, horizon, n_series)
            
            print(f"Step {step}: prediction shape {y_pred_np.shape}")
            print(f"Scaled prediction range: [{y_pred_np.min():.4f}, {y_pred_np.max():.4f}]")
            
            # IMPORTANT: Inverse transform to original scale
            # Create full array for proper inverse scaling
            full_pred = np.zeros((y_pred_np.shape[1], len(series_columns)))
            full_pred[:, :] = y_pred_np[0, :, :]
            
            # Inverse transform to get back to original scale
            pred_original_scale = scaler.inverse_transform(full_pred)
            
            print(f"Original scale prediction range: [{pred_original_scale.min():.4f}, {pred_original_scale.max():.4f}]")
            
            # Take the first predicted time step as the next forecast (original scale)
            next_step_original = pred_original_scale[0, :]  # shape: (n_series,)
            forecast.append(next_step_original)
            
            # For updating history, we need to use SCALED values (what the model expects)
            next_step_scaled = y_pred_np[0, 0, :]  # shape: (n_series,)
            
            # Update history: remove oldest, add newest prediction (in scaled space)
            history = np.vstack([history[1:], next_step_scaled])
            history_tensor = torch.tensor(history.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    
    return np.array(forecast)  # shape (n_steps, n_series) - in ORIGINAL scale



print(f"Data shape: {var_scaled.shape}")
print(f"Data range: [{var_scaled.min():.4f}, {var_scaled.max():.4f}]")
print(f"Data contains NaN: {np.isnan(var_scaled).any()}")
print(f"Data contains Inf: {np.isinf(var_scaled).any()}")

# Check model parameters
nan_params = []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_params.append(name)

if nan_params:
    print(f"WARNING: NaN found in model parameters: {nan_params}")
    print("Model was not trained properly or corrupted during saving.")
else:
    print("Model parameters look healthy (no NaN)")


if TARGET_SERIES is None:
    TARGET_SERIES = series_columns[0]

print(f"\nRunning forecast for series: {TARGET_SERIES}")
forecast_values = forecast_series(var_scaled, TARGET_SERIES, n_lags=N_LAGS, n_steps=N_STEPS)

if forecast_values is not None:
    series_idx = series_columns.index(TARGET_SERIES)
    forecast_for_target = forecast_values[:, series_idx]
    
    last_date = var_df.index[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=N_STEPS, freq='MS')
    
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "series_id": TARGET_SERIES,
        "forecast": forecast_for_target
    })
    
    print(f"\nForecast for series {TARGET_SERIES}:")
    print(forecast_df)
    
    # Compare with actual values if available
    print(f"\nLast few actual values for {TARGET_SERIES}:")
    actual_series = scaler.inverse_transform(var_scaled)[:, series_idx]
    recent_actual = pd.DataFrame({
        'date': var_df.index[-5:],
        'actual': actual_series[-5:]
    })
    print(recent_actual)
    
    # Save forecast
    forecast_df.to_csv(f"forecast_{TARGET_SERIES}.csv", index=False)
    print(f"\nForecast saved to forecast_{TARGET_SERIES}.csv")
else:
    print("Forecast failed due to NaN values")