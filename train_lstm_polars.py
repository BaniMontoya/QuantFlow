import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import sys

# Configuración
LOOKBACK = 20
HIDDEN_SIZE = 32
BATCH_SIZE = 512
EPOCHS = 5
FILENAME = "QuantFlow_TrainingData.csv"
MODEL_NAME = "QuantFlow_LSTM.onnx"
MAX_BARS = 500000

def calculate_indicators(df):
    """Calcula indicadores usando Polars (mucho más rápido que Pandas)"""
    df = df.with_columns([
        # RSI
        pl.col('<CLOSE>').diff().alias('delta')
    ])
    
    df = df.with_columns([
        pl.when(pl.col('delta') > 0).then(pl.col('delta')).otherwise(0).alias('gain'),
        pl.when(pl.col('delta') < 0).then(-pl.col('delta')).otherwise(0).alias('loss')
    ])
    
    df = df.with_columns([
        pl.col('gain').rolling_mean(window_size=14).alias('avg_gain'),
        pl.col('loss').rolling_mean(window_size=14).alias('avg_loss')
    ])
    
    df = df.with_columns([
        (100 - (100 / (1 + (pl.col('avg_gain') / (pl.col('avg_loss') + 1e-9))))).alias('RSI')
    ])
    
    # ATR
    df = df.with_columns([
        (pl.col('<HIGH>') - pl.col('<LOW>')).alias('hl'),
        (pl.col('<HIGH>') - pl.col('<CLOSE>').shift(1)).abs().alias('hc'),
        (pl.col('<LOW>') - pl.col('<CLOSE>').shift(1)).abs().alias('lc')
    ])
    
    df = df.with_columns([
        pl.max_horizontal(['hl', 'hc', 'lc']).alias('tr')
    ])
    
    df = df.with_columns([
        pl.col('tr').rolling_mean(window_size=14).alias('ATR')
    ])
    
    # Returns
    df = df.with_columns([
        pl.col('<CLOSE>').pct_change().alias('Returns')
    ])
    
    return df.drop_nulls()

print(f"Loading data with Polars...")
try:
    # Polars es mucho más rápido leyendo CSVs
    df_raw = pl.read_csv(
        FILENAME, 
        separator=' ',
        has_header=True,
        ignore_errors=True
    )
    
    if len(df_raw) > MAX_BARS:
        df_raw = df_raw.tail(MAX_BARS)
    
    print(f"Calculating indicators...")
    df = calculate_indicators(df_raw)
    
    # Convertir a numpy para el resto del pipeline
    features = ['Returns', 'RSI', 'ATR']
    data = df.select(features).to_numpy()
    prices = df.select('<CLOSE>').to_numpy().flatten()
    
    print(f"Data ready: {len(df)} bars")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Normalización
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data).astype(np.float32)

# Secuenciación rápida con Numpy
print("Sequencing...")
PREDICT_AHEAD = 5
THRESHOLD = 0.0001

from numpy.lib.stride_tricks import sliding_window_view
X = sliding_window_view(data_scaled[:-PREDICT_AHEAD], (LOOKBACK, data_scaled.shape[1])).squeeze()

# Target
curr_prices = prices[LOOKBACK-1 : -PREDICT_AHEAD]
fut_prices = prices[LOOKBACK + PREDICT_AHEAD - 1 :]
y = np.zeros(len(curr_prices), dtype=np.longlong)
y[fut_prices > curr_prices * (1 + THRESHOLD)] = 1
y[fut_prices < curr_prices * (1 - THRESHOLD)] = 2

# X y y deben tener el mismo tamaño
X = X[:len(y)]

print(f"Final training set: {X.shape}")

# Modelo
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel(len(features), HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Training
X_t = torch.from_numpy(X.copy())
y_t = torch.from_numpy(y)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

print("Training (5 epochs)...")
model.train()
for epoch in range(EPOCHS):
    l_sum = 0
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        l_sum += loss.item()
    print(f"Epoch {epoch+1} Loss: {l_sum/len(loader):.4f}")

# Export
print("Exporting ONNX...")
model.eval()
torch.onnx.export(model, torch.randn(1, LOOKBACK, len(features)), MODEL_NAME, 
                  input_names=['input'], output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("Done!")
