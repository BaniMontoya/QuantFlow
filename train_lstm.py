import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import onnx
import sys
import os

# Configuración
LOOKBACK = 20
HIDDEN_SIZE = 32 # Reducido para velocidad
BATCH_SIZE = 512 # Aumentado para velocidad
EPOCHS = 5
FILENAME = "QuantFlow_TrainingData.csv"
MODEL_NAME = "QuantFlow_LSTM.onnx"
MAX_BARS = 500000 # Limitamos a las últimas 500k velas (~1 año M1) para rapidez

def calculate_indicators(df):
    close = df['<CLOSE>']
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    # ATR
    hl = df['<HIGH>'] - df['<LOW>']
    hc = np.abs(df['<HIGH>'] - close.shift())
    lc = np.abs(df['<LOW>'] - close.shift())
    tr = np.max(np.vstack([hl, hc, lc]), axis=0)
    df['ATR'] = pd.Series(tr).rolling(14).mean().values
    df['Returns'] = close.pct_change()
    return df.dropna()

print(f"Loading data...")
try:
    # Leer solo las últimas filas si el archivo es gigante
    df_raw = pd.read_csv(FILENAME, sep='\s+', engine='python')
    if len(df_raw) > MAX_BARS:
        df_raw = df_raw.tail(MAX_BARS).copy()
    
    df = calculate_indicators(df_raw)
    features = ['Returns', 'RSI', 'ATR']
    data = df[features].values
    prices = df['<CLOSE>'].values
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

# X: [Samples, Lookback, Features]
# Usamos stride_tricks para no hacer un bucle for de millones de vueltas
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
X_t = torch.from_numpy(X)
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
