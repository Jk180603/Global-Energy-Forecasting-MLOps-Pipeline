import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import mlflow

class EnergyDataset(Dataset):
    def __init__(self, df, seq_len=5):
        self.seq_len = seq_len
        self.data = df[['primary_energy_consumption']].values
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.seq_len]), torch.FloatTensor(self.data[idx+self.seq_len])

class LSTMModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)
    def forward(self, x): 
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

# Train
df = pd.read_parquet("/content/energyglobal/data/processed/energy_data.parquet")
df = df[df['country'] == 'Germany'].sort_values('year')

dataset = EnergyDataset(df)
loader = DataLoader(dataset, batch_size=32)

model = LSTMModel()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

mlflow.set_experiment("energyglobal-lstm")
with mlflow.start_run():
    for epoch in range(10):
        for x, y in loader:
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
    mlflow.pytorch.log_model(model, "lstm_model")
    torch.save(model.state_dict(), "/content/energyglobal/models/lstm_germany.pth")
    print("LSTM trained for Germany!")
