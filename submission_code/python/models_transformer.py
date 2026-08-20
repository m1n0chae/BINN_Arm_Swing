# models_transformer_loso.py
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 3, dim_ff: int = 256, drop: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, out_dim)
    def forward(self, x):
        x = self.inp(x)
        x = self.pos(x)
        z = self.enc(x)
        y = self.head(z)
        return y
