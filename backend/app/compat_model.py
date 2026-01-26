# backend/app/compat_model.py
import torch
import torch.nn as nn

class CompatNet(nn.Module):
    def __init__(self, emb_dim=512, hidden=1024):
        super().__init__()
        in_dim = emb_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, a, b):
        # a,b  -> (N, emb_dim)
        prod = a * b
        diff = torch.abs(a - b)
        x = torch.cat([a, b, prod, diff], dim=1)
        return self.net(x).squeeze(1)  # (N,)
