__all__ = ['ModeEstimator']

import torch
import torch.nn as nn

class ModeEstimator(nn.Module):
    def __init__(self, in_dim = 1024, h_dims = [512, 256, 64], out_dim = 3):
        super().__init__()
        self.y2h = nn.Sequential(
            nn.Linear(in_dim, h_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(h_dims[0])
        )
        hidden_layers = []
        for i in range(len(h_dims) - 1):
            hidden_layers.extend(
                [
                nn.Linear(h_dims[i], h_dims[i+1]),
                nn.ReLU(), 
                nn.BatchNorm1d(h_dims[i+1])
                ]
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.h2mode = nn.Sequential(
            nn.Linear(h_dims[-1], out_dim)
        )
    
    def forward(self, y):
        h = self.y2h(y)
        for layer in self.hidden_layers:
            h = layer(h)
        out = self.h2mode(h)
        return out

