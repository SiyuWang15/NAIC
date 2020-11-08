__all__ = ['ModeEstimator']

import torch
import torch.nn as nn

class ModeEstimator(nn.Module):
    def __init__(self, in_dim = 1024, h_dim = [512, 256, 64], out_dim = 3):
        super().__init__()
        self.y2h = nn.Sequential(
            nn.Linear(in_dim, h_dim[0]),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim[0])
        )
        hidden_layers = []
        for i in range(len(h_dim) - 1):
            hidden_layers.extend(
                [
                nn.Linear(h_dim[i], h_dim[i+1]),
                nn.ReLU(), 
                nn.BatchNorm1d(h_dim[i+1])
                ]
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.h2mode = nn.Sequential(
            nn.Linear(h_dim[-1], out_dim)
        )
    
    def forward(self, y):
        h = self.y2h(y)
        for layer in self.hidden_layers:
            h = layer(h)
        out = self.h2mode(h)
        return out

