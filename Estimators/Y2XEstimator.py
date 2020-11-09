import torch 
import torch.nn as nn

class Y2XEstimator(nn.Module):
    def __init__(self, in_dim, out_dim, h_dims):
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
        self.h2x = nn.Sequential(
            nn.Linear(h_dims[-1], out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, y):
        h = self.y2h(y)
        for layer in self.hidden_layers:
            h = layer(h)
        x = self.h2x(h)
        return x