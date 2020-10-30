import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, n_blocks):
        super().__init__()
        self.y2h = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU()
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU()])
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.h2x = nn.Sequential(
                nn.Linear(h_dim, out_dim), 
                nn.Sigmoid()
        )
    
    def forward(self, y):
        h = self.y2h(y)
        for layer in self.hidden_layers:
            h = layer(h)
        x = self.h2x(h)
        return x

class ComplexMLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        super().__init__()
        self.y2h = nn.Sequential(
            nn.Linear(in_dim, h_dim[0]),
            nn.ReLU()
        )
        hidden_layers = []
        for i in range(len(h_dim) - 1):
            hidden_layers.extend(
                [
                nn.Linear(h_dim[i], h_dim[i+1]),
                nn.ReLU()
                ]
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.h2x = nn.Sequential(
            nn.Linear(h_dim[-1], out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, y):
        h = self.y2h(y)
        for layer in self.hidden_layers:
            h = layer(h)
        x = self.h2x(h)
        return x