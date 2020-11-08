import torch
import torch.nn as nn

class RouteEstimator(nn.Module):
    def __init__(self, in_dim, out_dim, h_dims, act):
        super().__init__()
        act_fun = nn.ReLU()
        if act == 'sigmoid':
            act_fun = nn.Sigmoid()
        elif act == 'tanh':
            act_fun = nn.Tanh()
        self.y2h = nn.Linear(in_dim, h_dims[0])
        hidden_layers = []
        for i in range(len(h_dims)-1):
            hidden_layers.extend(
                [
                    nn.Linear(h_dims[i], h_dims[i+1]),
                    act_fun,
                    nn.BatchNorm1d(h_dims[i+1])
                ]
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.linear = nn.Linear(h_dims[-1], out_dim)
    
    def forward(self, y):
        out = self.y2h(y)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.linear(out)
        return out 