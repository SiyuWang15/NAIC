import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ResizeConv2d, BasicBlockEnc, BasicBlockDec
from .resnet import *

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

class CNNRouteEstimator(nn.Module):
    def __init__(self, nc = 2, num_Blocks = [2,2,2,2], out_dim = 256):
        super().__init__()
        self.in_planes = 64 
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride = 2, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride = 2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride = 2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride = 2)
        self.linear = nn.Linear(512, self.out_dim)
    
    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNetRouteEstimator(nn.Module):
    def __init__(self, tag):
        super().__init__()
        if tag == 34:
            self.net = resnet34()
        elif tag == 50:
            self.net = resnet50()
        elif tag == 101:
            self.net = resnet101()
    
    def forward(self, x):
        return self.net(x)