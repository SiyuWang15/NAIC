import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base import ResizeConv2d, BasicBlockDec, BasicBlockEnc

class Y2XEstimator(nn.Module):
    def __init__(self, in_channels, out_channels, num_Blocks): 
        super().__init__()
        self.in_planes = 64 
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride = 2, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.encoder = nn.ModuleList([
            self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1), 
            self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride = 2),
            self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride = 2),
            self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride = 2)  # 512x1x2
        ])
        self.conv2 = ResizeConv2d(512, 512, kernel_size=3, scale_factor=2)
        self.decoder = nn.ModuleList([
            self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2),
            self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2),
            self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2),
            self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        ])
        self.conv3 = ResizeConv2d(64, out_channels, kernel_size=3, scale_factor=1)

    def _make_layer(self, mod, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [mod(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        for layer in self.encoder:
            x = layer(x)
        x = self.conv2(x)
        for layer in self.decoder:
            x = layer(x)
        return nn.Sigmoid()(self.conv3(x))
