import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):
    
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out