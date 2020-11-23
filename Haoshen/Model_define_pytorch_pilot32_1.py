import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNN_Estimation(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=256):
        super(CNN_Estimation, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*4*6, num_classes)
        self.linear = nn.Linear(512 * 1 * 32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Res_Estimation(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=256):
        super(Res_Estimation, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*4*6, num_classes)
        self.linear = nn.Linear(512 * 1 * 32, num_classes)

        # self.fft_linear = nn.Linear(256, 8*256)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, y, hf):
        x = torch.cat([y, hf], 2)
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = hf.reshape(hf.size(0), -1) - out
        return out

class yp2h_estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(yp2h_estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(h_dim)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU(inplace=True), nn.BatchNorm1d(h_dim)])
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

class hhat2h_estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(hhat2h_estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(h_dim)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU(inplace=True), nn.BatchNorm1d(h_dim)])
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self,h,n):
        y = h + n
        y = self.input_layer(y)
        for layer in self.hidden_layers:
            y = layer(y)
        y = self.output_layer(y)
        out = h - y
        return out


class FC_ELU_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_ELU_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(h_dim)
            #nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU(inplace=True), nn.BatchNorm1d(h_dim)]) # nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(self.out_dim/8))
        return x


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1024):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*4*6, num_classes)
        self.linear = nn.Linear(512 * 2 * 32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out


def CE_ResNet18():
    return CNN_Estimation(BasicBlock, [2,2,2,2])

def CE_ResNet34():
    return CNN_Estimation(BasicBlock, [3,4,6,3])



"""
class FC_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(h_dim)
            #nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),  nn.BatchNorm1d(h_dim)]) # nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(self.out_dim/8))
        return x

class FC_DropOut_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_DropOut_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(h_dim)
            nn.Dropout(p=0.1)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),   nn.Dropout(p=0.1)]) #
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(self.out_dim/8))
        return x

class FC_Sig_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_Sig_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Sigmoid(),
            nn.BatchNorm1d(h_dim)
            #nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.Sigmoid(), nn.BatchNorm1d(h_dim)]) # nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(self.out_dim/8))
        return x


class FC_ELU_DropOut_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_ELU_DropOut_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ELU(inplace=True),
            #nn.BatchNorm1d(h_dim)
            nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU(inplace=True),nn.Dropout(p=0.5)]) # nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(self.out_dim/8))
        return x

class ModeEstimator(nn.Module):
    def __init__(self, in_dim=1024, h_dim=[512, 256, 64], out_dim=3):
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
                    nn.Linear(h_dim[i], h_dim[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim[i + 1])
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

class FC_Res_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_Res_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(h_dim)
            #nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),  nn.BatchNorm1d(h_dim)]) # nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x0 = x
        x = x.view(-1, self.in_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(self.out_dim/8))
        x = x0 - x
        return x

class FC_Res_ELU_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_Res_ELU_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(h_dim)
            #nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU(inplace=True),  nn.BatchNorm1d(h_dim)]) # nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        x0 = x
        x = x.view(-1, self.in_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(self.out_dim/8))
        x = x0 - x
        return x

class FC_Res_Noise_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_Res_Noise_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(h_dim)
            #nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),  nn.BatchNorm1d(h_dim)]) # nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x, n):
        y = x + n
        y = y.view(-1, self.in_dim)
        y = self.input_layer(y)
        for layer in self.hidden_layers:
            y = layer(y)
        y = self.output_layer(y)
        y = y.view(-1, 2, 2, 2, int(self.out_dim/8))
        out = x - y
        return out

class FC_Res_Noise1_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_Res_Noise1_Estimation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(h_dim)
            nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]) #
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x, n):
        y = x + n
        y = y.view(-1, self.in_dim)
        y = self.input_layer(y)
        for layer in self.hidden_layers:
            y = layer(y)
        y = self.output_layer(y)
        y = y.view(-1, 2, 2, 2, int(self.out_dim/8))
        out = x - y
        return out

def NMSE_cuda(x_hat, x):
    x_real = x[:, 0, :, :].view(len(x),-1)
    x_imag = x[:, 1, :, :].view(len(x),-1)
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1)
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1)
    power = torch.sum(x_real**2 + x_imag**2, 1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, 1)
    nmse = mse/power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse
"""

def NMSE_cuda_for_real(x_hat, x):
    power = torch.sum(x**2)
    mse = torch.sum((x_hat - x)**2)
    nmse = mse/power
    return nmse


class NMSELoss_for_real(nn.Module):
    def __init__(self):
        super(NMSELoss_for_real, self).__init__()

    def forward(self, x_hat, x):
        nmse = NMSE_cuda_for_real(x_hat, x)

        return nmse

class DatasetFolder(Dataset):
    def __init__(self, matData, matLable):
        self.matdata = matData
        self.matlable = matLable

    def __len__(self):
        return self.matdata.shape[0]

    def __getitem__(self, index):
        return self.matdata[index], self.matlable[index]  # , self.matdata[index]

# dataLoader
class DatasetFolder3(Dataset):
    def __init__(self, matData,matNoise,matLable):
        self.matdata = matData
        self.matnoise = matNoise
        self.matlable = matLable


    def __len__(self):
        return self.matdata.shape[0]

    def __getitem__(self, index):
        return self.matdata[index], self.matnoise[index], self.matlable[index]  # , self.matdata[index]