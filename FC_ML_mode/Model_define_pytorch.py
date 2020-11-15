import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=2, out_ch=1, num_classes=1024):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(out_ch * 6 * 256, num_classes)
    #    self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = out.view(-1, self.out_ch*6*256)
        out = self.fc(out)
        out = torch.sigmoid(out)
        #d1 = self.active(out)

        return out



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


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=1024):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3,
                       stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*4*6, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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





class FC_Detection(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(FC_Detection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, self.hidden_dim[0]),
            nn.BatchNorm1d(self.hidden_dim[0]),
            nn.ReLU(inplace=True))

        self.hidden_layer = nn.Sequential()

        for idx in range(len(self.hidden_dim)-1):
            self.hidden_layer.add_module('fc{0}'.format(idx), nn.Linear(self.hidden_dim[idx], self.hidden_dim[idx+1]))
            self.hidden_layer.add_module('bn{0}'.format(idx), nn.BatchNorm1d(self.hidden_dim[idx+1]))
            # self.hidden_layer.add_module('dropout{0}'.format(idx), nn.Dropout( p = 0.5 ))
            self.hidden_layer.add_module('relu{0}'.format(idx), nn.ReLU(inplace=True))

        self.output = nn.Linear(self.hidden_dim[-1], out_dim)
    
        

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.fc1(x)
        x = self.hidden_layer(x)
        x = self.output(x)
        x = torch.sigmoid(x)        
        return x

class FC_Estimation(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC_Estimation, self).__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            # nn.ReLU(inplace=True),
            nn.ELU(inplace=True),
            # nn.Sigmoid(),
            nn.BatchNorm1d(h_dim)
            # nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            # hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),  nn.Dropout(p=0.5)]) 
            # # nn.BatchNorm1d(h_dim), nn.Dropout(p=0.5)
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU(inplace=True),  nn.BatchNorm1d(h_dim)]) 
            # hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.Sigmoid(),  nn.Dropout(p=0.5)]) 
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x):
        # x = x.view(-1, self.input_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x



class DnCNN(nn.Module):
    def __init__(self, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        channels = 2
        features = 128
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, 2, 4, 256)
        x = self.dncnn(x)
        x = x.view(-1,2,2,2,256)
        return x

def NMSE_cuda(x_hat, x):
    power = torch.sum(x**2)
    mse = torch.sum((x_hat - x)**2)
    nmse = mse/power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)

        return nmse


def NMSE_cuda2(x_hat, x):
    x_real = x[:, 0, :, :].view(len(x),-1)
    x_imag = x[:, 1, :, :].view(len(x),-1)
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1)
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1)
    power = torch.sum(x_real**2 + x_imag**2, 1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, 1)
    nmse = mse/power
    return nmse


class NMSELoss2(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss2, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda2(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse




# dataLoader
class DatasetFolder(Dataset):
    def __init__(self, matData,matLable):
        self.matdata = matData
        self.matlable = matLable


    def __len__(self):
        return self.matdata.shape[0]

    def __getitem__(self, index):
        return self.matdata[index], self.matlable[index]  # , self.matdata[index]



if __name__ == '__main__':

    input = torch.randn(256, 2, 6*8, 32).cuda()
    # model = ResNet().cuda()
    model = U_Net().cuda()

    output = model(input)

    print('Lovelive')