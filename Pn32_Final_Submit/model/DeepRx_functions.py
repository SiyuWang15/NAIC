from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, recompute_scale_factor = True)
        x = self.conv(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=True)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))
        
class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.channels = 32
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(self.channels, self.channels, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(self.channels, self.channels, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(self.channels, self.channels, [3, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(self.channels, self.channels, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(self.channels, self.channels, [3, 1])),
        ]))
        self.conv1x1 = ConvBN(self.channels * 2, self.channels, 1)
        #self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        #identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + x)
        return out


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model
        :param weight_decay:
        :param p: 
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
    def to(self,device):
        '''
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
    def forward(self, model):
        self.weight_list=self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
    def get_weight(self,model):
        '''
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        :param weight_list:
        :param p:
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
    def weight_info(self,weight_list):
        '''
        :param weight_list:
        :return:
        '''
#        print("---------------regularization weight---------------")
#        for name ,w in weight_list:
#            print(name)
#        print("---------------------------------------------------")
def NMSE_cuda(x_hat, x):
    x_real = x[:, 0, :, :].view(len(x),-1)
    x_imag = x[:, 1, :, :].view(len(x),-1)
    x_hat_real = x_hat[:, 0, :, :].contiguous().view(len(x_hat), -1)
    x_hat_imag = x_hat[:, 1, :, :].contiguous().view(len(x_hat), -1)
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