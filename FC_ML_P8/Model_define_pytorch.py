import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict



class FC(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim,n_blocks):
        super(FC, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(h_dim)
            nn.Dropout(p=0.5)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),  nn.Dropout(p=0.5)]) # nn.BatchNorm1d(h_dim), nn.Dropout(p=0.5)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(h_dim, out_dim)


    def forward(self, x, input_dim, output_dim):
        x = x.view(-1, input_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 2, 2, 2, int(output_dim/8))
        return x




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

# class NMSELoss(nn.Module):
#     def __init__(self, reduction='sum'):
#         super(NMSELoss, self).__init__()
#         self.reduction = reduction
#
#     def forward(self, x_hat, x):
#         x_hat = x_hat[:, 0, :, :, :] + 1j * x_hat[:, 1, :, :, :]
#         x = x[:, 0, :, :, :] + 1j * x[:, 1, :, :, :]
#         nmse = torch.sum(abs(x_hat - x) ** 2) / torch.sum(abs(x) ** 2)
#         # if self.reduction == 'mean':
#         #     nmse = torch.mean(nmse)
#         # else:
#         #     nmse = torch.sum(nmse)
#         return nmse


# dataLoader
class DatasetFolder(Dataset):
    def __init__(self, matData,matLable):
        self.matdata = matData
        self.matlable = matLable


    def __len__(self):
        return self.matdata.shape[0]

    def __getitem__(self, index):
        return self.matdata[index], self.matlable[index]  # , self.matdata[index]