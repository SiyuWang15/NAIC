import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict


# class FC_Detection(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, N_groups = 1):
#         super(FC_Detection, self).__init__()
#         self.N_groups = N_groups
#         self.BasicBlocks = []

#         for idx in range(self.N_groups):
#             self.BasicBlocks.append( BasicBlock(in_dim, out_dim, hidden_dim) )

    
        

#     def forward(self, x):

#         for idx in range(self.N_groups):
#             self.BasicBlocks[idx]( x[:,idx,:] ) 


#         return x

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

    input = torch.randn(256, 1024)

    in_dim = 1024
    h_dim = 4096
    out_dim = 256
    n_blocks =  2
    model = FC_Estimation(in_dim, h_dim, out_dim, n_blocks)
    output = model(input)

    print('Lovelive')