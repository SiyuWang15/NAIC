import numpy as np 
import random
import torch
from torch.utils.data import Dataset
from .Communication import MIMO

class RandomYXDataset(Dataset):
    def __init__(self, H, mode, Pn):
        super().__init__()
        self.mode = mode
        self.Pn = Pn
        self.H = H
    
    def __getitem__(self, index):
        SNRdb = np.random.uniform(8, 12)
        HH = self.H[index]
        bits0 = np.random.binomial(1, 0.5, size=(128*4,))
        bits1 = np.random.binomial(1, 0.5, size=(128*4,))
        YY = MIMO([bits0, bits1], HH, SNRdb, self.mode, self.Pn) / 20.
        YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        YY = np.reshape(YY, [4, 2, 256], order = 'F') # channel 4: real and imag, pilot and data
        YY = np.reshape(YY, [4, 16, 32], order = 'F') 
        XX = np.stack([bits0, bits1], axis = 0) # 2x512 channel 2:real and imag
        XX = np.reshape(XX, [2, 16, 32])
        return YY.astype('float32'), XX.astype('float32')
    
    def __len__(self):
        return len(self.H)

class Yp2modeDataset(Dataset):
    def __init__(self, Yp_modes):
        super().__init__()
        self.Yp_modes = Yp_modes
    
    def __len__(self):
        return len(self.Yp_modes)
    
    def __getitem__(self, index):
        d = self.Yp_modes[index]
        Yp = d[0]
        mode = d[1]
        return Yp, mode

class RandomYModeDataset(Dataset):
    def __init__(self, H, Pn):
        assert H.shape[1:] == (4, 32) # complex Nsx4x32
        self.H = H
        self.Pn = Pn
        print(f'This is a random generated (Y, mode) pair dataset for Pn={Pn}')
    
    def __len__(self):
        return len(self.H)

    def __getitem__(self, index):
        HH = self.H[index]
        # mode =  random.randint(0, 2)
        mode = 0 if np.random.rand() < 0.8 else 2
        SNRdb = np.random.uniform(8, 12)
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, self.Pn) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order='F')
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        return Yp, int(mode != 0)

    
# class YHDataset(Dataset):
#     def __init__(self, Y, H):
#         self.Y = Y
#         self.H = H
    
#     def __len__(self):
#         return len(self.Y)
    
#     def __getitem__(self, index):
#         YY = self.Y[index, :]
#         HH = self.H[index, :]
#         return YY, HH

class RandomYHDataset(Dataset):
    def __init__(self, H, mode, Pn, cnn:bool=False):
        assert H.shape[1:] == (4, 32)
        self.H = H
        self.mode = mode
        self.Pn = Pn
        self.cnn = cnn
        H_real=np.real(H)
        H_imag=np.imag(H)
        H_label=np.stack([H_imag, H_real], axis = 1)
        assert H_label.shape[1:] == (2,4,32)
        self.H_label = np.reshape(H_label, [len(H), -1]).astype('float32')
        print(f'This is a random generated (Yp, H) pair dataset for mode={mode}, Pn={Pn}')
    
    def __len__(self):
        return len(self.H)
    
    def __getitem__(self, index):
        HH = self.H[index]
        SNRdb = np.random.uniform(8, 12)
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        YY = MIMO([bits0, bits1], HH, SNRdb, self.mode, self.Pn) / 20.
        YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        if self.cnn:
            Yp = YY[:, 0, :, :].reshape(2, 16, 32, order = 'F') + np.random.randn(2, 16, 32) * 0.05     # for cnn model, input Yp should be Nsx2x2x256
        else:
            Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        return Yp.astype('float32'), self.H_label[index].astype('float32')

class RandomYHDataset4CNN(Dataset):
    def __init__(self, H, mode, Pn):
        assert H.shape[1:] == (4, 32)
        self.H = H
        self.mode = mode
        self.Pn = Pn
    
    def __len__(self):
        return len(self.H)