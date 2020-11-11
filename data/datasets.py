import numpy as np 
import random
import torch
from torch.utils.data import Dataset
from .Communication import MIMO

class YXdataset(Dataset):
    def __init__(self, X, H, x_part, x_dim):
        super().__init__()
        assert len(X) == len(H)
        self.X = X
        self.H = H
        self.x_part = x_part

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        XX = self.X[index].astype('float')
        HH = self.H[index]
        temp_X = [XX[:512], XX[512:]]
        YY = MIMO(temp_X, HH, 12, 0, 32) / 20.
        return YY, XX[x_dim * x_part:x_dim * (x_part+1)]

class RandomDataset(Dataset):
    def __init__(self, H):
        super().__init__()
        self.H = H
    
    def __getitem__(self, index):
        HH = self.H[index]
        bits0 = np.random.binomial(1, 0.5, size = (128*4, ))
        bits1 = np.random.binomial(1, 0.5, size = (128*4, ))
        YY = MIMO([bits0, bits1], HH, 12, 0, 32) / 20.
        XX = np.concatenate([bits0, bits1], 0)[:16]
        return YY, XX
    
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
        mode =  random.randint(0, 2)
        SNRdb = np.random.uniform(8, 12)
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, self.Pn) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order='F')
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        return Yp, mode

    
class YHDataset(Dataset):
    def __init__(self, Y, H):
        self.Y = Y
        self.H = H
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        YY = self.Y[index, :]
        HH = self.H[index, :]
        return YY, HH

class RandomYHDataset(Dataset):
    def __init__(self, H, mode, Pn):
        assert H.shape[1:] == (4, 32)
        self.H = H
        self.mode = mode
        self.Pn = Pn
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
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        return Yp.astype('float32'), self.H_label[index].astype('float32')