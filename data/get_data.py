import numpy as np 
import random
import torch
from torch.utils.data import Dataset
import os 
import struct

from .Communication import *

dataset_prefix = '/data/siyu/NAIC/'

class dataset(Dataset):
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

def get_Yp_modes(Pn):
    data_path = os.path.join(dataset_prefix, 'dataset/random_mode/Yp2mode_Pilot{}.npy'.format(Pn))
    Yp = np.load(data_path, allow_pickle=True)
    split = int(len(Yp) * 0.9)
    train = Yp[:split]
    val = Yp[split:]
    train_set = Yp2modeDataset(train)
    val_set = Yp2modeDataset(val)
    return train_set, val_set

def get_YH_data(mode, Pilotnum, H_domain = 'time'):
    # H: 32w x 4 x 256  complex number
    Yp_path = os.path.join(dataset_prefix, 'dataset/YHdata2/mode_{}_P_{}.npy'.format(mode, Pilotnum))
    H_path = os.path.join(dataset_prefix, 'dataset/YHdata/H_data.npy')
    Yp = np.load(Yp_path).astype('float32')
    H = np.load(H_path) # Nsx4x32 complex 
    H_real = np.real(H)
    H_imag = np.imag(H)
    H = np.stack([H_imag, H_real], axis=1)
    H = H.reshape(len(H), -1).astype('float32') # Ns * 256
    assert len(Yp) == len(H)
    split = int(len(H) * 0.9)
    H_train = H[:split, :]
    H_val = H[split:, :]
    Yp_train = Yp[:split, :]
    Yp_val = Yp[split:, :]
    train_set = YHDataset(Yp_train, H_train)
    val_set = YHDataset(Yp_val, H_val)
    return train_set, val_set
    
def get_YX_data( x_part, x_dim, random = False):
    H = np.load(os.path.join(dataset_prefix, 'dataset/H_data.npy'))
    split = int(0.9*len(H))
    H_train = H[:split, :, :].astype('float32')
    H_val = H[split:, :, :].astype('float32')

    if random:
        train_set = RandomDataset(H_train, x_part, x_dim)
        val_set = RandomDataset(H_val, x_part, x_dim)
    else:
        X = np.load(os.path.join(dataset_prefix, 'dataset/X_bin.npy'))
        X_train = X[:split, :]
        X_val = X[split:, :]
        train_set = dataset(X_train, H_train, x_part, x_dim)
        val_set = dataset(X_val, H_val, x_part, x_dim)
    return train_set, val_set
        
def get_test_data(Pn):
    tag = 1 if Pn == 32 else 2
    dp = os.path.join(dataset_prefix, f'dataset/Y_{tag}.csv')
    Y = np.loadtxt(dp, dtype = np.str, delimiter=',')
    Y = Y.astype(np.float32)
    Y = np.reshape(Y, (-1, 2, 2, 2, 256), order = 'F')
    Yp = np.reshape(Y[:, :, 0, :, :], [len(Y), 1024], order = 'F')
    Yd = np.reshape(Y[:, :, 1, :, :], [len(Y), 1024], order = 'F')
    return Yp, Yd

def get_val_data(Pn): # generate validation dataset based on H_val.bin
    # return
    #   Yp and Yd: Nsx1024 order F
    #   X: Nsx1024
    #   H: Nsx4x32 complex number can be directly fed into MIMO
    H_path = os.path.join(dataset_prefix, 'dataset/H_val.bin')
    H_data = open(H_path, 'rb')
    H = struct.unpack('f'*2*2*2*32*2000, H_data.read(4*2*2*2*32*2000))
    H = np.reshape(H, [2000, 2, 4, 32]).astype('float32')
    H = H[:, 1, :, :] + 1j*H[:, 0, :, :]
    X = []
    Yp = []
    Yd = []
    for i in range(len(H)):
        SNRdb = random.randint(8, 12)
        mode = random.randint(0, 2)
        bits0 = np.random.binomial(1, 0.5, size = (128*4, ))
        bits1 = np.random.binomial(1, 0.5, size = (128*4, ))
        HH = H[i, :, :]
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, Pn) / 20
        YY = np.reshape(YY,  [2, 2, 2, 256], order = 'F')
        YYp = YY[:, 0, :, :].reshape(1024, order = 'F')
        YYd = YY[:, 1, :, :].reshape(1024, order = 'F')
        XX = np.concatenate([bits0, bits1], 0)
        X.append(XX)
        Yp.append(YYp)
        Yd.append(YYd)
    Yp = np.stack(Yp, axis=0).astype('float32')
    Yd = np.stack(Yd, axis=0).astype('float32')
    X = np.stack(X, axis=0).astype('float32')
    return Yp, Yd,  X, H