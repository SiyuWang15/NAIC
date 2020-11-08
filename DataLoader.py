import numpy as np 
import random
import torch
from torch.utils.data import Dataset, DataLoader
from H_utils import *
import scipy.io as scio

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
        YY = MIMO(temp_X, HH, 12, 0, 32) / 20
        return YY, XX[x_dim * x_part:x_dim * (x_part+1)]

class RandomDataset(Dataset):
    def __init__(self, H):
        super().__init__()
        self.H = H
    
    def __getitem__(self, index):
        HH = self.H[index]
        bits0 = np.random.binomial(1, 0.5, size = (128*4, ))
        bits1 = np.random.binomial(1, 0.5, size = (128*4, ))
        YY = MIMO([bits0, bits1], HH, 12, 0, 32) / 20
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
    def __init__(self, YH):
        self.YH = YH
    
    def __len__(self):
        return len(self.YH)
    
    def __getitem__(self, index):
        d = self.YH[index]
        Y = d[0]
        H = d[1]
        return Y, H
        
def make_data():
    H = np.load('./dataset/H_data.npy')
    Yp2mode = []
    for i in range(len(H)):
        HH = H[i, :, :]
        mode = random.randint(0, 2)
        SNRdb = random.randint(8, 12)
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, 8)
        YY = np.reshape(YY, [2, 2, 2, 256], order='F')
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        Yp2mode.append((Yp, mode))
        if i % 10000 == 0:
            print('%d complete.' % i)
    
    np.save('/data/siyu/NAIC/dataset/random_mode/Yp2mode_Pilot8.npy', Yp2mode, allow_pickle=True)

def get_Yp_modes():
    Yp = np.load('/data/siyu/NAIC/dataset/random_mode/Yp2mode_Pilot8.npy', allow_pickle=True)
    split = int(len(Yp) * 0.9)
    train = Yp[:split]
    val = Yp[split:]
    train_set = Yp2modeDataset(train)
    val_set = Yp2modeDataset(val)
    return train_set, val_set

def get_data( x_part, x_dim, random = False):
    H = np.load('./dataset/H_data.npy')
    split = int(0.9*len(H))
    H_train = H[:split, :, :]
    H_val = H[split:, :, :]

    if random:
        train_set = RandomDataset(H_train, x_part, x_dim)
        val_set = RandomDataset(H_val, x_part, x_dim)
    else:
        X = np.load('./dataset/X_bin.npy')
        X_train = X[:split, :]
        X_val = X[split:, :]
        train_set = dataset(X_train, H_train, x_part, x_dim)
        val_set = dataset(X_val, H_val, x_part, x_dim)
    return train_set, val_set
        
if __name__ == "__main__":
    # H = np.load('./dataset/H_data.npy')
    # X = np.load('./dataset/X_bin.npy')
    # np.save('./dataset/H_part.npy', H[:10000, :, :])
    # np.save('./dataset/X_part.npy', X[:10000, :])
    make_data()
    # get_Yp_modes()