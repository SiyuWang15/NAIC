import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from H_utils import *

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
        YY = MIMO([bits0, bits1], HH, 12, 0, 32)
        XX = np.concatenate([bits0, bits1], 0)[:16]
        return YY, XX
    
    def __len__(self):
        return len(self.H)
        
def get_data(random = False, x_part, x_dim):
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
        
# if __name__ == "__main__":
#     H = np.load('./dataset/H_data.npy')
#     X = np.load('./dataset/X_bin.npy')
#     np.save('./dataset/H_part.npy', H[:10000, :, :])
#     np.save('./dataset/X_part.npy', X[:10000, :])
