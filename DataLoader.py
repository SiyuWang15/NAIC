import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from H_utils import *

class dataset(Dataset):
    def __init__(self, X, H):
        super().__init__()
        assert len(X) == len(H)
        self.X = X
        self.H = H

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        XX = self.X[index].astype('float')
        HH = self.H[index]
        temp_X = [XX[:512], XX[512:]]
        YY = MIMO(temp_X, HH, 12, 0, 32) / 20
        return YY, XX[:16]

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
        
def get_data(random = False):
    H = np.load('./dataset/H_part.npy')
    split = int(0.9*len(H))
    H_train = H[:split, :, :]
    H_val = H[split:, :, :]

    if random:
        train_set = RandomDataset(H_train)
        val_set = RandomDataset(H_val)
    else:
        X = np.load('./dataset/X_part.npy')
        X_train = X[:split, :]
        X_val = X[split:, :]
        train_set = dataset(X_train, H_train)
        val_set = dataset(X_val, H_val)
    return train_set, val_set
        
if __name__ == "__main__":
    H = np.load('./dataset/H_data.npy')
    X = np.load('./dataset/X_bin.npy')
    np.save('./dataset/H_part.npy', H[:10000, :, :])
    np.save('./dataset/X_part.npy', X[:10000, :])
# class DataLoader():
#     def __init__(self, X_path = './dataset/X_bin.npy', H_path = './dataset/H_data.npy'):
#         X = np.load(X_path)
#         H = np.load(H_path)
#         assert len(X) == len(H)
#         split = int(0.9* len(X))
#         self.train_num = split
#         self.val_num = len(X) - split
#         self.X_train = X[:split, :]
#         self.X_val = X[split:, :]
#         self.H_train = H[:split, :, :]
#         self.H_val = H[split:, :, :]
#         self.train_ind = 0
#         self.val_ind = 0
    
#     def get_train_data(self, batch_size, device):
#         labels = []
#         samples = []
#         end = False
#         for i in range(batch_size):
#             HH = self.H_train[self.train_ind]
#             XX = self.X_train[self.train_ind].astype('float')
#             XX = [XX[:512], XX[512:]]
#             YY = MIMO(XX, HH, 12, 0, 32) / 20
#             XX = np.concatenate(XX, 0)[:16]
#             labels.append(XX)
#             samples.append(YY)
#             self.train_ind += 1
#             if self.train_ind == self.train_num:
#                 self.train_ind = 0
#                 end = True
#         batch_y = torch.from_numpy(np.asarray(samples)).float().to(device)
#         batch_x = torch.from_numpy(np.asarray(labels)).float().to(device)
#         return batch_y, batch_x, end

#     def get_val_data(self, batch_size, device):
#         labels = []
#         samples = []
#         for i in range(batch_size):
#             HH = self.H_val[self.val_ind]
#             XX = self.X_val[self.val_ind].astype('float')
#             XX = [XX[:512], XX[512:]]
#             YY = MIMO(XX, HH, 12, 0, 32) / 20
#             XX = np.concatenate(XX, 0)[:16]
#             labels.append(XX)
#             samples.append(YY)
#             self.val_ind += 1
#             if self.val_ind == self.val_num:
#                 self.val_ind = 0
#         batch_y = torch.from_numpy(np.asarray(samples)).float().to(device)
#         batch_x = torch.from_numpy(np.asarray(labels)).float().to(device)
#         return batch_y, batch_x
