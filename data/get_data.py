import numpy as np 
import random
import torch
from torch.utils.data import Dataset
import os 
import logging
import struct

from .datasets import *
from .Communication import *

dataset_prefix = '/data/siyu/NAIC/'

def get_Yp_modes(Pn):
    data_path = os.path.join(dataset_prefix, 'dataset/random_mode/Yp2mode_Pilot{}.npy'.format(Pn))
    Yp = np.load(data_path, allow_pickle=True)
    split = int(len(Yp) * 0.9)
    train = Yp[:split]
    val = Yp[split:]
    train_set = Yp2modeDataset(train)
    val_set = Yp2modeDataset(val)
    return train_set, val_set

def get_Yp_modes_random(Pn):
    data_path = os.path.join(dataset_prefix, 'dataset/H_data.npy')
    H = np.load(data_path, allow_pickle = True)
    split = int(len(H) * 0.9)
    train = H[:split]
    val = H[split:]
    train_set = RandomYModeDataset(H, Pn)
    val_set = RandomYModeDataset(H, Pn)
    return train_set, val_set

def get_YH_data(mode, Pilotnum, H_domain = 'time'):
    # H: 32w x 4 x 256  complex number
    Yp_path = os.path.join(dataset_prefix, 'dataset/YHdata2/mode_{}_P_{}.npy'.format(mode, Pilotnum))
    H_path = os.path.join(dataset_prefix, 'dataset/YHdata/H_data.npy')
    Yp = np.load(Yp_path).astype('float32')
    H = np.load(H_path) # Nsx4x32 complex 
    H_real = np.real(H)
    H_imag = np.imag(H)
    H = np.stack([H_imag, H_real], axis=1) # Nsx2x4x32 
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

def get_YH_data_random(mode, Pn):
    H_path = os.path.join(dataset_prefix, 'dataset/H_data.npy')
    H = np.load(H_path)
    split = int(0.9*len(H))
    H_train = H[:split]
    H_val = H[split:]
    train_set = RandomYHDataset(H_train, mode, Pn)
    val_set = RandomYHDataset(H_val, mode, Pn)
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
    logging.info(f'loading test data from {dp}')
    Y = np.loadtxt(dp, dtype = np.str, delimiter=',')
    Y = Y.astype(np.float32) # 10000x2048
    Y = np.reshape(Y, (-1, 2, 2, 2, 256), order = 'F')
    Yp = np.reshape(Y[:, :, 0, :, :], [len(Y), 1024], order = 'F')
    Yd = np.reshape(Y[:, :, 1, :, :], [len(Y), 1024], order = 'F')
    return Yp, Yd # Nsx1024

def get_val_data(Pn): # generate validation dataset based on H_val.bin
    # return
    #   Yp and Yd: Nsx1024 order F
    #   X: Nsx1024
    #   H: Nsx4x32 complex number can be directly fed into MIMO
    H_path = os.path.join(dataset_prefix, 'dataset/H_val.bin')
    H_data = open(H_path, 'rb')
    H = struct.unpack('f'*2*2*2*32*2000, H_data.read(4*2*2*2*32*2000))
    H = np.reshape(H, [2000, 2, 4, 32]).astype('float32')
    H_label = np.reshape(H, [len(H), -1])
    H = H[:, 1, :, :] + 1j*H[:, 0, :, :]
    modes = []
    X = []
    Yp = []
    Yd = []
    th = 0.97 if Pn == 8 else 0.999
    for i in range(len(H)):
        # SNRdb = random.randint(8, 12)
        SNRdb = np.random.uniform(8, 12)
        # mode = random.randint(0, 2)
        mode = 0 if np.random.rand() < th else 2
        modes.append(mode)
        bits0 = np.random.binomial(1, 0.5, size = (128*4, ))
        bits1 = np.random.binomial(1, 0.5, size = (128*4, ))
        HH = H[i, :, :]
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, Pn) / 20.
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
    modes = np.array(modes)
    return Yp, Yd,  X, H_label