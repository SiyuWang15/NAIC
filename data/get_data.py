import numpy as np 
import random
import torch
from torch.utils.data import Dataset
import os 
import logging
import struct

from .datasets import *
from .Communication import *

dataset_prefix = '/data/siyu/NAIC/dataset'

def get_YH_data_random(mode, Pn):
    N1 = 320000
    data1 = open('/data/siyu/NAIC/dataset/H.bin','rb')
    H1 = struct.unpack('f'*2*2*2*32*N1,data1.read(4*2*2*2*32*N1))
    H1 = np.reshape(H1,[N1,2,4,32])
    H_tra = H1[:,1,:,:]+1j*H1[:,0,:,:]   # time-domain channel for training 

    data2 = open('/data/siyu/NAIC/dataset/H_val.bin','rb')
    H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
    H2 = np.reshape(H2,[2000,2,4,32])
    H_val = H2[:,1,:,:]+1j*H2[:,0,:,:] 
    trainset = RandomDataset(H_tra, Pilot_num=Pn, mode = mode)
    valset = RandomDataset(H_val, Pilot_num=Pn, mode=mode)
    return trainset, valset

def get_test_data(Pn):
    tag = 1 if Pn == 32 else 2
    dp = os.path.join(dataset_prefix, f'Y_{tag}.csv')
    logging.info(f'loading test data from {dp}')
    Y = np.loadtxt(dp, dtype = np.str, delimiter=',')
    Y = Y.astype(np.float32) # 10000x2048
    Y = np.reshape(Y, (-1, 2, 2, 2, 256), order = 'F')
    return Y 


def get_val_data(Pn, mode):
    H_path = os.path.join(dataset_prefix, 'H_val.bin')
    H_data = open(H_path, 'rb')
    H = struct.unpack('f'*2*2*2*32*2000, H_data.read(4*2*2*2*32*2000))
    H = np.reshape(H, [2000, 2, 4, 32]).astype('float32')
    # H_label = np.reshape(H, [len(H), -1])
    H = H[:, 1, :, :] + 1j*H[:, 0, :, :]
    val_set = RandomDataset(H, Pilot_num=Pn, mode = mode)
    return val_set

def get_denoise_data(mode, Pn):
    N1 = 320000
    data1 = open('/data/siyu/NAIC/dataset/H.bin','rb')
    H1 = struct.unpack('f'*2*2*2*32*N1,data1.read(4*2*2*2*32*N1))
    H1 = np.reshape(H1,[N1,2,4,32])
    H_tra = H1[:,1,:,:]+1j*H1[:,0,:,:]   # time-domain channel for training 

    data2 = open('/data/siyu/NAIC/dataset/H_val.bin','rb')
    H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
    H2 = np.reshape(H2,[2000,2,4,32])
    H_val = H2[:,1,:,:]+1j*H2[:,0,:,:] 
    trainset = DenoiseDataset(H_tra, Pn=Pn)
    valset = DenoiseDataset(H_val, Pn=Pn)
    return trainset, valset