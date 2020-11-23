import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import struct
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import init
import torch.autograd as autograd
import torch.utils.data as Data
import time
import model
from model import Receiver
from functions import Regularization, NMSELoss, ConvBN, CRBlock
import channel_pre_estimation as cpe

# Read the training dataset and the testing dataset

# data1=open('X_1.bin','rb')
# X1=struct.unpack('f'*10000*1024, data1.read(4*10000*1024))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
train = 1
num_pilot = 8
mode = '0'
modev = '0'
print('----------------------------------')
print('  num_pilot:',num_pilot)
print('  training mode:',mode)
print('  validation mode:',modev)
print('----------------------------------')


if train and num_pilot==32:
    print('###############################################')
    print("Loading training data ...")
    X = np.load("/data/HaoJiang_data/AI second part/training dataset/X_m"+mode+"_d10_p"+str(num_pilot)+".npy")
    Y = np.load("/data/HaoJiang_data/AI second part/training dataset/Y_m"+mode+"_d10_p"+str(num_pilot)+".npy")
    #H = np.load("/data/HaoJiang_data/AI second part/training dataset/H_m"+mode+"_d10_p"+str(num_pilot)+".npy")
    num_data = X.shape[0]
    H_hat = []
    for i in range(160):
        hat = cpe.channel_pre_32(Y[i*2000:(i+1)*2000,:], mode, num_pilot)
        print(i, hat.shape)
        H_hat.append(hat)
    H_hat = np.stack(H_hat,0)
    print(H_hat.shape)
    H_hat = H_hat.reshape([num_data,2,2,2,256])
    print(H_hat.shape)
    np.save("/data/HaoJiang_data/AI second part/training dataset/H_hat_m"+mode+"_d10_p32_1.npy", H_hat)
    #H = torch.from_numpy(H[:num_train,:])
    
    Xv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Xv_m"+modev+"_d10_p"+str(num_pilot)+".npy")
    Yv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Yv_m"+modev+"_d10_p"+str(num_pilot)+".npy")
    #Hv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Hv_m"+modev+"_d10_p"+str(num_pilot)+".npy")
    num_test = Xv.shape[0]
    
    Xv = torch.from_numpy(Xv)
    Yv = torch.from_numpy(Yv)
    Hv_hat = cpe.channel_pre_32(Yv, mode, num_pilot)
    print(Hv_hat.shape)
    np.save("/data/HaoJiang_data/AI second part/validation dataset/Hv_hat_m"+mode+"_d10_p32_1.npy", Hv_hat)


elif train and num_pilot==8:
    print('###############################################')
    print("Loading training data ...")
    #X = np.load("/data/HaoJiang_data/AI second part/training dataset/X_m"+mode+"_d12_p"+str(num_pilot)+".npy")
    Y = np.load("/data/HaoJiang_data/AI second part/training dataset/Y_m"+mode+"_d12_p"+str(num_pilot)+".npy")
    #H = np.load("/data/HaoJiang_data/AI second part/training dataset/H_m"+mode+"_d10_p"+str(num_pilot)+".npy")
    num_data = Y.shape[0]
    H_hat = []
    for i in range(160):
        hat = cpe.channel_pre_8(Y[i*2000:(i+1)*2000,:], mode, num_pilot)
        print(i, hat.shape)
        H_hat.append(hat)
    H_hat = np.stack(H_hat,0)
    print(H_hat.shape)
    H_hat = H_hat.reshape([num_data,2,2,2,256])
    print(H_hat.shape)
    np.save("/data/HaoJiang_data/AI second part/training dataset/H_hat_m"+mode+"_d12_p8_1.npy", H_hat)
    #H = torch.from_numpy(H[:num_train,:])
    
    #Xv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Xv_m"+modev+"_d12_p"+str(num_pilot)+".npy")
    Yv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Yv_m"+modev+"_d12_p"+str(num_pilot)+".npy")
    #Hv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Hv_m"+modev+"_d10_p"+str(num_pilot)+".npy")
    num_test = Yv.shape[0]
    
    #Xv = torch.from_numpy(Xv)
    Yv = torch.from_numpy(Yv)
    Hv_hat = cpe.channel_pre_8(Yv, mode, num_pilot)
    print(Hv_hat.shape)
    np.save("/data/HaoJiang_data/AI second part/validation dataset/Hv_hat_m"+mode+"_d12_p8_1.npy", Hv_hat)
