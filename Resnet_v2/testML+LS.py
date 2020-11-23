from utils import *
import struct
import torch.nn as nn
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat


from MLreceiver import MLReceiver
from LSreveiver import LSequalization
from generate_data import generatorXY

# Parameters for training
gpu_list = '4,5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

data1=open('/data/CuiMingyao/AI_competition/OFDMReceiver/H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1=np.reshape(H1,[320000,2,4,32])
H = H1[:,1,:,:]+1j*H1[:,0,:,:]

data2 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]

Pilot_num = 8
batch_num = 2000
SNRdb = 10
mode = 0
# 生成测试数据 Y：-1*2048； X：-1*1024； H：-1*4*32 时域信道
Y, X, H = generatorXY(batch_num,H_val,Pilot_num, SNR=SNRdb, m=mode)



# 完美的频域信道
Hf = np.fft.fft(H, 256)/20
Hf = np.reshape(Hf, (-1, 2,2,256), order='F')


#### ML的输入 ####

Y = np.reshape(Y, (-1, 2,2,2,256), order = 'F')
Yp = Y[:,:,0,:,:]
Yp = Yp[:,0,:,:] + 1j*Yp[:,1,:,:]

Yd = Y[:,:,1,:,:]
Yd = Yd[:,0,:,:] + 1j*Yd[:,1,:,:]

#### 基于ML恢复发送比特流X ####
X_LS = LSequalization(Hf, Yd)

#### 基于ML恢复发送比特流X ####
X_ML2, X_bits2 = MLReceiver(Yd,Hf)



error2 = X_bits2 - X
bit_error2 = np.sum(np.sum(np.abs(error2)<0.1))/ error2.size
print(bit_error2)






error2 = X_bits2 - X
bit_error2 = np.sum(np.sum(np.abs(error2)<0.1))/ error2.size
print(bit_error2)

# X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
# X_1.tofile('./X_1.bin')

# print(np.sum(bit_error==0)/len(bit_error))









