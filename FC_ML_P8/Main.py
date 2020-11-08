from utils import *
import struct
import torch.nn as nn
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat

from Model_define_pytorch import FC_Estimation, NMSELoss, DatasetFolder, FC_Estimation_f2c, FC_Estimation_t2f, FC
from LS_CE import LS_Estimation,MMSE_Estimation, Interpolation_f, LS_Estimation_Partial

from MLreceiver import MLReceiver, MakeCodebook
from generate_data import generatorXY

# Parameters for training
gpu_list = '5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

data1=open('H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1=np.reshape(H1,[320000,2,4,32])
H = H1[:,1,:,:]+1j*H1[:,0,:,:]

data2 = open('H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]

Pilot_num = 8
batch_num = 2000

# # 生成测试数据 Y：-1*2048； X：-1*1024； H：-1*4*32 时域信道
Y, X, H = generatorXY(batch_num,H_val,Pilot_num, -1,-1)

# data_load_address = '/data/AI_Wireless/'
# Y = np.load(data_load_address+'validation_Y_P=8.npy')
# Ns = Y.shape[0]
# H = H_val

# 完美的频域信道
Hf = np.fft.fft(H, 256)/20
Hf = np.reshape(Hf, (-1, 2,2,256), order='F')

# 基于LS估计导频位置上的频域信道
Hf_partial = LS_Estimation(Y,Pilot_num)

#### 通过部分频域信道估计全频域信道 ####

#### The first method
in_dim = 2048
h_dim = 4096
out_dim = 2048
n_blocks =  2

# Model Construction
model = FC(in_dim, h_dim, out_dim, n_blocks)
model = torch.nn.DataParallel(model).cuda()  # model.module
# Load weights
model_path = './Modelsave/FC_Estimation_f2f_2048_4096_2048_2.pth.tar'
# model_path = './Modelsave/FC_Estimation_for_8_SNR=100_mode=0_off.pth.tar'
model.load_state_dict(torch.load(model_path)['state_dict'])
print("Weight Loaded!")

# complex ---> real + imag
Hf_input = np.zeros(shape=[batch_num, 2, 2, 2, 256], dtype=np.float32)
Hf_input[:, 0, :, :, :] = Hf_partial.real
Hf_input[:, 1, :, :, :] = Hf_partial.imag
model.eval()
Hf_output = model(torch.tensor(Hf_input).to('cuda'), in_dim, out_dim)
criterion = NMSELoss(reduction='mean')

# complex ---> real +imag
Hf_label = np.zeros(shape=[batch_num, 2, 2, 2, 256], dtype=np.float32)
Hf_label[:, 0, :, :, :] = Hf.real
Hf_label[:, 1, :, :, :] = Hf.imag
loss = criterion(Hf_output.cuda(), torch.tensor(Hf_label).cuda())
print(loss)

Hf_output1 = Hf_output.cuda().data.cpu().numpy()
Hf_hat = Hf_output1[:,0,:,:,:]+1j*Hf_output1[:,1,:,:,:]

# #### The second method
# Hf_hat1 = Interpolation_f(Hf,Pilot_num)
# Hf_hat2 = Interpolation_f(Hf_partial2,Pilot_num)

NMSE1 = np.mean(abs(Hf_hat-Hf)**2)/np.mean(abs(Hf)**2)
print(NMSE1)
# NMSE2 = np.sum(abs(Hf_hat2-Hf)**2)/np.sum(abs(Hf)**2)
# print(NMSE2)

#### ML的输入 ####

Y = np.reshape(Y, (-1, 2,2,2,256), order = 'F')
Y = Y [:,0,:,:,:] + 1j*Y [:,1,:,:,:]
Y = Y[:,1,:,:]


# 生成码本
G = 4
codebook = MakeCodebook(G)

#### 基于ML恢复发送比特流X ####

# X_ML1, X_bits1 = MLReceiver(Y,Hf,codebook)
X_ML2, X_bits2 = MLReceiver(Y,Hf_hat,codebook)

# 可用的复数input：样本数 * 发射天线数 * 子载波数
X1 = np.reshape(X, (-1, 2, 512))
input = np.zeros((batch_num, 2,256), dtype=complex)
for num in range(batch_num):
    for idx in range(2):
        bits = X1[num, idx, :]
        # 与 utils文件里的modulation一模一样的操作
        bits = np.reshape(bits, (256,2))
        input[num, idx,:] = 0.7071 * (2 * bits[:, 0] - 1) + 0.7071j * (2 * bits[:, 1] - 1)

# error1 = X_ML1 - input
# bit_error1 = np.sum(np.sum(np.abs(error1)<0.1))/ error1.size
# print(bit_error1)

# error1 = X_bits1 - X
# bit_error1 = np.sum(np.sum(np.abs(error1)<0.1))/ error1.size
# print(bit_error1)

# error2 = X_ML2 - input
# bit_error2 = np.sum(np.sum(np.abs(error2)<0.1))/ error2.size
# print(bit_error2)

error2 = X_bits2 - X
bit_error2 = np.sum(np.sum(np.abs(error2)<0.1))/ error2.size
print(bit_error2)

# X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
# X_1.tofile('./X_1.bin')

# print(np.sum(bit_error==0)/len(bit_error))









