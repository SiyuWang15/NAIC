from utils import *
import struct
import torch.nn as nn
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat

from Model_define_pytorch import FC_Estimation, NMSELoss, DatasetFolder
from LS_CE import LS_Estimation,MMSE_Estimation, Interpolation_f

from MLreceiver import MLReceiver, MakeCodebook
from SoftMLreceiver import SoftMLReceiver
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

# 生成测试数据 Y：-1*2048； X：-1*1024； H：-1*4*32 时域信道
Y, X, H = generatorXY(batch_num,H_val,Pilot_num)


# 完美的频域信道
Hf = np.fft.fft(H, 256)/20
Hf = np.reshape(Hf, (-1, 2,2,256), order='F')

# 基于LS估计导频位置上的频域信道
Hf_partial = LS_Estimation(Y,Pilot_num)
# Hf_partial = MMSE_Estimation(Y,Pilot_num)


#### 通过部分频域信道估计全频域信道 ####

#### The first method
# Model Construction
model = FC_Estimation(2048, 4096, 4096, 2048)
model = torch.nn.DataParallel(model).cuda()  # model.module
# Load weights
model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_for_'+str(Pilot_num)+'.pth.tar'
model.load_state_dict(torch.load(model_path)['state_dict'])
print("Weight Loaded!")
# 通过网络估计全频域信道
# complex ---> real + imag
Hf_input = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_input[:,0,:,:,:] = Hf_partial.real
Hf_input[:,1,:,:,:] = Hf_partial.imag
model.eval()
Hf_output = model(torch.tensor(Hf_input).to('cuda'))
criterion = NMSELoss(reduction='mean')

Hf_label = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_label[:,0,:,:,:] = Hf.real
Hf_label[:,1,:,:,:] = Hf.imag
loss = criterion(Hf_output.cuda(), torch.tensor(Hf_label).cuda())
print(loss)

# Hf_output1 = Hf_output.cuda().data.cpu().numpy()
Hf_output1 = Hf_output.detach().cpu().numpy()
Hf_hat = Hf_output1[:,0,:,:,:]+1j*Hf_output1[:,1,:,:,:]


# #### The second method
# Hf_hat1 = Interpolation_f(Hf,Pilot_num)
# Hf_hat2 = Interpolation_f(Hf_partial2,Pilot_num)

NMSE1 = np.sum(abs(Hf_hat-Hf)**2)/np.sum(abs(Hf)**2)
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
# X_ML2, X_bits2 = SoftMLReceiver(Y,Hf_hat)

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









