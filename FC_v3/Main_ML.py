from utils import *
import struct
import torch.nn as nn
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat

from Model_define_pytorch import FC_Estimation, NMSELoss, NMSELoss2, DatasetFolder
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

Pilotnum = 8
batch_num = 2000
group_index = list(range(32))
group_num = 32
G = 256 // group_num
N_train_groups = len(group_index)
mode = -1

# 生成测试数据 Y：-1*2048； X：-1*1024； H：-1*4*32 时域信道
Y, X, H = generatorXY(batch_num,H_val,Pilotnum)





#### 通过部分频域信道估计全频域信道 ####

#### The first method
# Model Construction
in_dim = 1024
h_dim = 2048
out_dim = 256
n_blocks =  4
model = FC_Estimation(in_dim, h_dim, out_dim, n_blocks)

model = torch.nn.DataParallel(model).cuda()  # model.module
# Load weights
model_path = model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_Pilot'+str(Pilotnum)+'_mode'+str(mode)+'.pth.tar'
model.load_state_dict(torch.load(model_path)['state_dict'])
print("Weight Loaded!")


# 通过网络估计全频域信道
# complex ---> real + imag

Y_reshape = np.reshape(Y, [batch_num, 2, 2, 2, 256], order='F')
Yp = Y_reshape[:,:,0,:,:]
Yp = np.reshape(Yp, [batch_num, 2*2*256])
Yp = torch.Tensor(Yp).to('cuda')
# print(Yp.shape)
model.eval()
Ht_output = model(Yp)
Ht_output = Ht_output.detach().cpu().numpy()
Ht_reshape = Ht_output.reshape([-1,2,4,32])

Ht_complex = Ht_reshape[:,0,:,:] + 1j*Ht_reshape[:,1,:,:] 
Hf_complex = np.fft.fft(Ht_complex, 256)/20
Hf_reshape = np.reshape(Hf_complex, (-1,2,2,256), order='F')
Hf_output = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_output[:,0,:,:,:] = Hf_reshape.real
Hf_output[:,1,:,:,:] = Hf_reshape.imag




# 完美的频域信道
Hf = np.fft.fft(H, 256)/20
Hf = np.reshape(Hf, (-1, 2,2,256), order='F')
Hf_label = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_label[:,0,:,:,:] = Hf.real
Hf_label[:,1,:,:,:] = Hf.imag
criterion = NMSELoss2(reduction = 'mean')

loss = criterion(torch.tensor(Hf_output).cuda(), torch.tensor(Hf_label).cuda())
print(loss)

# Hf_output1 = Hf_output.cuda().data.cpu().numpy()
Hf_output1 = Hf_output
Hf_hat = Hf_output1[:,0,:,:,:]+1j*Hf_output1[:,1,:,:,:]
NMSE1 = np.zeros((N_train_groups))
for item in range(N_train_groups):
    NMSE1[item] = np.sum(abs(Hf_hat[:,:,:, G*group_index[item]: G*(group_index[item]+1)]\
        -Hf[:,:,:, G*group_index[item]: G*(group_index[item]+1)])**2)/\
        np.sum(abs(Hf[:,:,:, G*group_index[item]: G*(group_index[item]+1)])**2)
    print('Group[%d]'%group_index[item], 'NMSE: %.4f' % NMSE1[item])

NMSE2 = np.sum(abs(Hf_hat-Hf)**2)/np.sum(abs(Hf)**2)
print('avg NMSE:', NMSE2)

#### ML的输入 ####

Y = np.reshape(Y, (-1, 2,2,2,256), order = 'F')
Y = Y [:,0,:,:,:] + 1j*Y [:,1,:,:,:]
Y = Y[:,1,:,:]


# 生成码本
G = 4
codebook = MakeCodebook(G)

#### 基于ML恢复发送比特流X ####

X_ML, X_bits = MLReceiver(Y,Hf_hat,codebook)

# 可用的复数input：样本数 * 发射天线数 * 子载波数
X_reshape = np.reshape(X, (-1, 2, 512))
input = np.zeros((batch_num, 2,256), dtype=complex)
for num in range(batch_num):
    for idx in range(2):
        bits = X_reshape[num, idx, :]
        # 与 utils文件里的modulation一模一样的操作
        bits = np.reshape(bits, (256,2))
        input[num, idx,:] = 0.7071 * (2 * bits[:, 0] - 1) + 0.7071j * (2 * bits[:, 1] - 1)



X_reshape = np.reshape(X , (-1, 2, 512))
X_bits_reshape = np.reshape(X_bits, (-1, 2, 512))
accuracy = np.zeros((N_train_groups))

for item in range(N_train_groups):
    X_item = X_reshape[:, :, 2*G*group_index[item]: 2*G*(group_index[item] + 1)]
    X_bits_item = X_bits_reshape[:, :, 2*G*group_index[item]: 2*G*(group_index[item] + 1)]
    accuracy[item] = np.mean( (np.abs(X_item - X_bits_item)<0.1)*1. )

for item in range(N_train_groups):
    print('Group[%d]'%group_index[item], 'accuracy: %.4f' % accuracy[item])

avg_accuracy = np.mean((np.abs(X - X_bits)<0.1)*1.)
print('avg acc:', avg_accuracy) 
print('Lovelive')










