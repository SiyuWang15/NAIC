from utils import *

# from LS_CE import LS_Estimation,MMSE_Estimation, Interpolation_f

from Model_define_pytorch import FC_Estimation, FC_Detection, NMSELoss, DatasetFolder, DnCNN

from MLreceiver import MLReceiver, MakeCodebook

import torch.nn as nn
import random
import torch

Pilot_num = 8
mode = 0
# Parameters for training
gpu_list = '4,5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# 加载用于测试的接收数据 Y shape=[batch*2048]
if Pilot_num == 32:
    Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_1.csv', dtype=np.str,delimiter=',')

if Pilot_num == 8:
    Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_2.csv', dtype=np.str,delimiter=',')

Y = Y.astype(np.float32)
batch_num = Y.shape[0]


# #### 通过部分频域信道估计全频域信道 ####
in_dim = 1024
h_dim = 4096
out_dim = 256
n_blocks =  2
model = FC_Estimation(in_dim, h_dim, out_dim, n_blocks)
model = torch.nn.DataParallel(model).cuda() 


# Load weights
model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_Pilot'+str(Pilot_num)+'_mode'+str(mode)+'.pth.tar'
model.load_state_dict(torch.load(model_path)['state_dict'])
print("Weight Loaded!")

model.eval()

# 通过网络估计全频域信道
# complex ---> real + imag

Y_reshape = np.reshape(Y, [batch_num, 2, 2, 2, 256], order='F')
Yp = Y_reshape[:,:,0,:,:]
Yp = np.reshape(Yp, [batch_num, 2*2*256])
Yp = torch.Tensor(Yp).to('cuda')
# print(Yp.shape)

Ht_output = model(Yp)
Ht_output = Ht_output.detach().cpu().numpy()

Ht_reshape = Ht_output.reshape([-1,2,4,32], order='F')

Ht_complex = Ht_reshape[:,0,:,:] + 1j*Ht_reshape[:,1,:,:] 
Hf_complex = np.fft.fft(Ht_complex, 256)/20
Hf_reshape = np.reshape(Hf_complex, (-1,2,2,256), order='F')
Hf_output = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_output[:,0,:,:,:] = Hf_reshape.real
Hf_output[:,1,:,:,:] = Hf_reshape.imag

Hf_output1 = Hf_output
Hf_hat = Hf_output1[:,0,:,:,:]+1j*Hf_output1[:,1,:,:,:]

#### ML的输入 ####
Y = np.reshape(Y, (-1, 2,2,2,256), order = 'F')
Y = Y [:,0,:,:,:] + 1j*Y [:,1,:,:,:]
Y = Y[:,1,:,:]
# 生成码本
G = 4
codebook = MakeCodebook(G)

### 基于ML恢复发送比特流X ####
X_ML, X_bits = MLReceiver(Y,Hf_hat,codebook)

# X_bits = np.reshape(X_hat, (-1, 1024))

if Pilot_num == 32:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('/data/CuiMingyao/AI_competition/OFDMReceiver/X_pre_1.bin')

if Pilot_num == 8:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('/data/CuiMingyao/AI_competition/OFDMReceiver/X_pre_2.bin')








