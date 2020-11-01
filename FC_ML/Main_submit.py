from utils import *

from LS_CE import LS_Estimation,MMSE_Estimation, Interpolation_f

from Model_define_pytorch import FC_Estimation, NMSELoss, DatasetFolder, DnCNN

from MLreceiver import MLReceiver, MakeCodebook

import torch.nn as nn
import random
import torch

Pilot_num = 32

# Parameters for training
gpu_list = '4,5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# 加载用于测试的接收数据 Y shape=[batch*2048]
if Pilot_num == 32:
    Y = np.loadtxt('Y_1.csv', dtype=np.str,delimiter=',')

if Pilot_num == 8:
    Y = np.loadtxt('Y_2.csv', dtype=np.str,delimiter=',')

Y = Y.astype(np.float32)
batch_num = Y.shape[0]

# 基于LS估计导频位置上的频域信道
Hf_partial = LS_Estimation(Y,Pilot_num)

# #### 通过部分频域信道估计全频域信道 ####
# Hf_hat = Interpolation_f(Hf_partial,Pilot_num)

#### 通过神经网络估计全频域信道 ####

# Model Construction
model = FC_Estimation(2048, 4096, 4096, 2048)
model = torch.nn.DataParallel(model).cuda()  # model.module
# Load weights
model_path = './Modelsave/FC_Estimation_for_'+str(Pilot_num)+'.pth.tar'
model.load_state_dict(torch.load(model_path)['state_dict'])
print("Weight Loaded!")

# 网络的输入 shape=[-1*2*2*2*256]
# complex ---> real + imag
Hf_input = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_input[:,0,:,:,:] = Hf_partial.real
Hf_input[:,1,:,:,:] = Hf_partial.imag
model.eval()
Hf_output = model(torch.tensor(Hf_input).to('cuda'))

# Hf_output1 = Hf_output.cuda().data.cpu().numpy()
Hf_output1 = Hf_output.detach().cpu().numpy()
Hf_hat = Hf_output1[:,0,:,:,:]+1j*Hf_output1[:,1,:,:,:]

#### ML的输入 ####
Y = np.reshape(Y, (-1, 2,2,2,256), order = 'F')
Y = Y [:,0,:,:,:] + 1j*Y [:,1,:,:,:]
Y = Y[:,1,:,:]
# 生成码本
G = 4
codebook = MakeCodebook(G)

#### 基于ML恢复发送比特流X ####
X_ML, X_bits = MLReceiver(Y,Hf_hat,codebook)

if Pilot_num == 32:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_1.bin')

if Pilot_num == 8:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_2.bin')








