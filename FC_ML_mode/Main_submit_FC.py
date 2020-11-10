from utils import *

from LS_CE import LS_Estimation,MMSE_Estimation, Interpolation_f

from Model_define_pytorch import FC_Estimation, FC_Detection, NMSELoss, DatasetFolder, DnCNN

from MLreceiver import MLReceiver, MakeCodebook

import torch.nn as nn
import random
import torch

Pilot_num = 8

# Parameters for training
gpu_list = '5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# 加载用于测试的接收数据 Y shape=[batch*2048]
if Pilot_num == 32:
    Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_1.csv', dtype=np.str,delimiter=',')

if Pilot_num == 8:
    Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_2.csv', dtype=np.str,delimiter=',')

Y = Y.astype(np.float32)
batch_num = Y.shape[0]

# 基于LS估计导频位置上的频域信道
Hf_partial = LS_Estimation(Y,Pilot_num)

# #### 通过部分频域信道估计全频域信道 ####
# Hf_hat = Interpolation_f(Hf_partial,Pilot_num)

#### 通过神经网络估计全频域信道 ####

# Model Construction
CE_model = FC_Estimation(2048, 4096, 4096, 2048)
CE_model = torch.nn.DataParallel(CE_model).cuda()  # model.module
# Load weights
CE_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_for_'+str(Pilot_num)+'.pth.tar'
CE_model.load_state_dict(torch.load(CE_model_path)['state_dict'])
print("CE Weight Loaded!")

# 网络的输入 shape=[-1*2*2*2*256]
# complex ---> real + imag
Hf_input = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_input[:,0,:,:,:] = Hf_partial.real
Hf_input[:,1,:,:,:] = Hf_partial.imag
CE_model.eval()
Hf_output = CE_model(torch.tensor(Hf_input).to('cuda'))

# Hf_output1 = Hf_output.cuda().data.cpu().numpy()
Hf_output1 = Hf_output.detach().cpu().numpy()
Hf_hat = Hf_output1[:,0,:,:,:]+1j*Hf_output1[:,1,:,:,:]

group_index = list(range(32))
group_num = 32
G = 256 // group_num
N_train_groups = len(group_index)

SD_model = []
for idx in range(N_train_groups):
    SD_model.append(FC_Detection(G*(4+8),  G*4, [512,1024,2048,2048,1024,512]))

for idx in range(N_train_groups):
    SD_model[idx] = torch.nn.DataParallel(SD_model[idx]).cuda() # model.module

for idx in range(N_train_groups):
    SD_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Detection_Pilot_'+str(Pilot_num)+'_'+str(group_num)+'Group'+str(group_index[idx])+'.pth.tar'
    SD_load_model = torch.load(SD_model_path)['state_dict']
    SD_model_dict = SD_model[idx].state_dict()
    SD_state_dict = {k:v for k,v in SD_load_model.items() if k in SD_model_dict.keys()}
    SD_model_dict.update(SD_state_dict)
    SD_model[idx].load_state_dict( SD_model_dict )

for idx in range(N_train_groups):
    SD_model[idx].eval()

Y_reshape = np.reshape(Y, (-1, 2,2,2, 256), order = 'F')
X_hat = np.zeros(( batch_num, 2, 512))
for item in range(N_train_groups):
    # 提取H的第index组
    Hf_output_item = Hf_output[:,:,:,:, G*group_index[item]: G*(group_index[item]+1)]
    Hf_output_item = Hf_output_item.reshape(-1, G*8)


    # 提取Y_data的第index组
    Y_data_item = np.reshape(Y_reshape[:,:,1,:,  G*group_index[item]: G*(group_index[item]+1)], (-1, G*4))
    Y_data_item = torch.Tensor(Y_data_item).to('cuda')

    SD_input = torch.cat((Y_data_item, Hf_output_item), 1)
    SD_output = SD_model[item](SD_input)


    X_hat_item = (SD_output > 0.5)*1.
    X_hat_item = np.array( X_hat_item.cpu() )
    X_hat_item = np.reshape( X_hat_item, (-1, 2, G*2))
    X_hat[:, :, 2*G*group_index[item]: 2*G*(group_index[item] + 1)] = X_hat_item

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








