from utils import *
import struct
import torch.nn as nn
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat

from Model_define_pytorch import FC_Estimation, FC_Detection, NMSELoss, DatasetFolder
from LS_CE import LS_Estimation,MMSE_Estimation, Interpolation_f

from MLreceiver import MLReceiver, MakeCodebook
from SoftMLreceiver import SoftMLReceiver
from generate_data import generatorXY

# Parameters for training
gpu_list = '0,1,2,3'
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

# 生成测试数据 Y：-1*2048； X：-1*1024； H：-1*4*32 时域信道
Y, X, H = generatorXY(batch_num,H_val,Pilotnum)


# 完美的频域信道
Hf = np.fft.fft(H, 256)/20
Hf = np.reshape(Hf, (-1, 2,2,256), order='F')

# 基于LS估计导频位置上的频域信道
Hf_partial = LS_Estimation(Y,Pilotnum)
# Hf_partial = MMSE_Estimation(Y,Pilotnum)


#### 通过部分频域信道估计全频域信道 ####

#### The first method
# Model Construction
in_dim = 2048
h_dim = 4096
out_dim = 2048
n_blocks =  2
CE_model = FC_Estimation(in_dim, h_dim, out_dim, n_blocks)
CE_model = torch.nn.DataParallel(CE_model).cuda()  # model.module
# Load weights
CE_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation'+ '_f2f_' +'Pilot'+str(Pilotnum)+'_'+ str(in_dim) +'_'+ str(h_dim) +'_'+ str(out_dim) +'_'+ str(n_blocks) + '.pth.tar'
CE_model.load_state_dict(torch.load(CE_model_path)['state_dict'])
print("CE Weight Loaded!")
# 通过网络估计全频域信道
# complex ---> real + imag
Hf_input = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_input[:,0,:,:,:] = Hf_partial.real
Hf_input[:,1,:,:,:] = Hf_partial.imag
CE_model.eval()
Hf_output = CE_model(torch.tensor(Hf_input).to('cuda'))
criterion = NMSELoss(reduction='mean')

Hf_label = np.zeros(shape=[batch_num,2,2,2,256],dtype=np.float32)
Hf_label[:,0,:,:,:] = Hf.real
Hf_label[:,1,:,:,:] = Hf.imag
NMSE_loss = criterion(Hf_output.cuda(), torch.tensor(Hf_label).cuda())

# Hf_output1 = Hf_output.cuda().data.cpu().numpy()
Hf_output1 = Hf_output.detach().cpu().numpy()
Hf_hat = Hf_output1[:,0,:,:,:]+1j*Hf_output1[:,1,:,:,:]


# #### The second method
# Hf_hat1 = Interpolation_f(Hf,Pilotnum)
# Hf_hat2 = Interpolation_f(Hf_partial2,Pilotnum)

NMSE1 = np.sum(abs(Hf_hat-Hf)**2)/np.sum(abs(Hf)**2)
print('NMSE:', NMSE1)
# NMSE2 = np.sum(abs(Hf_hat2-Hf)**2)/np.sum(abs(Hf)**2)
# print(NMSE2)

#### 信号检测 ####
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
    SD_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Detection_Pilot_'+str(Pilotnum)+'_'+str(group_num)+'Group'+str(group_index[idx])+'.pth.tar'
    SD_load_model = torch.load(SD_model_path)['state_dict']
    SD_model_dict = SD_model[idx].state_dict()
    SD_state_dict = {k:v for k,v in SD_load_model.items() if k in SD_model_dict.keys()}
    SD_model_dict.update(SD_state_dict)
    SD_model[idx].load_state_dict( SD_model_dict )

for idx in range(N_train_groups):
    SD_model[idx].eval()

# 输入网络
X_reshape = np.reshape(X, (-1, 2, 512))
Y_reshape = np.reshape(Y, (-1, 2,2,2, 256), order = 'F')

accuracy = np.zeros((N_train_groups, 1))
X_hat = np.zeros(( batch_num, 2, 512))

for item in range(N_train_groups):
    # 提取X的第index组
    X_temp = X_reshape[:,:,2*G*group_index[item]: 2*G*(group_index[item] + 1)]
    X_item = np.reshape(X_temp, (-1, G*4))

    # 提取H的第index组
    Hf_output_item = Hf_output[:,:,:,:, G*group_index[item]: G*(group_index[item]+1)]
    Hf_output_item = Hf_output_item.reshape(-1, G*8)


    # 提取Y_data的第index组
    Y_data_item = np.reshape(Y_reshape[:,:,1,:,  G*group_index[item]: G*(group_index[item]+1)], (-1, G*4))
    Y_data_item = torch.Tensor(Y_data_item).to('cuda')

    SD_input = torch.cat((Y_data_item, Hf_output_item), 1)
    SD_output = SD_model[item](SD_input)

    SD_label = torch.Tensor(X_item)

    X_hat_item = (SD_output > 0.5)*1. 
    error = X_hat_item.cpu() - SD_label 
    accuracy[item] = (np.array((torch.sum(torch.abs(error)<0.1)*1.)/(error.numel()*1.)))

    X_hat_item = np.array( X_hat_item.cpu() )
    X_hat_item = np.reshape(X_hat_item, (-1, 2, G*2))
    X_hat[:, :, 2*G*group_index[item]: 2*G*(group_index[item] + 1)] = X_hat_item


for item in range(N_train_groups):
    print('Group[%d]'%group_index[item], 'accuracy: %.4f' % accuracy[item])

X_hat = np.reshape(X_hat , (-1, 1024))
avg_accuracy = np.mean((np.abs(X - X_hat)<0.1)*1.)
print('avg acc:', avg_accuracy) 
print('Lovelive')









