from utils import *
import struct
import os
import torch.nn as nn
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat
import time
from Model_define_pytorch import NMSELoss, ResNet18, ResNet34,ResNet50,U_Net,FC_ELU_Estimation,CE_ResNet18
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from EMLreceiver import EMLReceiver, getERH
from MLreceiver import MLReceiver
from SoftMLreceiver import SoftMLReceiver
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'Resnet18')
parser.add_argument('--gpu_list',  type = str,  default='6', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
args = parser.parse_args()


# Parameters for training
SNRdb = args.SNR
mode = args.mode
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


epochs = 300

num_workers = 16
Pilot_num = 8
best_accuracy = 0.5

# 加载用于测试的接收数据 Y shape=[batch*2048]
if Pilot_num == 32:
    Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_1.csv', dtype=np.float32,delimiter=',')

if Pilot_num == 8:
    Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_2.csv', dtype=np.float32,delimiter=',')

# Model load for channel estimation
##### model construction for channel estimation #####

in_dim = 1024
h_dim = 4096
out_dim = 2*4*256
n_blocks = 2
act = 'ELU'
# Model Construction #input: batch*2*2*256 Received Yp # output: batch*2*2*32 time-domain channel
FC = FC_ELU_Estimation(in_dim, h_dim, out_dim, n_blocks)
CNN = CE_ResNet18()


path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/best_P' + str(Pilot_num) + '.pth'
state_dicts = torch.load(path)
FC.load_state_dict(state_dicts['fc'])
CNN.load_state_dict(state_dicts['cnn'])
print("Model for CE has been loaded!")

FC = torch.nn.DataParallel( FC ).cuda()  # model.module
CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module



print('==========Begin Training=========')
iter = 0

Y = torch.tensor(Y)

FC.eval()
CNN.eval()
with torch.no_grad():
    Y_test = Y
    Ns = Y_test.shape[0]

    # 接收数据与接收导频划分
    Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
    Y_input_test = Y_input_test.float()
    Yp_input_test = Y_input_test[:,:,0,:,:]
    Yd_input_test = Y_input_test[:,:,1,:,:] 


    # 第一层网络输入
    input1 = Yp_input_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
    input1 = input1.cuda()
    # 第一层网络输出
    output1 = FC(input1)


    # 第二层网络输入预处理
    output1 = output1.reshape(Ns, 2, 4, 256)

    input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)

    # 第二层网络的输出
    output2 = CNN(input2)

    #第三层网络输入预处理
    H_test_padding = output2.reshape(Ns, 2, 4, 32)


    H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True).cuda()],3)
    H_test_padding = H_test_padding.permute(0,2,3,1)

    H_test_padding = torch.fft(H_test_padding, 1)/20
    H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

    # ML性能对比
    Yd = np.array(Yd_input_test[:,0,:,:] + 1j*Yd_input_test[:,1,:,:])
    Hf = H_test_padding.detach().cpu().numpy()
    Hf = Hf[:,0,:,:] + 1j*Hf[:,1,:,:] 
    Hf = np.reshape(Hf, [-1,2,2,256], order = 'F')

    X_ML, X_bits = MLReceiver(Yd, Hf)


    X_EML, X_Ebits = SoftMLReceiver(Yd, Hf, SNRdb = 10)


    print('Love live')


if Pilot_num == 32:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_1_CE_FC_CNN_ML.bin')

    X_E1 = np.array(np.floor(X_Ebits + 0.5), dtype=np.bool)
    X_E1.tofile('./X_pre_1_CE_FC_CNN_EML.bin')


if Pilot_num == 8:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_2_CE_FC_CNN_ML.bin')
    
    X_E1 = np.array(np.floor(X_Ebits + 0.5), dtype=np.bool)
    X_E1.tofile('./X_pre_2_CE_FC_CNN_EML.bin')





