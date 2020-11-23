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
from Model_define_pytorch import NMSELoss, FC_ELU_Estimation,CE_ResNet18,XYH2X_ResNet18, XYH2X_ResNet34
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from LSreveiver import *
from MLreceiver import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'Resnet34')
parser.add_argument('--gpu_list',  type = str,  default='6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
args = parser.parse_args()


# Parameters for training
SNRdb = args.SNR
mode = args.mode
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


def get_ML(Yd_input, Hf):
    Yd = np.array(Yd_input[:,0,:,:] + 1j*Yd_input[:,1,:,:])
    Hf = Hf[:,0,:,:] + 1j*Hf[:,1,:,:] 
    Hf = np.reshape(Hf, [-1,2,2,256], order = 'F')
    X_ML, X_bits = MLReceiver(Yd, Hf)
    return X_ML, X_bits

def get_LS(Yd_input, Hf):
    Yd = np.array(Yd_input[:,0,:,:] + 1j*Yd_input[:,1,:,:])
    Hf = np.array(Hf[:,0,:,:] + 1j*Hf[:,1,:,:]) 
    Hf = np.reshape(Hf, [-1,2,2,256], order = 'F')
    X_LS = LSequalization(Hf, Yd)
    X_LS.real = (X_LS.real > 0)*2 - 1
    X_LS.imag = (X_LS.imag > 0)*2 - 1
    return X_LS

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


if args.model == 'Resnet18':
    SD = XYH2X_ResNet18()
elif args.model == 'Resnet34':
    SD = XYH2X_ResNet34()
SD = torch.nn.DataParallel( SD ).cuda()  # model.module

if 'Resnet' in args.model:
    SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XYH2X_' + args.model + '_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'
else:
    SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XYH2X_Unet_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar' 
SD.load_state_dict(torch.load(SD_path)['state_dict'])
print("Weight For SD PD Loaded!")


print('==========Begin Training=========')
iter = 0
Y = torch.tensor(Y)
X_bits = []

SD.eval()
FC.eval()
CNN.eval()
with torch.no_grad():
    for i in range(5):
        Y_test = Y[i*2000 : (i+1)*2000, :]
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

        # ML 初步估计
        X_LS, _ = get_ML(Yd_input_test, H_test_padding.detach().cpu())
        X_input_test = torch.zeros([Ns, 2, 2, 256], dtype = torch.float32)
        X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
        X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)


        input3 = torch.cat([X_input_test.cuda(), Yp_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)

        # 第三层网络的输出
        output3 = SD(input3)
        output3 = (output3 > 0.5)*1.
        output3 = output3.cpu().detach().numpy()
        X_bits.append(output3)

        print('Love live') 

X_bits = np.concatenate(X_bits , axis = 0 )

if Pilot_num == 32:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_1_CE_FC_CNN_SD_XYH2X_' + str(args.model) +  '.bin')


if Pilot_num == 8:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_2_CE_FC_CNN_SD_XYH2X_' + str(args.model) + '.bin')






