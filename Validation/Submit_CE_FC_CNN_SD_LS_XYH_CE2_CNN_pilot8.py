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
from Model_define_pytorch import NMSELoss, FC_ELU_Estimation,CE_ResNet18,XYH2X_ResNet18,XDH2H_Resnet
from Densenet_SD import DenseNet100, DenseNet121,DenseNet201
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from MLreceiver_wo_print import *
from LSreveiver import *
from SoftMLreceiver import SoftMLReceiver
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


batch_size = 2000
epochs = 300


num_workers = 16
print_freq = 100
Pilot_num = 8


# 加载用于测试的接收数据 Y shape=[batch*2048]
# if Pilot_num == 32:
#     Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_1.csv', dtype=np.float32,delimiter=',')

# if Pilot_num == 8:
#     Y = np.loadtxt('/data/CuiMingyao/AI_competition/OFDMReceiver/Y_2.csv', dtype=np.float32,delimiter=',')
d = np.load('evaluation.npy', allow_pickle=True).item()
Y = d['y']

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
# FC = torch.nn.DataParallel( FC ).cuda()  # model.module
# CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module


SD = XYH2X_ResNet18()

SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XYH2X_Resnet18_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                
SD.load_state_dict(torch.load(SD_path)['state_dict'])
print("Weight For SD PD Loaded!")
# SD = torch.nn.DataParallel( SD ).cuda()  # model.module
# SD = SD.cuda()


CE2 = XDH2H_Resnet()

CE2_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XDH2H_'+str(args.model)+'_CE2_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                
CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
print("Weight For SD PD Loaded!")
# CE2 = torch.nn.DataParallel( CE2 ).cuda()  # model.module









Y = torch.tensor(Y)
X_bits = []
print('==========Begin Testing=========')
iter = 0



SD.eval()
FC.eval()
CNN.eval()
CE2.eval()
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
        # 第一层网络输出
        output1 = FC(input1)

        # 第二层网络输入预处理
        output1 = output1.reshape(Ns, 2, 4, 256)
        input2 = torch.cat([Yd_input_test, Yp_input_test, output1], 2)

        # 第二层网络的输出
        output2 = CNN(input2)

        #第三层网络输入预处理
        H_test_padding = output2.reshape(Ns, 2, 4, 32)
        H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
        H_test_padding = H_test_padding.permute(0,2,3,1)

        H_test_padding = torch.fft(H_test_padding, 1)/20
        H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256


        # ML 初步估计
        # X_LS, _ = get_ML(Yd_input_test, H_test_padding.detach().cpu().numpy())

        X_LS = get_LS(Yd_input_test, H_test_padding.detach().cpu())
        X_input_test = torch.zeros([Ns, 2, 2, 256], dtype = torch.float32)
        X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
        X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)


        
        input3 = torch.cat([X_input_test, Yp_input_test, Yd_input_test, H_test_padding], 2)

        # 第三层网络的输出
        output3 = SD(input3)

        # X_refine_test = ((output3 > 0.5)*2. - 1)/np.sqrt(2)
        
        output3 = output3.reshape([Ns,2,256,2])
        output3 = output3.permute(0,3,1,2).contiguous()

        # input4 = torch.cat([X_input_test, Yd_input_test, H_test_padding], 2)
        input4 = torch.cat([output3, Yd_input_test, H_test_padding], 2)

        output4 = CE2(input4)
        output4 = output4.reshape(Ns, 2, 4, 32)
        # 计算loss


        H_test_padding2 = output4
        H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
        H_test_padding2 = H_test_padding2.permute(0,2,3,1)

        H_test_padding2 = torch.fft(H_test_padding2, 1)/20
        H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256


        # ML性能对比
        Yd = np.array(Yd_input_test[:,0,:,:] + 1j*Yd_input_test[:,1,:,:])


        Hf2 = H_test_padding2.detach().cpu().numpy()
        Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:] 
        Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')



        X_SoftML, X_SoftMLbits = SoftMLReceiver(Yd, Hf2, SNRdb = 5)

        X_bits.append(X_SoftMLbits)

        print('Love live')

X_bits = np.concatenate(X_bits , axis = 0 )

if Pilot_num == 32:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_1_CE_FC_CNN_SD_XYH2X_Resnet18_CE2_' + str(args.model)+ '.bin')


if Pilot_num == 8:
    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_pre_2_CE_FC_CNN_SD_XYH2X_Resnet18_CE2_' + str(args.model)+ '.bin')
            
