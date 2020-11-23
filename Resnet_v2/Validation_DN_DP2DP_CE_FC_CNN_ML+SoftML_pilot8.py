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
from Model_define_pytorch import NMSELoss, ResNet18, ResNet34,ResNet50,U_Net,FC_ELU_Estimation,CE_ResNet18,DP2DP_U_Net
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from EMLreceiver import EMLReceiver, getERH
from MLreceiver import MLReceiver
from SoftMLreceiver import SoftMLReceiver
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_list',  type = str,  default='6', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
args = parser.parse_args()


# Parameters for training
SNRdb = args.SNR
mode = args.mode
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list



batch_size = 2000
epochs = 300
eps = 0.1
num_workers = 16
Pilot_num = 8
best_accuracy = 0.5

# channel data for validation
data2 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]   # time-domain channel for training




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

# Model load for denoise
DN = DP2DP_U_Net()
path_DN = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/DP2DP_Unet_Filter_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'
DN.load_state_dict(torch.load(path_DN)['state_dict'])

# FC = torch.nn.DataParallel( FC ).cuda()  # model.module
# CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
# DN = torch.nn.DataParallel( DN ).cuda()  # model.module

criterion_DN = NMSELoss()
criterion_CE =  NMSELoss()
criterion_SD =  torch.nn.BCELoss(weight=None, reduction='mean')



print('==========Begin Training=========')
iter = 0

Y_test, X_test, H_test,Y_test_woN = generatorXY_DN(batch_size, H_val,Pilot_num, SNR=SNRdb, m= mode)
Y_test = torch.tensor(Y_test)
X_test = torch.tensor(X_test)
H_test = torch.tensor(H_test)
Y_test_woN = torch.tensor(Y_test_woN)

FC.eval()
CNN.eval()
DN.eval()
with torch.no_grad():
    Ns = Y_test.shape[0]
    # 真实的频域信道，获取标签
    Hf_test = np.fft.fft(np.array(H_test), 256)/20 # 4*256
    Hf_test_label = torch.zeros([Ns, 2, 4, 256], dtype=torch.float32)
    Hf_test_label[:, 0, :, :] = torch.tensor(Hf_test.real, dtype = torch.float32)
    Hf_test_label[:, 1, :, :] = torch.tensor(Hf_test.imag, dtype = torch.float32)
    # 真实的时域信道，获取标签
    Ht_test_label = torch.zeros([Ns, 2, 4, 32], dtype=torch.float32)
    Ht_test_label[:, 0, :, :] = H_test.real.float()
    Ht_test_label[:, 1, :, :] = H_test.imag.float()



    # 接收数据与接收导频划分
    Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
    Y_input_test = Y_input_test.float()
    Yp_input_test = Y_input_test[:,:,0,:,:]
    Yd_input_test = Y_input_test[:,:,1,:,:] 

    Y_input_test_woN = np.reshape(Y_test_woN, [Ns, 2, 2, 2, 256], order='F')
    Y_input_test_woN = Y_input_test_woN.float()
    Yp_input_test_woN = Y_input_test_woN[:,:,0,:,:]
    Yd_input_test_woN = Y_input_test_woN[:,:,1,:,:] 

    # 第一层网络输入
    input1 = Yp_input_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
    # 第一层网络输出
    output1 = FC(input1)


    # 第二层网络输入预处理
    output1 = output1.reshape(Ns, 2, 4, 256)

    nmse1 = criterion_CE(output1.detach(), Hf_test_label)
    print('The first layer nmse in frequency domain:',nmse1)

    input2 = torch.cat([Yd_input_test, Yp_input_test, output1], 2)

    # 第二层网络的输出
    output2 = CNN(input2)
    #获取信道
    H_test_padding = output2.reshape(Ns, 2, 4, 32)

    nmse2 = criterion_CE(H_test_padding.detach(), Ht_test_label)
    print('The second layer nmse in time domain:',nmse2)

    H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
    H_test_padding = H_test_padding.permute(0,2,3,1)

    H_test_padding = torch.fft(H_test_padding, 1)/20
    H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256


    #降噪  
    input3 = torch.cat([Yp_input_test, Yd_input_test], 2)
    output = DN(input3)

    '''
    降噪前
    '''
    # ML性能对比
    Yd = np.array(Yd_input_test[:,0,:,:] + 1j*Yd_input_test[:,1,:,:])
    Hf = H_test_padding.detach().numpy()
    Hf = Hf[:,0,:,:] + 1j*Hf[:,1,:,:] 
    Hf = np.reshape(Hf, [-1,2,2,256], order = 'F')

    X_ML, X_bits = MLReceiver(Yd, Hf)

    error_ML = np.abs(X_bits - X_test.numpy())
    accuracy_ML = np.sum(error_ML < eps)/error_ML.size
    print('Before Filter：ML accuracy %.4f' % accuracy_ML)

    # SoftML性能
    hf_hat = Hf
    hf_true = np.reshape(Hf_test, [-1,2,2,256], order = 'F')
    X_SoftML, X_Softbits = SoftMLReceiver(Yd, Hf, SNRdb = 5) 

    error_SoftML = np.abs(X_Softbits - X_test.numpy())
    accuracy_SoftML = np.sum(error_SoftML < eps)/error_SoftML.size
    print('Before Filter：SoftML accuracy %.4f' % accuracy_SoftML)

    '''
    降噪后
    '''
    Yd_DN = output[:,:,2:,:].detach()
    Yd_DN = np.array(Yd_DN[:,0,:,:] + 1j*Yd_DN[:,1,:,:])

    Yd_woN = np.array(Yd_input_test_woN[:,0,:,:] + 1j*Yd_input_test_woN[:,1,:,:])
    print(criterion_DN(input3[:,:,2:,:].cpu(), Yd_input_test_woN))
    print(criterion_DN(output[:,:,2:,:], Yd_input_test_woN))
    # ML性能对比
    X_ML, X_bits = MLReceiver(Yd_DN, Hf)

    error_ML = np.abs(X_bits - X_test.numpy())
    accuracy_ML = np.sum(error_ML < eps)/error_ML.size
    print('After Filter：ML accuracy %.4f' % accuracy_ML)

    # SoftML性能
    hf_hat = Hf
    hf_true = np.reshape(Hf_test, [-1,2,2,256], order = 'F')
    X_SoftML, X_Softbits = SoftMLReceiver(Yd_DN, Hf, SNRdb = 5) 

    error_SoftML = np.abs(X_Softbits - X_test.numpy())
    accuracy_SoftML = np.sum(error_SoftML < eps)/error_SoftML.size
    print('After Filter：SoftML accuracy %.4f' % accuracy_SoftML)



    print('Love live')







