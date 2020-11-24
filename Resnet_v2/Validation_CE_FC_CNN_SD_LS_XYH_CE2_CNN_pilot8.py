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

lr_threshold = 1e-6
lr_freq = 10
num_workers = 16
print_freq = 100
Pilot_num = 8
best_nmse = 1



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
# FC = torch.nn.DataParallel( FC ).cuda()  # model.module
# CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module


SD = XYH2X_ResNet18()

SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XYH2X_Resnet18_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                
SD.load_state_dict(torch.load(SD_path)['state_dict'])
print("Weight For SD PD Loaded!")
# SD = torch.nn.DataParallel( SD ).cuda()  # model.module
# SD = SD.cuda()


CE2 = XDH2H_Resnet()

CE2_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XDH2H_'+str(args.model)+'_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                
CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
print("Weight For SD PD Loaded!")
# CE2 = torch.nn.DataParallel( CE2 ).cuda()  # model.module






criterion_CE =  NMSELoss()
criterion_SD =  torch.nn.BCELoss(weight=None, reduction='mean')

Y_test, X_test, H_test = generatorXY(batch_size, H_val,Pilot_num, SNR=SNRdb, m= mode)
Y_test = torch.tensor(Y_test)
X_test = torch.tensor(X_test)
H_test = torch.tensor(H_test)
print('==========Begin Testing=========')
iter = 0



SD.eval()
FC.eval()
CNN.eval()
CE2.eval()
with torch.no_grad():

    # print('lr:%.4e' % optimizer_CE2.param_groups[0]['lr'])


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
    # X_LS, X_bits = get_ML(Yd_input_test, H_test_padding.detach().cpu().numpy())

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
    nmse1 = criterion_CE(output2.reshape(Ns, 2, 4, 32).detach().cpu(), Ht_test_label)
    nmse1 = nmse1.numpy()
    nmse2 = criterion_CE(output4.detach().cpu(), Ht_test_label)
    nmse2 = nmse2.numpy()


    # print('ML Accuracy:%.4f' % average_accuracy_ML, 'SD Accuracy:%.4f' % average_accuracy_SD)
    print('CE1 NMSE:%.4f' % nmse1)
    print('CE2 NMSE:%.4f' % nmse2)

    H_test_padding2 = output4
    H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
    H_test_padding2 = H_test_padding2.permute(0,2,3,1)

    H_test_padding2 = torch.fft(H_test_padding2, 1)/20
    H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256


    # ML性能对比
    Yd = np.array(Yd_input_test[:,0,:,:] + 1j*Yd_input_test[:,1,:,:])

    Hf1 = H_test_padding.detach().cpu().numpy()
    Hf1 = Hf1[:,0,:,:] + 1j*Hf1[:,1,:,:] 
    Hf1 = np.reshape(Hf1, [-1,2,2,256], order = 'F')


    Hf2 = H_test_padding2.detach().cpu().numpy()
    Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:] 
    Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')

    eps = 0.1

    X_SoftML, X_SoftMLbits = SoftMLReceiver(Yd, Hf1, SNRdb = 5)

    error_SoftML = np.abs(X_SoftMLbits - X_test.numpy())
    accuracy_SoftML = np.sum(error_SoftML < eps)/error_SoftML.size
    print('CE1:SNR=',5,'SoftML accuracy %.4f' % accuracy_SoftML)


    X_SoftML, X_SoftMLbits = SoftMLReceiver(Yd, Hf2, SNRdb = 5)

    error_SoftML = np.abs(X_SoftMLbits - X_test.numpy())
    accuracy_SoftML = np.sum(error_SoftML < eps)/error_SoftML.size
    print('CE2:SNR=',5,'SoftML accuracy %.4f' % accuracy_SoftML)
        
