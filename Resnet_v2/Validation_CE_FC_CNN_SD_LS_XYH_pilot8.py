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
    X_ML.real = (X_ML.real > 0)*2 - 1
    X_ML.imag = (X_ML.imag > 0)*2 - 1
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
lr_freq = 15
num_workers = 16
print_freq = 100
Pilot_num = 8
best_accuracy = 0.5


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



if args.model == 'Resnet18':
    SD = XYH2X_ResNet18()
elif args.model == 'Resnet34':
    SD = XYH2X_ResNet34()

if 'Resnet' in args.model:
    SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XYH2X_' + args.model + '_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'
else:
    SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XYH2X_Unet_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar' 
SD.load_state_dict(torch.load(SD_path)['state_dict'])
print("Weight For SD PD Loaded!")

criterion_CE =  NMSELoss()
criterion_SD =  torch.nn.BCELoss(weight=None, reduction='mean')




test_dataset  = RandomDataset(H_val,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

print('==========Begin Training=========')
iter = 0



SD.eval()
FC.eval()
CNN.eval()
with torch.no_grad():
    for Y_test, X_test, H_test in test_dataloader:
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

        nmse1 = criterion_CE(output1.detach().cpu(), Hf_test_label)
        print('The first layer nmse in frequency domain:',nmse1)

        input2 = torch.cat([Yd_input_test, Yp_input_test, output1], 2)

        # 第二层网络的输出
        output2 = CNN(input2)

        #第三层网络输入预处理
        H_test_padding = output2.reshape(Ns, 2, 4, 32)

        nmse2 = criterion_CE(H_test_padding.detach().cpu(), Ht_test_label)
        print('The second layer nmse in time domain:',nmse2)

        H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
        H_test_padding = H_test_padding.permute(0,2,3,1)

        H_test_padding = torch.fft(H_test_padding, 1)/20
        H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

        nmse3 = criterion_CE(H_test_padding.detach().cpu(), Hf_test_label)
        print('The second layer nmse in frequency domain:',nmse3)

        # ML 初步估计
        X_LS = get_LS(Yd_input_test, H_test_padding.detach().cpu())
        X_ML, X_bits = get_ML(Yd_input_test, H_test_padding.detach().cpu())

        eps = 0.1
        # ML性能
        error2 = np.abs(X_bits - X_test.numpy())
        accuracy2 = np.sum(error2 < eps)/error2.size

        print('ML accuracy %.4f' % accuracy2)


        # LS + CNN 性能
        X_input_test = torch.zeros([Ns, 2, 2, 256], dtype = torch.float32)
        X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
        X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)


        input3 = torch.cat([X_input_test, Yp_input_test, Yd_input_test, H_test_padding], 2)

        # 第三层网络的输出
        output3 = SD(input3)

        label = X_test.float().numpy()
        output3 = (output3 > 0.5)*1.
        output3 = output3.cpu().detach().numpy()


        error = np.abs(output3 - label)
        average_accuracy = np.sum(error < eps) / error.size

        print('LS+XYH2X accuracy %.4f' % average_accuracy)


        # ML+CNN 性能
        X_input_test = torch.zeros([Ns, 2, 2, 256], dtype = torch.float32)
        X_input_test[:,0,:,:] = torch.tensor(X_ML.real, dtype = torch.float32)
        X_input_test[:,1,:,:] = torch.tensor(X_ML.imag, dtype = torch.float32)


        input3 = torch.cat([X_input_test, Yp_input_test, Yd_input_test, H_test_padding], 2)

        # 第三层网络的输出
        output3 = SD(input3)

        label = X_test.float().numpy()
        output3 = (output3 > 0.5)*1.
        output3 = output3.cpu().detach().numpy()

        eps = 0.1
        error = np.abs(output3 - label)
        average_accuracy = np.sum(error < eps) / error.size

        print('ML+XYH2X accuracy %.4f' % average_accuracy)



print('Lovelive')




