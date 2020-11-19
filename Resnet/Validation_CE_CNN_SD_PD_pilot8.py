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
from Model_define_pytorch_CE_CNN import CE_ResNet18
from Model_define_pytorch import NMSELoss, ResNet18, ResNet34,ResNet50,U_Net
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'Resnet18')
parser.add_argument('--gpu_list',  type = str,  default='4,5', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
args = parser.parse_args()


# Parameters for training
SNRdb = args.SNR
mode = args.mode
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list



batch_num = 2000
epochs = 300
num_workers = 16
print_freq = 100
Pilot_num = 8


# channel data for training and validation

data2 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]   # time-domain channel for training


# Model load for channel estimation
##### model construction for channel estimation #####


CE = CE_ResNet18()
CE = torch.nn.DataParallel( CE ).cuda()  # model.module
CE_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/Resnet18_DirectCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
CE.load_state_dict(torch.load( CE_path )['state_dict'])
CE.eval()
print("Model for CE has been loaded!")


if args.model == 'Resnet18':
    SD = ResNet18()
elif args.model == 'Resnet34':
    SD = ResNet34()
elif args.model == 'Resnet50':
    SD = ResNet50()
else:
    SD = U_Net()
SD = torch.nn.DataParallel( SD ).cuda()  # model.module

if 'Resnet' in args.model:
    SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_' + args.model + '_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'
else:
    SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_Unet_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar' 
SD.load_state_dict(torch.load(SD_path)['state_dict'])
SD.eval()
print("Weight For SD PD Loaded!")



criterion_CE =  NMSELoss()
 # 生成测试数据 Y：-1*2048； X：-1*1024； H：-1*4*32 时域信道
Y_test, X_test, H_test = generatorXY(batch_num,H_val,Pilot_num, SNR=SNRdb, m= mode)
Y_test = torch.tensor(Y_test)
X_test = torch.tensor(X_test)
H_test = torch.tensor(H_test)


Ns = Y_test.shape[0]

# 真实的频域信道，获取标签
Hf_test = np.fft.fft(np.array(H_test), 256)/20 # 4*256
Hf_test_label = torch.zeros([Ns, 2, 4, 256], dtype=torch.float32)
Hf_test_label[:, 0, :, :] = torch.tensor(Hf_test.real, dtype = torch.float32)
Hf_test_label[:, 1, :, :] = torch.tensor(Hf_test.imag, dtype = torch.float32)

# 第一层网络输入
Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
Y_input_test = Y_input_test.float()
Yp_input_test = Y_input_test[:,:,0,:,:]
Yd_input_test = Y_input_test[:,:,1,:,:] 

net_input = torch.cat([Yp_input_test, Yd_input_test], 2)
net_input = torch.reshape(net_input, [Ns, 1, 8, 256])

with torch.no_grad():
    net_input = net_input.cuda()
    H_test_refine = CE(net_input)
    #第二级网络输入
    H_test_padding = H_test_refine.reshape(Ns, 2, 4, 32)

    # nmse1 = criterion_CE(H_test_refine, H_test.cuda())
    # print('NMSE time domain:', nmse1.detech().cpu().numpy())

    H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True).cuda()],3)
    H_test_padding = H_test_padding.permute(0,2,3,1)

    H_test_padding = torch.fft(H_test_padding, 1)/20
    H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

    nmse2 = criterion_CE(H_test_padding, Hf_test_label.cuda())
    print('NMSE frequency domain:', nmse2.cpu().numpy())
    net2_input = torch.cat([Yp_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)

    net2_input = torch.reshape(net2_input, [Ns, 1, 16, 256])

    output = SD(net2_input)

    label = X_test.float().numpy()
    output = (output > 0.5)*1.
    output = output.cpu().detach().numpy()

    eps = 0.1
    error = np.abs(output - label)
    average_accuracy = np.sum(error < eps) / error.size

    print('accuracy %.4f' % average_accuracy)

                
