import sys 
sys.path.append('../')
from utils import *
from model import *
import struct
import os
import torch.nn as nn
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat
import time
# from Model_define_pytorch import NMSELoss, FC_ELU_Estimation,CE_ResNet18,XYH2X_ResNet18,XDH2H_Resnet

import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import model.DeepRx_port as pD
import model.DeepRx_model

def run(prefix = './', gpu_list = '6,7', N = 3):

    SNRdb = -1
    mode = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


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
    Pilot_num = 8




    # 加载用于测试的接收数据 Y shape=[batch*2048]
    if Pilot_num == 32:
        Y = np.loadtxt(prefix + 'data/Y_1.csv', dtype=np.float32,delimiter=',')

    if Pilot_num == 8:
        Y = np.loadtxt(prefix + 'data/Y_2.csv', dtype=np.float32,delimiter=',')

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


    path = prefix + 'checkpoints/best_P8.pth'
    state_dicts = torch.load(path)
    FC.load_state_dict(state_dicts['fc'])
    CNN.load_state_dict(state_dicts['cnn'])
    print("Model for CE has been loaded!")



    SD_CE2_path = prefix + 'checkpoints/CE2_SD_0.14486.pth'

    SD = XYH2X_ResNet18()
    CE2 = XDH2H_Resnet()
    SD = torch.nn.DataParallel( SD ).cuda()  # model.module
    CE2 = torch.nn.DataParallel( CE2 ).cuda()
    state_dicts = torch.load(SD_CE2_path)
    SD.load_state_dict(state_dicts['SD'])
    CE2.load_state_dict(state_dicts['CE2'])

    print("Weight For SD CE2 Loaded!")


    FC = torch.nn.DataParallel( FC ).cuda()  # model.module
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module


    Y = torch.tensor(Y)
    X_bits = []
    print('==========Begin Testing=========')
    SD.eval()
    FC.eval()
    CNN.eval()
    CE2.eval()

    with torch.no_grad():

        # print('lr:%.4e' % optimizer_CE2.param_groups[0]['lr'])

        for i in range(2):
            Y_test = Y[i*5000 : (i+1)*5000, :]

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
#             print('第一层')

            # 第二层网络输入预处理
            output1 = output1.reshape(Ns, 2, 4, 256)
            # input2 = torch.cat([Yd_input_test, Yp_input_test, output1], 2)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)    
            # 第二层网络的输出
            output2 = CNN(input2)
            output2 = output2.reshape(Ns, 2, 4, 32)
#             print('第二层')

            start = output2
            for idx in range(N):
                #第三层网络输入预处理
                H_test_padding = start.cpu()
                H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)

                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256


                X_LS = get_LS(Yd_input_test, H_test_padding.detach().cpu())
                X_input_test = torch.zeros([Ns, 2, 2, 256], dtype = torch.float32)
                X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
                X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)


                
                input3 = torch.cat([X_input_test.cuda(), Yp_input_test.cuda(), Yd_input_test.cuda(), H_test_padding.cuda()], 2)

                # 第三层网络的输出
                output3 = SD(input3)
#                 print('第三层')
                
                X_1 = output3.reshape([Ns,2,256,2])
                X_1 = X_1.permute(0,3,1,2).contiguous()

                # 第四层网络的输入
                # input4 = torch.cat([X_input_test, Yd_input_test, H_test_padding], 2)
                # input4 = torch.cat([X_1, Yd_input_test, H_test_padding], 2)
                input4 = torch.cat([X_1, Yd_input_test.cuda(), H_test_padding.cuda()], 2)
                output4 = CE2(input4)
                output4 = output4.reshape(Ns, 2, 4, 32)
#                 print('第四层')

                #第五层网络输入预处理
                end = output4

                H_test_padding2 = end.cpu().detach()
                H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
                H_test_padding2 = H_test_padding2.permute(0,2,3,1)

                H_test_padding2 = torch.fft(H_test_padding2, 1)/20
                H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

                start = end
            
            eps = 0.1
            # ML性能对比
            Yd = np.array(Yd_input_test[:,0,:,:] + 1j*Yd_input_test[:,1,:,:])

            Hf2 = H_test_padding2.numpy()
            Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:] 
            Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')

            X_SoftML, X_SoftMLbits, _ = SoftMLReceiver(Yd, Hf2, SNRdb = 5)


            X_bits.append(X_SoftMLbits)

            print('Love live')
        
    X_bits = np.concatenate(X_bits , axis = 0 )

    if Pilot_num == 32:
        X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
        X_1.tofile(prefix + '/results/X_pre_1_CE_FC_CNN_SD_LS_XYH_CE2_Resnet34_NSDCE_N_'+str(N)+ '.bin')


    if Pilot_num == 8:
        X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
        X_1.tofile(prefix + '/results/X_pre_2_CE_FC_CNN_SD_LS_XYH_CE2_Resnet34_NSDCE_N_'+str(N)+ '.bin')


if __name__ == '__main__':
    run(prefix = '../')
            
