# from utils import *
import struct
import torch
import numpy as np
from Model_define_pytorch_pilot32_1 import yp2h_estimation, CNN_Estimation
from Model_define_pytorch_CE_CNN_pilot8_1 import FC_ELU_Estimation, CE_ResNet18, CE_ResNet34, CE_ResNet50
import argparse


def channel_pre_32(Y, mode, Pilot_num):
    this_type_str = type(Y)
    if this_type_str is not np.ndarray:
        Y = Y.detach().cpu().numpy()
    batch_num = Y.shape[0]
    Ns = batch_num
    ##### Obtain input data #####
    Y = np.reshape(Y, [-1, 2, 2, 2, 256], order='F').astype(np.float32)
    Yp = Y[:,:,0,:,:]   # 取出接收导频信号，实部虚部*两根接收天线*256子载波
    Yd = Y[:,:,1,:,:]   # 取出接收导频信号，实部虚部*两根接收天线*256子载波
    
    ##### 根据不同mode估计时域信道 #####
    in_dim = 1024
    h_dim = 4096
    out_dim = 2048
    n_blocks = 2
    
    # Model Construction #input: batch*2*2*256 Received Yp # output: batch*2*2*256 time-domain channel
    FC = yp2h_estimation(in_dim, h_dim, out_dim, n_blocks)
    
    FC = torch.nn.DataParallel(FC).cuda()  # model.module
    # Load weights
    FC_path = '/data/HaoJiang_data/AI second part/Modelsave/FC_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    FC.load_state_dict(torch.load(FC_path)['state_dict'])
    print("Model for FC CE has been loaded!")
    # Estimate the time-domain channel
    FC.eval()
    Hf_output1 = FC(torch.tensor(Yp.reshape(batch_num, 2*2*256)).to('cuda'))
    
    CNN = CNN_Estimation()
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
    CNN_path = '/data/HaoJiang_data/AI second part/Modelsave/CNN_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CNN.load_state_dict(torch.load( CNN_path )['state_dict'])
    print("Model for CNN CE has been loaded!")
    
    CNN.eval()
    
    Hf_output1 = Hf_output1.reshape(batch_num, 2, 4, 256)
    input = torch.cat([torch.tensor(Yd).cuda(), Hf_output1], 2)
    Ht_output = CNN(input)
    Ht_output = Ht_output.reshape(batch_num, 2, 4, 32)
    
    Ht_output1 = Ht_output.cuda().data.cpu().numpy()
    Ht_hat = Ht_output1[:,0,:,:]+1j*Ht_output1[:,1,:,:]
    
    Hf_hat = np.fft.fft(Ht_hat, 256) / 20  # frequency-domain channel
    Hf_hat = np.reshape(Hf_hat, [Ns, 2, 2, 256], order='F')
    
    Hfr_hat = np.zeros([Ns, 2, 2, 2, 256])
    Hfr_hat[:,0,:,:,:] = Hf_hat.real
    Hfr_hat[:,1,:,:,:] = Hf_hat.imag
    # Hfr_hat = torch.from_numpy(Hfr_hat).float().cuda()
    
    # Hf = np.fft.fft(H, 256) / 20  # frequency-domain channel
    # Hf = np.reshape(Hf, [Ns, 2, 2, 256], order='F')
    
    
    # NMSE = np.mean(abs(Hf_hat-Hf)**2)/np.mean(abs(Hf)**2)
    # print('NMSE: ' +str(NMSE))
    return Hfr_hat

def channel_pre_8(Y, mode, Pilot_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'Resnet18')
    parser.add_argument('--load_FC',  type = int,  default= 1)
    parser.add_argument('--load_CNN',  type = int,  default= 1)
    parser.add_argument('--freeze_CE',  type = int,  default= 1)
    parser.add_argument('--load_CE',  type = int,  default= 1)
    args = parser.parse_args()
    
    in_dim = 1024
    h_dim = 4096
    out_dim = 2*4*256
    n_blocks = 2
    
    FC = FC_ELU_Estimation(in_dim, h_dim, out_dim, n_blocks)
    CNN = CE_ResNet18()
    if args.load_CE:
        path = '/data/HaoJiang_data/AI second part/Modelsave/best_P' + str(Pilot_num) + '.pth'
        state_dicts = torch.load(path)
        FC.load_state_dict(state_dicts['fc'])
        CNN.load_state_dict(state_dicts['cnn'])
        print("Model for CE has been loaded!")
    FC = torch.nn.DataParallel( FC ).cuda()  # model.module
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
    if args.freeze_CE == 1:
        for params in FC.parameters():
            FC.requires_grad = False
        for params in CNN.parameters():
            CNN.requires_grad = False
        print('freeze CE channel estimation!')
        FC.eval()
        CNN.eval()
    this_type_str = type(Y)
    if this_type_str is not np.ndarray:
        Y = Y.detach().cpu().numpy()
    batch_size = Y.shape[0]
    Ns = batch_size
    # 第一层网络输入
    Y_input_train = np.reshape(Y, [batch_size, 2, 2, 2, 256], order='F')
    Yp_input_train = Y_input_train[:,:,0,:,:]
    Yd_input_train = Y_input_train[:,:,1,:,:] 
    Yp_input_train = torch.from_numpy(Yp_input_train).float().cuda()
    Yd_input_train = torch.from_numpy(Yd_input_train).float().cuda()
    # 第一层网络输入
    input1 = Yp_input_train.reshape(batch_size, 2*2*256) # 取出接收导频信号，实部虚部*2*256
    # 第一层网络输出
    output1 = FC(input1)

    # 第二层网络输入预处理
    output1 = output1.reshape(batch_size, 2, 4, 256)
    input2 = torch.cat([Yd_input_train, Yp_input_train, output1], 2)
    
    # 第二层网络的输出
    output2 = CNN(input2)
    print(output2.shape)
    Ht_train_refine = output2.reshape(batch_size, 2, 4, 32)
    
    Ht_train_refine = Ht_train_refine.cuda().data.cpu().numpy()
    Ht_hat = Ht_train_refine[:,0,:,:]+1j*Ht_train_refine[:,1,:,:]
    
    Hf_hat = np.fft.fft(np.array(Ht_hat), 256)/20 # 4*256
    Hf_train_hat = np.zeros([batch_size, 2, 4, 256])
    Hf_train_hat[:, 0, :, :] = Hf_hat.real
    Hf_train_hat[:, 1, :, :] = Hf_hat.imag
    
    Hf_train_hat = Hf_train_hat.reshape(batch_size, 2, 2, 2, 256)
    return Hf_train_hat
    

def channel_pre_8_0(Y, mode, Pilot_num):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'Resnet18')
    parser.add_argument('--load_FC',  type = int,  default= 1)
    parser.add_argument('--load_CNN',  type = int,  default= 1)
    args = parser.parse_args()
    
    if args.model == 'Resnet18':
        CNN = CE_ResNet18()
    elif args.model == 'Resnet34':
        CNN = CE_ResNet34()
    elif args.model == 'Resnet50':
        CNN = CE_ResNet50()
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
    if args.load_CNN:
        CNN_path = '/data/HaoJiang_data/AI second part/Modelsave/'+ str(args.model) +'_DirectCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
        CNN.load_state_dict(torch.load( CNN_path )['state_dict'])
        print("Model for CNN CE has been loaded!")
    
    this_type_str = type(Y)
    if this_type_str is not np.ndarray:
        Y = Y.detach().cpu().numpy()
    batch_size = Y.shape[0]
    Ns = batch_size
    # 第一层网络输入
    Y_input_train = np.reshape(Y, [batch_size, 2, 2, 2, 256], order='F')
    Yp_input_train = Y_input_train[:,:,0,:,:]
    Yd_input_train = Y_input_train[:,:,1,:,:] 
    Yp_input_train = torch.from_numpy(Yp_input_train).float().cuda()
    Yd_input_train = torch.from_numpy(Yd_input_train).float().cuda()

    net_input = torch.cat([Yp_input_train, Yd_input_train], 2)
    net_input = torch.reshape(net_input, [batch_size, 1, 8, 256])
    
    Ht_train_refine = CNN(net_input)
    Ht_train_refine = Ht_train_refine.reshape(batch_size, 2, 4, 32)
    
    Ht_train_refine = Ht_train_refine.cuda().data.cpu().numpy()
    Ht_hat = Ht_train_refine[:,0,:,:]+1j*Ht_train_refine[:,1,:,:]
    
    Hf_hat = np.fft.fft(np.array(Ht_hat), 256)/20 # 4*256
    Hf_train_hat = np.zeros([batch_size, 2, 4, 256])
    Hf_train_hat[:, 0, :, :] = Hf_hat.real
    Hf_train_hat[:, 1, :, :] = Hf_hat.imag
    
    Hf_train_hat = Hf_train_hat.reshape(batch_size, 2, 2, 2, 256)
    
    return Hf_train_hat
    
    
    
    
    