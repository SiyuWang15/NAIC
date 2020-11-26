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
import DeepRx_port as pD
from Densenet_SD import DenseNet100, DenseNet121,DenseNet201
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from MLreceiver_wo_print import *
from LSreveiver import *
import DeepRx_model
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'HS')
parser.add_argument('--gpu_list',  type = str,  default='6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
parser.add_argument('--lr',  type = float,  default= 1e-3, help='input mode')
parser.add_argument('--freeze_CE',  type = int,  default= 1)
parser.add_argument('--freeze_SD',  type = int,  default= 1)
parser.add_argument('--freeze_CE2',  type = int,  default= 1)
parser.add_argument('--load_CE',  type = int,  default= 1)
parser.add_argument('--load_SD',  type = int,  default= 1)
parser.add_argument('--load_CE2',  type = int,  default= 1)
parser.add_argument('--load_SD2',  type = int,  default= 1)

args = parser.parse_args()


# Parameters for training
learning_rate = args.lr  # bigger to train faster
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


batch_size = 10
epochs = 300

lr_threshold = 1e-6
lr_freq = 10
num_workers = 16
print_freq = 100
Pilot_num = 8
best_acc = 0.5

# channel data for training and validation
data1 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H.bin','rb')
H1 = struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1 = np.reshape(H1,[320000,2,4,32])
H_tra = H1[:,1,:,:]+1j*H1[:,0,:,:]   # time-domain channel for training

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


if args.load_CE:
    path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/best_P' + str(Pilot_num) + '.pth'
    state_dicts = torch.load(path)
    FC.load_state_dict(state_dicts['fc'])
    CNN.load_state_dict(state_dicts['cnn'])
    print("Model for CE has been loaded!")
FC = torch.nn.DataParallel( FC ).cuda()  # model.module
CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
# FC = torch.nn.DataParallel( FC ).cuda()  # model.module
# CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module


SD = XYH2X_ResNet18()

if args.load_SD:
    SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XYH2X_Resnet18_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                
    SD.load_state_dict(torch.load(SD_path)['state_dict'])
    print("Weight For SD PD Loaded!")
SD = torch.nn.DataParallel( SD ).cuda()  # model.module
# SD = SD.cuda()


CE2 = XDH2H_Resnet()
if args.load_CE2:
    CE2_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/XDH2H_Resnet34_CE2_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                
    CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
    print("Weight For CE2 Loaded!")
CE2 = torch.nn.DataParallel( CE2 ).cuda()  # model.module




SD2_path = '/home/hjiang/AI second part/Modelsave0_p'+str(Pilot_num)+'/Rhy5_f4_'
SD2 = pD.DeepRx(SD2_path, cuda_gpu=True, gpus= [0,1])              
print("Weight For SD2 Loaded!")
# SD2[0] = torch.nn.DataParallel( SD2[0] ).cuda()  # model.module
# SD2[1] = torch.nn.DataParallel( SD2[1] ).cuda()  # model.module



for params in FC.parameters():
    FC.requires_grad = False
for params in CNN.parameters():
    CNN.requires_grad = False
for params in SD.parameters():
    SD.requires_grad = False
for params in CE2.parameters():
    CE2.requires_grad = False

FC.eval()
CNN.eval()
SD.eval()
CE2.eval()
print('freeze CE channel estimation!')
print('freeze SD channel estimation!')
print('freeze CE2 channel estimation!')

SD2[0].train()
SD2[1].train()

optimizer_SD2 = []
optimizer_SD2.append(torch.optim.Adam(SD2[0].parameters(), lr=learning_rate))
optimizer_SD2.append(torch.optim.Adam(SD2[1].parameters(), lr=learning_rate))


criterion_SD =  torch.nn.BCELoss(weight=None, reduction='mean')

train_dataset  = RandomDataset(H_tra,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True)

test_dataset  = RandomDataset(H_val,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2000, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)



print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e' % optimizer_SD2[0].param_groups[0]['lr'])
    # model training

    SD2[0].train()
    SD2[1].train()

    for it, (Y_train, X_train, H_train) in enumerate(train_dataloader):
        
        optimizer_SD2[0].zero_grad()
        optimizer_SD2[1].zero_grad()

        # 真实的频域信道，获取标签
        Hf_train = np.fft.fft(np.array(H_train), 256)/20 # 4*256
        Hf_train_label = torch.zeros([batch_size, 2, 4, 256], dtype=torch.float32)
        Hf_train_label[:, 0, :, :] = torch.tensor(Hf_train.real, dtype = torch.float32)
        Hf_train_label[:, 1, :, :] = torch.tensor(Hf_train.imag, dtype = torch.float32)
        # 真实的时域信道，获取标签
        Ht_train_label = torch.zeros([batch_size, 2, 4, 32], dtype=torch.float32)
        Ht_train_label[:, 0, :, :] = H_train.real.float()
        Ht_train_label[:, 1, :, :] = H_train.imag.float()

        # 接收数据与接收导频划分
        Y_input_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F')
        Y_input_train = Y_input_train.float()
        Yp_input_train = Y_input_train[:,:,0,:,:]
        Yd_input_train = Y_input_train[:,:,1,:,:] 


        # 第一层网络输入
        input1 = Yp_input_train.reshape(batch_size, 2*2*256) # 取出接收导频信号，实部虚部*2*256
        input1 = input1.cuda()
        # 第一层网络输出
        output1 = FC(input1)

        # 第二层网络输入预处理
        output1 = output1.reshape(batch_size, 2, 4, 256)
        input2 = torch.cat([Yd_input_train.cuda(), Yp_input_train.cuda(), output1], 2)

        # 第二层网络的输出
        output2 = CNN(input2)

        #第三层网络输入预处理
        '''
        输入时域信道，输出频域信道
        '''
        H_train_padding = output2.reshape(batch_size, 2, 4, 32)
        H_train_padding = torch.cat([H_train_padding, torch.zeros(batch_size,2,4,256-32, requires_grad=True).cuda()],3)
        H_train_padding = H_train_padding.permute(0,2,3,1)
        H_train_padding = torch.fft(H_train_padding, 1)/20
        H_train_padding = H_train_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256
        '''
        以上
        '''

        # LS 初步估计
        X_LS = get_LS(Yd_input_train, H_train_padding.detach().cpu())
        
        X_input_train = torch.zeros([batch_size, 2, 2, 256], dtype = torch.float32)
        X_input_train[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
        X_input_train[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)
        
        input3 = torch.cat([X_input_train.cuda(), Yp_input_train.cuda(), Yd_input_train.cuda(), H_train_padding], 2)

        # 第三层网络的输出
        output3 = SD(input3)

    
        # 第四层网络输入, X,Yd,H_train_padding
        output3 = output3.reshape([batch_size,2,256,2])
        output3 = output3.permute(0,3,1,2).contiguous()

        input4 = torch.cat([output3, Yd_input_train.cuda(), H_train_padding], 2)
        # 第四层网络输出
        output4 = CE2(input4)
        output4 = output4.reshape(batch_size, 2, 4, 32)

        # 第五层网络预处理
        H_train_padding2 = output4
        H_train_padding2 = torch.cat([H_train_padding2, torch.zeros(batch_size,2,4,256-32, requires_grad=True).cuda()],3)
        H_train_padding2 = H_train_padding2.permute(0,2,3,1)
        H_train_padding2 = torch.fft(H_train_padding2, 1)/20
        H_train_padding2 = H_train_padding2.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256


        # 第五层网络的输出
        output5 = pD.Pred_DeepRX(SD2, torch.reshape(H_train_padding2, [-1, 2,2,2,256]), Yd_input_train.cuda())
        label = X_train.float().cuda()
        loss = criterion_SD(output5, label)
        loss.backward()
        
        optimizer_SD2[0].step()
        optimizer_SD2[1].step()

        if it % print_freq == 0:
            # print(nmse)
            print('Mode:{0}'.format(mode), 'Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, epochs, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer_SD2[0].param_groups[0]['lr'] =  optimizer_SD2[0].param_groups[0]['lr'] * 0.5
            optimizer_SD2[1].param_groups[0]['lr'] =  optimizer_SD2[1].param_groups[0]['lr'] * 0.5
   
        if optimizer_SD2[0].param_groups[0]['lr'] < lr_threshold:
            optimizer_SD2[0].param_groups[0]['lr'] = lr_threshold
            optimizer_SD2[1].param_groups[0]['lr'] = lr_threshold





    SD2[0].eval()
    SD2[1].eval()

    with torch.no_grad():

        print('lr:%.4e' % optimizer_SD2[0].param_groups[0]['lr'])
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
            input1 = Yp_input_test.reshape(Ns, 2*2*256).cuda() # 取出接收导频信号，实部虚部*2*256
            # 第一层网络输出
            output1 = FC(input1)

            # 第二层网络输入预处理
            output1 = output1.reshape(Ns, 2, 4, 256)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)

            # 第二层网络的输出
            output2 = CNN(input2)
            output2 = output2.reshape(Ns, 2, 4, 32)
            #第三层网络输入预处理
            H_test_padding = output2
            H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True).cuda()],3)
            H_test_padding = H_test_padding.permute(0,2,3,1)

            H_test_padding = torch.fft(H_test_padding, 1)/20
            H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256


            X_LS = get_LS(Yd_input_test, H_test_padding.detach().cpu())
            X_input_test = torch.zeros([Ns, 2, 2, 256], dtype = torch.float32)
            X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
            X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)


            
            input3 = torch.cat([X_input_test.cuda(), Yp_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)

            # 第三层网络的输出
            output3 = SD(input3)

            
            output3 = output3.reshape([Ns,2,256,2])
            output3 = output3.permute(0,3,1,2).contiguous()

            # 第四层网络的输入
            # input4 = torch.cat([X_input_test, Yd_input_test, H_test_padding], 2)
            input4 = torch.cat([output3, Yd_input_test.cuda(), H_test_padding], 2)

            output4 = CE2(input4)
            output4 = output4.reshape(Ns, 2, 4, 32)
            

            #第五层网络输入预处理
            H_test_padding2 = output4
            H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True).cuda()],3)
            H_test_padding2 = H_test_padding2.permute(0,2,3,1)

            H_test_padding2 = torch.fft(H_test_padding2, 1)/20
            H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256



            # 第五层网络的输出
            output5 = pD.Pred_DeepRX(SD2, torch.reshape(H_test_padding2, [-1, 2,2,2,256]), Yd_input_test.cuda())
            eps = 0.1
            output5 = (output5 > 0.5)*1.
            output5 = output5.cpu().detach().numpy()
            label = X_test.float().numpy()

            error = np.abs(output5 - label)
            average_acc = np.sum(error < eps) / error.size 


                
            if average_acc > best_acc:
                # model save
                DeepRx_model.Save_Model(SD, 2, '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/HS_SD2')
                print('SD2 Model saved!')
                best_acc = average_acc