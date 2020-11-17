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
from Model_define_pytorch import *
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_list',  type = str,  default='4,5,6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--lr',  type = float,  default= 1e-3, help='input mode')
parser.add_argument('--freeze_FC',  type = bool,  default= False)
parser.add_argument('--load_FC',  type = bool,  default= False)
args = parser.parse_args()

learning_rate = args.lr  # bigger to train faster
# Parameters for training
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list



batch_size = 512
epochs = 100 # bigger to train faster
lr_threshold = 1e-5
lr_freq = 5

num_workers = 16
print_freq = 100
val_freq = 100
iterations = 10000
Pilot_num = 8
mode = 0
SNRdb = -1
load_flag = 1
best_loss = 10

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
FC = torch.nn.DataParallel( FC ).cuda()  # model.module
# Load weights
if args.load_FC:
    FC_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    FC.load_state_dict(torch.load( FC_path )['state_dict'])
    print("Model for FC CE has been loaded!")

if args.freeze_FC:
    for params in FC.parameters():
        FC.requires_grad = False
    print('freeze FC channel estimation!')


criterion =  NMSELoss()


optimizer = torch.optim.Adam(FC.parameters(), lr=learning_rate)


train_dataset  = RandomDataset(H_tra,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True)

test_dataset  = RandomDataset(H_val,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2000, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer.param_groups[0]['lr'])
    # model training

    FC.train()
    for it, (Y_train, X_train, H_train) in enumerate(train_dataloader):
        
        optimizer.zero_grad()

        # 真实的频域信道，获取标签
        Hf_train = np.fft.fft(np.array(H_train), 256)/20 # 4*256
        Hf_train_label = torch.zeros([batch_size, 2, 4, 256], dtype=torch.float32)
        Hf_train_label[:, 0, :, :] = torch.tensor(Hf_train.real, dtype = torch.float32)
        Hf_train_label[:, 1, :, :] = torch.tensor(Hf_train.imag, dtype = torch.float32)

        # 第一层网络输入
        Y_input_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F')
        Y_input_train = Y_input_train.float()
        Yp_train = Y_input_train[:,:,0,:,:]

        #输入网络
        Yp_train = Yp_train.reshape(batch_size, 2*2*256) # 取出接收导频信号，实部虚部*2*256
        Yp_train =  Yp_train.cuda()
        Hf_train_output = FC(Yp_train)
        # 第一级网络输出
        Hf_train_output = Hf_train_output.reshape(batch_size, 2, 4, 256)

        # 计算loss
        loss = criterion(Hf_train_output, Hf_train_label.cuda())

        loss.backward()
        optimizer.step()

        if it % print_freq == 0:
            # print(nmse)
            print('Mode:{0}'.format(mode), 'Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, epochs, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.2
        if optimizer.param_groups[0]['lr'] < lr_threshold:
            optimizer.param_groups[0]['lr'] = lr_threshold

    FC.eval()

    with torch.no_grad():

        print('lr:%.4e' % optimizer.param_groups[0]['lr'])

        for Y_test, X_test, H_test in test_dataloader:
            Ns = Y_test.shape[0]

            # 真实的频域信道，获取标签
            Hf_test = np.fft.fft(np.array(H_test), 256)/20 # 4*256
            Hf_test_label = torch.zeros([Ns, 2, 4, 256], dtype=torch.float32)
            Hf_test_label[:, 0, :, :] = torch.tensor(Hf_test.real, dtype = torch.float32)
            Hf_test_label[:, 1, :, :] = torch.tensor(Hf_test.imag, dtype = torch.float32)

            # 第一层网络输入
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = Y_input_test.float()

            Yp_test = Y_input_test[:,:,0,:,:] # 取出接收导频信号，实部虚部*2*256
            Yp_test = Yp_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
            Yp_test = Yp_test.cuda()
            Hf_test_output = FC(Yp_test)
            # 第一级网络输出
            Hf_test_output = Hf_test_output.reshape(Ns, 2, 4, 256)

            # 计算loss
            loss = criterion(Hf_test_output, Hf_test_label.cuda())

            print('NMSE %.4f' % loss)
            if loss < best_loss:
                # model save
                # if 'Resnet' in args.model:
                #     modelSave = '/data/XiuhongWei/AI_competition/OFDMReceiver/Modelsave/' + args.model + '_SD_mode' + str(
                #         mode) + '.pth.tar'
                # else:
                #     modelSave = '/data/XiuhongWei/AI_competition/OFDMReceiver/Modelsave/Unet_SD_mode' + str(
                #         mode) + '.pth.tar'
                FCSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
                try:
                    torch.save({'state_dict': FC.state_dict(), }, FCSave, _use_new_zipfile_serialization=False)
                except:
                    torch.save({'state_dict': FC.module.state_dict(), }, FCSave,
                               _use_new_zipfile_serialization=False)
                print('Model saved!')
                best_loss = loss

