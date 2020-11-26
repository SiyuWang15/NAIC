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
from Model_define_pytorch_CE_CNN import *
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'Resnet18')
parser.add_argument('--gpu_list',  type = str,  default='6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--lr',  type = float,  default= 1e-3, help='input mode')
parser.add_argument('--freeze_FC',  type = int,  default= 1)
parser.add_argument('--load_FC',  type = int,  default= 1)
parser.add_argument('--load_CNN',  type = int,  default= 0)
args = parser.parse_args()

print(args.freeze_FC, args.load_FC, args.load_CNN)
learning_rate = args.lr  # bigger to train faster
# Parameters for training
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list



batch_size = 256
epochs = 100
learning_rate = args.lr  # bigger to train faster
lr_threshold = 1e-5
lr_freq = 30

num_workers = 16
print_freq = 100
Pilot_num = 8
SNRdb = -1
mode = args.mode
best_loss = 10000

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


if args.model == 'Resnet18':
    CNN = CE_ResNet18()
elif args.model == 'Resnet34':
    CNN = CE_ResNet34()
elif args.model == 'Resnet50':
    CNN = CE_ResNet50()

CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
if args.load_CNN:
    CNN_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/'+ str(args.model) +'_DirectCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CNN.load_state_dict(torch.load( CNN_path )['state_dict'])
    print("Model for CNN CE has been loaded!")



criterion =  NMSELoss()


optimizer_CNN = torch.optim.Adam(CNN.parameters(), lr=learning_rate)



train_dataset  = RandomDataset(H_tra,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True)

test_dataset  = RandomDataset(H_val,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2000, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer_CNN.param_groups[0]['lr'])
    # model training

    CNN.train()


    for it, (Y_train, X_train, H_train) in enumerate(train_dataloader):
        
        optimizer_CNN.zero_grad()


        # 真实的频域信道，获取标签
        Hf_train = np.fft.fft(np.array(H_train), 256)/20 # 4*256
        Hf_train_label = torch.zeros([batch_size, 2, 4, 256], dtype=torch.float32)
        Hf_train_label[:, 0, :, :] = torch.tensor(Hf_train.real, dtype = torch.float32)
        Hf_train_label[:, 1, :, :] = torch.tensor(Hf_train.imag, dtype = torch.float32)

        # 第一层网络输入
        Y_input_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F')
        Y_input_train = Y_input_train.float()
        Yp_input_train = Y_input_train[:,:,0,:,:]
        Yd_input_train = Y_input_train[:,:,1,:,:] 


        net_input = torch.cat([Yp_input_train, Yd_input_train], 2)
        net_input = torch.reshape(net_input, [batch_size, 1, 8, 256])

        net_input = net_input.cuda()
        
        Ht_train_refine = CNN(net_input)

        #第二级网络输出
        Ht_train_refine = Ht_train_refine.reshape(batch_size, 2, 4, 32)

        # 第二级标签
        Ht_train_label = torch.zeros([batch_size, 2, 4, 32], dtype=torch.float32)
        Ht_train_label[:, 0, :, :] = H_train.real.float()
        Ht_train_label[:, 1, :, :] = H_train.imag.float()

        # 计算loss
        loss = criterion(Ht_train_refine, Ht_train_label.cuda())
        loss.backward()
        optimizer_CNN.step()
        if args.freeze_FC == 0:
            optimizer_FC.step()

        if it % print_freq == 0:
            # print(nmse)
            print('Mode:{0}'.format(mode), 'Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, epochs, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer_CNN.param_groups[0]['lr'] =  optimizer_CNN.param_groups[0]['lr'] * 0.5
     
        if optimizer_CNN.param_groups[0]['lr'] < lr_threshold:
            optimizer_CNN.param_groups[0]['lr'] = lr_threshold



    CNN.eval()
    with torch.no_grad():

        print('lr:%.4e' % optimizer_CNN.param_groups[0]['lr'])

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
            Yp_input_test = Y_input_test[:,:,0,:,:]
            Yd_input_test = Y_input_test[:,:,1,:,:] 

            net_input = torch.cat([Yp_input_test, Yd_input_test], 2)
            net_input = torch.reshape(net_input, [Ns, 1, 8, 256])
            
            net_input = net_input.cuda()
            
            Ht_test_refine = CNN(net_input)

            #第二级网络输出
            Ht_test_refine = Ht_test_refine.reshape(Ns, 2, 4, 32)

            # 第二级标签
            Ht_test_label = torch.zeros([Ns, 2, 4, 32], dtype=torch.float32)
            Ht_test_label[:, 0, :, :] = H_test.real.float()
            Ht_test_label[:, 1, :, :] = H_test.imag.float()

            # 计算loss
            loss = criterion(Ht_test_refine, Ht_test_label.cuda())

            print('NMSE %.4f' % loss)
            if loss < best_loss:
                CNNSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/'+ str(args.model)+'_DirectCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'

                torch.save({'state_dict': CNN.module.state_dict(), }, CNNSave,  _use_new_zipfile_serialization=False)
                print('CNN Model saved!')

                best_loss = loss

