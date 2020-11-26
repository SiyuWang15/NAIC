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
from Model_define_pytorch import NMSELoss, FC_ELU_Estimation,CE_ResNet18,DP2DP_U_Net
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'Unet')
parser.add_argument('--gpu_list',  type = str,  default='7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
parser.add_argument('--lr',  type = float,  default= 0, help='input mode')
parser.add_argument('--load_Filter',  type = int,  default= 1)
args = parser.parse_args()

print(args.load_Filter)

# Parameters for training
learning_rate = args.lr  # bigger to train faster
SNRdb = args.SNR
mode = args.mode
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list



batch_size = 256
epochs = 300

lr_threshold = 1e-6
lr_freq = 10
num_workers = 16
print_freq = 100
Pilot_num = 8
best_nmse = 1

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

print('Lovelive')

if args.model == 'Unet':
    Filter = DP2DP_U_Net()
    
Filter = torch.nn.DataParallel( Filter ).cuda()  # model.module
if args.load_Filter:
    Filter_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/DP2DP_' + args.model + '_Filter_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                
    Filter.load_state_dict(torch.load(Filter_path)['state_dict'])
    print("Weight For Filter PD Loaded!")





optimizer_Filter = torch.optim.Adam(Filter.parameters(), lr=learning_rate)


criterion_Filter =  NMSELoss()

train_dataset  = RandomDataset(H_tra,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True)

test_dataset  = RandomDataset(H_val,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2000, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer_Filter.param_groups[0]['lr'])
    # model training

    Filter.train()


    for it, (Y_train, X_train, H_train, Y_woN_train) in enumerate(train_dataloader):
        
        optimizer_Filter.zero_grad()



        # 接收数据与接收导频划分
        Y_input_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F')
        Y_input_train = Y_input_train.float()
        Yp_input_train = Y_input_train[:,:,0,:,:]
        Yd_input_train = Y_input_train[:,:,1,:,:] 

        Y_input_label = np.reshape(Y_woN_train, [batch_size, 2, 2, 2, 256], order='F')
        Y_input_label = Y_input_label.float()
        Yp_input_label = Y_input_label[:,:,0,:,:]
        Yd_input_label = Y_input_label[:,:,1,:,:] 

        input = torch.cat([Yp_input_train.cuda(), Yd_input_train.cuda()], 2)
        label = torch.cat([Yp_input_label.cuda(), Yd_input_label.cuda()], 2)
        # 第三层网络的输出
        output = Filter(input)

        # 计算loss
        loss = criterion_Filter(output, label)
        loss.backward()
        
        optimizer_Filter.step()

        if it % print_freq == 0:
            # print(nmse)
            print('Mode:{0}'.format(mode), 'Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, epochs, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer_Filter.param_groups[0]['lr'] =  optimizer_Filter.param_groups[0]['lr'] * 0.5
        if optimizer_Filter.param_groups[0]['lr'] < lr_threshold:
            optimizer_Filter.param_groups[0]['lr'] = lr_threshold


    Filter.eval()
    with torch.no_grad():

        print('lr:%.4e' % optimizer_Filter.param_groups[0]['lr'])

        for Y_test, X_test, H_test,Y_woN_test in test_dataloader:
            Ns = Y_test.shape[0]


            # 接收数据与接收导频划分
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = Y_input_test.float()
            Yp_input_test = Y_input_test[:,:,0,:,:]
            Yd_input_test = Y_input_test[:,:,1,:,:] 

            Y_input_label = np.reshape(Y_woN_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_label = Y_input_label.float()
            Yp_input_label = Y_input_label[:,:,0,:,:]
            Yd_input_label = Y_input_label[:,:,1,:,:] 

            input = torch.cat([Yp_input_test.cuda(), Yd_input_test.cuda()], 2)
            label = torch.cat([Yp_input_label.cuda(), Yd_input_label.cuda()], 2)
            # 第三层网络的输出
            output = Filter(input)


            # 计算loss
            nmse = criterion_Filter(input, label)
            nmse_p = criterion_Filter(input[:,:,0:2,:], label[:,:,0:2,:])
            nmse_d = criterion_Filter(input[:,:,2:,:], label[:,:,2:,:])
            nmse_p = nmse_p.detach().cpu().numpy()
            nmse_d = nmse_d.detach().cpu().numpy()
            # print('ML Accuracy:%.4f' % average_accuracy_ML, 'Filter Accuracy:%.4f' % average_accuracy_Filter)
            print('Before Filter NMSE:%.4f' % nmse)
            print('Before Filter Pilot NMSE:%.4f' % nmse_p)
            print('Before Filter Data NMSE:%.4f' % nmse_d)

            # 计算loss
            nmse = criterion_Filter(output, label)
            nmse_p = criterion_Filter(output[:,:,0:2,:], label[:,:,0:2,:])
            nmse_d = criterion_Filter(output[:,:,2:,:], label[:,:,2:,:])
            nmse_p = nmse_p.detach().cpu().numpy()
            nmse_d = nmse_d.detach().cpu().numpy()
            # print('ML Accuracy:%.4f' % average_accuracy_ML, 'Filter Accuracy:%.4f' % average_accuracy_Filter)
            print('After Filter NMSE:%.4f' % nmse)
            print('After Filter Pilot NMSE:%.4f' % nmse_p)
            print('After Filter Data NMSE:%.4f' % nmse_d)
            if nmse < best_nmse:
                # model save
                
                FilterSave = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/DP2DP_' + args.model + '_Filter_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                

                torch.save({'state_dict': Filter.module.state_dict(), }, FilterSave,_use_new_zipfile_serialization=False)
                print('Filter Model saved!')
                

                best_nmse = nmse
                
