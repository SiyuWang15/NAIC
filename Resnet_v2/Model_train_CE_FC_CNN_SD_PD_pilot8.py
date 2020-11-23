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
from Model_define_pytorch import NMSELoss, ResNet18, ResNet34,ResNet50,U_Net,FC_ELU_Estimation,CE_ResNet18
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'Resnet18')
parser.add_argument('--gpu_list',  type = str,  default='6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
parser.add_argument('--lr',  type = float,  default= 0, help='input mode')
parser.add_argument('--freeze_CE',  type = int,  default= 1)
parser.add_argument('--load_CE',  type = int,  default= 1)
parser.add_argument('--load_SD',  type = int,  default= 1)
args = parser.parse_args()

print(args.freeze_CE, args.load_CE, args.load_SD)

# Parameters for training
learning_rate = args.lr  # bigger to train faster
SNRdb = args.SNR
mode = args.mode
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list



batch_size = 256
epochs = 300

lr_threshold = 1e-6
lr_freq = 15
num_workers = 16
print_freq = 100
Pilot_num = 8
best_accuracy = 0.5

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


if args.model == 'Resnet18':
    SD = ResNet18()
elif args.model == 'Resnet34':
    SD = ResNet34()
elif args.model == 'Resnet50':
    SD = ResNet50()
else:
    SD = U_Net()
SD = torch.nn.DataParallel( SD ).cuda()  # model.module
if args.load_SD:
    if 'Resnet' in args.model:
        SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_' + args.model + '_SD_mode'+str(mode)+'.pth.tar'
    else:
        SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_Unet_SD_mode'+str(mode)+'.pth.tar' 
    SD.load_state_dict(torch.load(SD_path)['state_dict'])
    print("Weight For SD PD Loaded!")



if args.freeze_CE == 1:
    for params in FC.parameters():
        FC.requires_grad = False
    for params in CNN.parameters():
        CNN.requires_grad = False
    print('freeze CE channel estimation!')
    FC.eval()
    CNN.eval()

if args.freeze_CE == 0:
    optimizer_FC = torch.optim.Adam(FC.parameters(), lr=learning_rate)
    optimizer_CNN = torch.optim.Adam(CNN.parameters(), lr=learning_rate)
optimizer_SD = torch.optim.Adam(SD.parameters(), lr=learning_rate)


criterion_CE =  NMSELoss()
criterion_SD =  torch.nn.BCELoss(weight=None, reduction='mean')

train_dataset  = RandomDataset(H_tra,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True)

test_dataset  = RandomDataset(H_val,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2000, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer_SD.param_groups[0]['lr'])
    # model training

    SD.train()

    if args.freeze_CE == 0:
        FC.train()
        CNN.train()
    else:
        FC.eval()
        CNN.eval()

    for it, (Y_train, X_train, H_train) in enumerate(train_dataloader):
        
        optimizer_SD.zero_grad()
        if args.freeze_CE == 0:
            optimizer_FC.zero_grad()
            optimizer_CNN.zero_grad()

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
        H_train_padding = output2.reshape(batch_size, 2, 4, 32)
        # nmset = criterion_CE(H_train_padding.detach().cpu(), Ht_train_label)
        # print(nmset)
        H_train_padding = torch.cat([H_train_padding, torch.zeros(batch_size,2,4,256-32, requires_grad=True).cuda()],3)
        H_train_padding = H_train_padding.permute(0,2,3,1)

        H_train_padding = torch.fft(H_train_padding, 1)/20
        H_train_padding = H_train_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

        # nmsef = criterion_CE(H_train_padding.detach().cpu(), Hf_train_label)
        # print(nmsef)
        input3 = torch.cat([Yp_input_train.cuda(), Yd_input_train.cuda(), H_train_padding], 2)

        input3 = torch.reshape(input3, [batch_size, 1, 16, 256])

        # 第三层网络的输出
        output3 = SD(input3)

        # 计算loss
        label = X_train.float().cuda()
        loss = criterion_SD(output3, label)
        loss.backward()
        
        optimizer_SD.step()
        if args.freeze_CE == 0:
            optimizer_FC.step()
            optimizer_CNN.step()

        if it % print_freq == 0:
            # print(nmse)
            print('Mode:{0}'.format(mode), 'Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, epochs, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer_SD.param_groups[0]['lr'] =  optimizer_SD.param_groups[0]['lr'] * 0.5
            if args.freeze_CE == 0:
                optimizer_FC.param_groups[0]['lr'] =  optimizer_FC.param_groups[0]['lr'] * 0.5
                optimizer_CNN.param_groups[0]['lr'] =  optimizer_CNN.param_groups[0]['lr'] * 0.5

        if optimizer_SD.param_groups[0]['lr'] < lr_threshold:
            optimizer_SD.param_groups[0]['lr'] = lr_threshold
            if args.freeze_CE == 0:
                optimizer_FC.param_groups[0]['lr'] =  lr_threshold
                optimizer_CNN.param_groups[0]['lr'] =  lr_threshold


    SD.eval()
    FC.eval()
    CNN.eval()
    with torch.no_grad():

        print('lr:%.4e' % optimizer_SD.param_groups[0]['lr'])

        for Y_test, X_test, H_test in test_dataloader:
            Ns = Y_test.shape[0]

            # 真实的频域信道，获取标签
            Hf_test = np.fft.fft(np.array(H_test), 256)/20 # 4*256
            Hf_test_label = torch.zeros([Ns, 2, 4, 256], dtype=torch.float32)
            Hf_test_label[:, 0, :, :] = torch.tensor(Hf_test.real, dtype = torch.float32)
            Hf_test_label[:, 1, :, :] = torch.tensor(Hf_test.imag, dtype = torch.float32)

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

            # 第二层网络输入预处理
            output1 = output1.reshape(Ns, 2, 4, 256)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)

            # 第二层网络的输出
            output2 = CNN(input2)

            #第三层网络输入预处理
            H_test_padding = output2.reshape(Ns, 2, 4, 32)
            H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True).cuda()],3)
            H_test_padding = H_test_padding.permute(0,2,3,1)

            H_test_padding = torch.fft(H_test_padding, 1)/20
            H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

            input3 = torch.cat([Yp_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)

            input3 = torch.reshape(input3, [Ns, 1, 16, 256])

            # 第三层网络的输出
            output3 = SD(input3)

            label = X_test.float().numpy()
            output3 = (output3 > 0.5)*1.
            output3 = output3.cpu().detach().numpy()

            eps = 0.1
            error = np.abs(output3 - label)
            average_accuracy = np.sum(error < eps) / error.size

            print('accuracy %.4f' % average_accuracy)
            if average_accuracy > best_accuracy:
                # model save
                if 'Resnet' in args.model:
                    SDSave = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_' + args.model + '_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'
                else:
                    SDSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_Unet_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                     
                torch.save({'state_dict': SD.module.state_dict(), }, SDSave,_use_new_zipfile_serialization=False)
                print('SD Model saved!')
                
                if args.freeze_CE == 0:
                    FCSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
                    torch.save({'state_dict': FC.module.state_dict(), }, FCSave, _use_new_zipfile_serialization=False)
                    
                    CNNSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/Resnet18_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'

                    torch.save({'state_dict': CNN.module.state_dict(), }, FCSave, _use_new_zipfile_serialization=False)         
                    
                    
                    print('CE Model saved!')

                best_accuracy = average_accuracy
                
