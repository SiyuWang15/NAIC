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
parser.add_argument('--gpu_list',  type = str,  default='6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--SNR',  type = int,  default= -1, help='input mode')
parser.add_argument('--lr',  type = float,  default= 1e-3, help='input mode')
parser.add_argument('--freeze_CE',  type = int,  default= 1)
parser.add_argument('--load_CE',  type = int,  default= 1)
parser.add_argument('--load_SD',  type = int,  default= 0)
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

lr_threshold = 1e-5
lr_freq = 10
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


CE = CE_ResNet18()
CE = torch.nn.DataParallel( CE ).cuda()  # model.module
if args.load_CE:
    CE_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/Resnet18_DirectCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CE.load_state_dict(torch.load( CE_path )['state_dict'])
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
if args.load_SD:
    if 'Resnet' in args.model:
        SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_' + args.model + '_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'
    else:
        SD_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_Unet_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar' 
    SD.load_state_dict(torch.load(SD_path)['state_dict'])
    print("Weight For SD PD Loaded!")



if args.freeze_CE == 1:
    for params in CE.parameters():
        CE.requires_grad = False
    print('freeze CE channel estimation!')
    CE.eval()

if args.freeze_CE == 0:
    optimizer_CE = torch.optim.Adam(CE.parameters(), lr=learning_rate)
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
        CE.train()
    for it, (Y_train, X_train, H_train) in enumerate(train_dataloader):
        
        optimizer_SD.zero_grad()
        if args.freeze_CE == 0:
            optimizer_CE.zero_grad()

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
        H_train_refine = CE(net_input)

        #第二级网络输入
        H_train_padding = H_train_refine.reshape(batch_size, 2, 4, 32)
        H_train_padding = torch.cat([H_train_padding, torch.zeros(batch_size,2,4,256-32, requires_grad=True).cuda()],3)
        H_train_padding = H_train_padding.permute(0,2,3,1)

        H_train_padding = torch.fft(H_train_padding, 1)/20
        H_train_padding = H_train_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

        net2_input = torch.cat([Yp_input_train.cuda(), Yd_input_train.cuda(), H_train_padding], 2)

        net2_input = torch.reshape(net2_input, [batch_size, 1, 16, 256])

        output = SD(net2_input)

        # 计算loss
        label = X_train.float().cuda()
        loss = criterion_SD(output, label)
        loss.backward()
        
        optimizer_SD.step()
        if args.freeze_CE == 0:
            optimizer_CE.step()

        if it % print_freq == 0:
            # print(nmse)
            print('Mode:{0}'.format(mode), 'Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, epochs, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer_SD.param_groups[0]['lr'] =  optimizer_SD.param_groups[0]['lr'] * 0.5
            if args.freeze_CE == 0:
                optimizer_CE.param_groups[0]['lr'] =  optimizer_CE.param_groups[0]['lr'] * 0.5

        if optimizer_SD.param_groups[0]['lr'] < lr_threshold:
            optimizer_SD.param_groups[0]['lr'] = lr_threshold
            if args.freeze_CE == 0:
                optimizer_CE.param_groups[0]['lr'] =  lr_threshold


    SD.eval()
    CE.eval()
    with torch.no_grad():

        print('lr:%.4e' % optimizer_SD.param_groups[0]['lr'])

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
            H_test_refine = CE(net_input)

            #第二级网络输入
            H_test_padding = H_test_refine.reshape(Ns, 2, 4, 32)
            H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True).cuda()],3)
            H_test_padding = H_test_padding.permute(0,2,3,1)

            H_test_padding = torch.fft(H_test_padding, 1)/20
            H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256

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
            if average_accuracy > best_accuracy:
                # model save
                if 'Resnet' in args.model:
                    SDSave = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_' + args.model + '_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'
                else:
                    SDSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_Unet_SD_mode'+str(mode)+'_Pilot'+str(Pilot_num)+'.pth.tar'                     
                try:
                    torch.save({'state_dict': SD.state_dict(), }, SDSave, _use_new_zipfile_serialization=False)
                except:
                    torch.save({'state_dict': SD.module.state_dict(), }, SDSave,_use_new_zipfile_serialization=False)
                print('SD Model saved!')
                
                if args.freeze_CE == 0:
                    CESave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/Resnet18_DirectCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
                    try:
                        torch.save({'state_dict': CE.state_dict(), }, CESave, _use_new_zipfile_serialization=False)
                    except:
                        torch.save({'state_dict': CE.module.state_dict(), }, CESave, _use_new_zipfile_serialization=False)
                    print('CE Model saved!')

                best_accuracy = average_accuracy
                
