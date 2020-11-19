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
from Model_define_pytorch_SD_PD import NMSELoss, DatasetFolder, ResNet, U_Net, ResNet34, ResNet18, ResNet50
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'Resnet18')
parser.add_argument('--load_flag', type = bool, default = False)
parser.add_argument('--gpu_list',  type = str,  default='6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument('--lr',  type = float,  default= 1e-3, help='input mode')
args = parser.parse_args()

learning_rate = args.lr  # bigger to train faster
# Parameters for training
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# def seed_everything(seed=42):
#     random.seed(seed)
#     os.environ['PYHTONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

# SEED = 66
# seed_everything(SEED)

batch_size = 512
epochs = 200

lr_threshold = 1e-5
lr_freq = 20

num_workers = 16
print_freq = 30
Pilotnum = 8
mode = args.mode # 0,1,2,-1
SNRdB = -1
load_flag = args.load_flag
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


# Model Construction
if args.model == 'Resnet18':
    model = ResNet18()
elif args.model == 'Resnet34':
    model = ResNet34()
elif args.model == 'Resnet50':
    model = ResNet50()
else:
    model = U_Net()
criterion =  torch.nn.BCELoss(weight=None, reduction='mean')


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda() # model.module
else:
    model = model.cuda()

if load_flag:
    if 'Resnet' in args.model:
        model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_' + args.model + '_SD_mode'+str(mode)+'.pth.tar'
    else:
        model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_Unet_SD_mode'+str(mode)+'.pth.tar'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    print("Weight Loaded!")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# test_data = generator(2000,H_tra,Pilotnum)
train_dataset  = RandomDataset(H_tra,Pilot_num=Pilotnum,SNRdb=SNRdB,mode=mode)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

test_dataset  = RandomDataset(H_val,Pilot_num=Pilotnum,SNRdb=SNRdB,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2000, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True)

print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer.param_groups[0]['lr'])
    # model training

    model.train()
    for it, (Y_train, X_train, H_train) in enumerate(train_dataloader):

        
        Y_input_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F')
        Y_input_train = Y_input_train.float()
        Yd_input_train = Y_input_train[:,:,1,:,:]   # 取出接收数据信号，实部虚部*2*256
        Yp_input_train = Y_input_train[:,:,0,:,:]
        
        Hf_train = np.fft.fft(np.array(H_train), 256)/20 # 4*256
        H_input_train = torch.zeros([batch_size, 2, 4, 256], dtype=torch.float32)
        H_input_train[:, 0, :, :] = torch.tensor(Hf_train.real, dtype = torch.float32)
        H_input_train[:, 1, :, :] = torch.tensor(Hf_train.imag, dtype = torch.float32)

        input = torch.cat([Yp_input_train, Yd_input_train, H_input_train], 2)
        # input = torch.reshape(input, [batch_size, 2, 6*8, 32])
        input = torch.reshape(input, [batch_size, 1, 16, 256])
        input = input.cuda()

        label = X_train.float().cuda()

        optimizer.zero_grad()
        output = model(input)


        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        if it % print_freq == 0:
            print('Net:{0} Mode:{1}'.format(args.model, mode), 'Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.5
        if optimizer.param_groups[0]['lr'] < lr_threshold:
            optimizer.param_groups[0]['lr'] = lr_threshold

    model.eval()

    with torch.no_grad():
        
        print('lr:%.4e'%optimizer.param_groups[0]['lr']) 

        for Y_test, X_test, H_test in test_dataloader:        
            Ns = Y_test.shape[0]
            
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = Y_input_test.float()
            Yd_input_test = Y_input_test[:,:,1,:,:]   # 取出接收数据信号，实部虚部*2*256
            Yp_input_test = Y_input_test[:,:,0,:,:]

            Hf_test = np.fft.fft(np.array(H_test), 256)/20 # 4*256
            H_input_test = torch.zeros([Ns, 2, 4, 256], dtype=torch.float32)
            H_input_test[:, 0, :, :] = torch.tensor(Hf_test.real, dtype = torch.float32)
            H_input_test[:, 1, :, :] = torch.tensor(Hf_test.imag, dtype = torch.float32)

            input = torch.cat([Yp_input_test, Yd_input_test, H_input_test], 2)
            # input = torch.reshape(input, [Ns, 2, 6*8, 32])
            input = torch.reshape(input, [Ns, 1, 16, 256])
            input = input.cuda()

            output = model(input)

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
                    modelSave = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_' + args.model + '_SD_mode'+str(mode)+'.pth.tar'
                else:
                    modelSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/PD_Unet_SD_mode'+str(mode)+'.pth.tar'        
                
                try:
                    torch.save({'state_dict': model.state_dict(), }, modelSave, _use_new_zipfile_serialization=False)
                except:
                    torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
                print('Model saved!')
                best_accuracy = average_accuracy
            
