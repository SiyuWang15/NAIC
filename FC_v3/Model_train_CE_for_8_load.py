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
from Model_define_pytorch import FC_Estimation, NMSELoss, DatasetFolder, DnCNN
from generate_data import generator,generatorXY
import argparse
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--load_flag', type = bool, default = False)
parser.add_argument('--gpu_list',  type = str,  default='4,5,6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= -1, help='input mode')
args = parser.parse_args()


# Parameters for training
gpu_list = args.gpu_list
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)

batch_size = 256
epochs = 200
learning_rate = 1e-3  # bigger to train faster
lr_threshold = 1e-8
lr_freq = 10

num_workers = 4
print_freq = 100
val_freq = 500
iterations = 10000
Pilotnum = 8
mode = args.mode # 0,1,2,-1
load_flag = args.load_flag
best_loss = 100

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
in_dim = 1024
h_dim = 4096
out_dim = 256
n_blocks =  2
model = FC_Estimation(in_dim, h_dim, out_dim, n_blocks)
# model = DnCNN()
criterion = NMSELoss()
criterion_test = NMSELoss()


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda() # model.module
else:
    model = model.cuda()

if load_flag:
    model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_Pilot'+str(Pilotnum)+'_mode'+str(mode)+'.pth.tar'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    print("Weight Loaded!")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# generate data fro training
data_load_address = '/data/CuiMingyao/AI_competition/OFDMReceiver/'
Y_train = np.load(data_load_address+'training_Y_P='+str(Pilotnum)+'_mode='+str(mode)+'.npy')

# Y_train = Y_train[:300000,:]
N_train = Y_train.shape[0]
Y_train = Y_train.astype(np.float32)

# Obtain input data
Y_train = np.reshape(Y_train, [-1, 2, 2, 2, 256], order='F')
Y_input_train = Y_train[:,:,0,:,:]   # 取出接收导频信号，实部虚部*两根接收天线*256子载波
Y_input_train = np.reshape(Y_input_train, [N_train, 2*2*256])

# Obtain label data for training
Ht = np.reshape(H_tra,[-1,2,2,32], order='F') # time-domain channel
Ht_label_train = np.zeros(shape=[N_train, 2, 2, 2, 32], dtype=np.float32)
Ht_label_train[:, 0, :, :, :] = Ht.real
Ht_label_train[:, 1, :, :, :] = Ht.imag
Ht_label_train = np.reshape(Ht_label_train , [-1, 2*4*32])

# dataLoader for training
train_dataset = DatasetFolder(Y_input_train, Ht_label_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

# test_data = generator(2000,H_tra,Pilotnum)

print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer.param_groups[0]['lr'])
    # model training
    model.train()
    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.2
        if optimizer.param_groups[0]['lr'] < lr_threshold:
            optimizer.param_groups[0]['lr'] = lr_threshold

    for i, data in enumerate(train_loader):
        Y_input, Ht_label = data
        Y_input = Y_input.cuda()
        Ht_label_train = Ht_label.cuda()
        optimizer.zero_grad()
        Ht_hat_train = model(Y_input)


        loss = criterion(Ht_hat_train, Ht_label_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # if i > 200:
        #     break


    model.eval()
    Y_test, X_test, H_test = generatorXY(2000,H_val,Pilotnum)
    N_test = Y_test.shape[0]
    # Obtain input data based on the LS channel estimation

    Y_reshape_test = np.reshape(Y_test, [-1, 2, 2, 2, 256], order='F')
    Yp_test = Y_reshape_test[:,:,0,:,:]
    Yp_test = np.reshape(Yp_test, [-1, 2*2*256])
    Yp_test = torch.Tensor(Yp_test).to('cuda')

    H_hat_test = model(Yp_test)


    H_label_test =  np.zeros(shape=[N_test,2,4,32],dtype=np.float32)
    H_label_test[:,0,:,:] = H_test.real
    H_label_test[:,1,:,:] = H_test.imag
    H_label_test = np.reshape(H_label_test , [-1, 2*4*32])
    H_label_test = torch.Tensor(H_label_test).to('cuda')


    average_loss = criterion(H_hat_test, H_label_test).item()
    print('NMSE %.4f' % average_loss)
    if average_loss < best_loss:
        # model save
        modelSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_Pilot'+str(Pilotnum)+'_mode'+str(mode)+'.pth.tar'
        try:
            torch.save({'state_dict': model.state_dict(), }, modelSave, _use_new_zipfile_serialization=False)
        except:
            torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
        print('Model saved!')
        best_loss = average_loss
    
    model.train()

