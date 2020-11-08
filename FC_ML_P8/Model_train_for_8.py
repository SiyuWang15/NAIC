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
from Model_define_pytorch import FC, NMSELoss, DatasetFolder
from LS_CE import LS_Estimation
from generate_data import generator,generatorXY


# Parameters for training
gpu_list = '0,1,2,3'
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
learning_rate = 1e-4  # bigger to train faster
num_workers = 4
print_freq = 100
val_freq = 500
iterations = 10000
Pilotnum = 8

load_flag = 1
best_loss = 100

# channel data for training and validation
data1 = open('H.bin','rb')
H1 = struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1 = np.reshape(H1,[320000,2,4,32])
H_tra = H1[:,1,:,:]+1j*H1[:,0,:,:]   # time-domain channel for training

data2 = open('H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]   # time-domain channel for training


in_dim = 2048
h_dim = 4096
out_dim = 2048
n_blocks =  2

# Model Construction
model = FC(in_dim, h_dim, out_dim, n_blocks)

criterion = NMSELoss(reduction='mean')
criterion_test = NMSELoss(reduction='sum')


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda() # model.module
else:
    model = model.cuda()

if load_flag:
    model_path = './Modelsave/FC_Estimation'+ '_f2f_' + str(in_dim) +'_'+ str(h_dim) +'_'+ str(out_dim) +'_'+ str(n_blocks) + '.pth.tar'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    print("Weight Loaded!")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#### Data loading for training ####
data_load_address = '/data/AI_Wireless/'
Y_train = np.load(data_load_address+'training_Y_P=8.npy')
# Y_train = Y_train[:300000,:]
Ns = Y_train.shape[0]

# Obtain input data based on the LS channel estimation
Hf_partial = LS_Estimation(Y_train, Pilotnum)
# complex ---> real + imag
Hf_input_train = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
Hf_input_train[:, 0, :, :, :] = Hf_partial.real
Hf_input_train[:, 1, :, :, :] = Hf_partial.imag

Hf = np.fft.fft(H_tra, 256) / 20  # frequency-domain channel
Hf = np.reshape(Hf, [Ns, 2, 2, 256], order='F')
# complex ---> real +imag
Hf_label_train = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
Hf_label_train[:, 0, :, :, :] = Hf.real
Hf_label_train[:, 1, :, :, :] = Hf.imag

# dataLoader for training
train_dataset = DatasetFolder(Hf_input_train,Hf_label_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

#### Data loading for validation ####
data_load_address = '/data/AI_Wireless/'
Y_test = np.load(data_load_address+'validation_Y_P=8.npy')
Ns = Y_test.shape[0]

# Obtain input data based on the LS channel estimation
Hf_partial = LS_Estimation(Y_test, Pilotnum)
# complex ---> real + imag
Hf_input_test = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
Hf_input_test[:, 0, :, :, :] = Hf_partial.real
Hf_input_test[:, 1, :, :, :] = Hf_partial.imag

# Obtain label data based on the perfect channel
Hf = np.fft.fft(H_val[:Ns], 256) / 20  # frequency-domain channel
Hf = np.reshape(Hf, [Ns, 2, 2, 256], order='F')
# complex ---> real +imag
Hf_label_test = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
Hf_label_test[:, 0, :, :, :] = Hf.real
Hf_label_test[:, 1, :, :, :] = Hf.imag


# dataLoader for validation
test_dataset = DatasetFolder(Hf_input_test,Hf_label_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2000, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

print('==========Begin Training=========')
best_loss = 100
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer.param_groups[0]['lr'])
    # model training
    model.train()
    if epoch % 100 ==0:
        optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.25

    for i, data in enumerate(train_loader):

        Hf_input, Hf_label = data
        Hf_input = Hf_input.cuda()
        Hf_label = Hf_label.cuda()
        optimizer.zero_grad()
        Hf_hat = model(Hf_input,in_dim, out_dim)
        loss = criterion(Hf_hat, Hf_label)
        loss.backward()
        optimizer.step()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            Hf_input, Hf_label = data
            Hf_input = Hf_input.cuda()
            Hf_label = Hf_label.cuda()
            Hf_hat = model(Hf_input, in_dim, out_dim)
            total_loss += criterion_test(Hf_hat, Hf_label).item()
        average_loss = total_loss / len(test_dataset)
        print('NMSE %.4f' % average_loss)
        if average_loss < best_loss:
            # model save
            modelSave = './Modelsave/FC_Estimation'+ '_f2f_' + str(in_dim) +'_'+ str(h_dim) +'_'+ str(out_dim) +'_'+ str(n_blocks) + '.pth.tar'
            # modelSave = './Modelsave/DnCNN_Estimation_for_' + str(Pilotnum) + '_SNR=100_mode=0_off.pth.tar'
            try:
                torch.save({'state_dict': model.state_dict(), }, modelSave, _use_new_zipfile_serialization=False)
            except:
                torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
            print('Model saved!')
            best_loss = average_loss