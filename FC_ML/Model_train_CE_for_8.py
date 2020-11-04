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
from Model_define_pytorch import FC_Estimation, FC_Estimation32, NMSELoss, DatasetFolder, DnCNN
from LS_CE import LS_Estimation, LS_Estimation32
from generate_data import generator,generatorXY


# Parameters for training
gpu_list = '4,5,6,7'
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
print_freq = 20
val_freq = 100
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


# Model Construction
model = FC_Estimation(2048, 4096, 4096, 2048)
# model = DnCNN()
criterion = NMSELoss(reduction='mean')
criterion_test = NMSELoss(reduction='sum')


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda() # model.module
else:
    model = model.cuda()

if load_flag:
    model_path = './Modelsave/FC_Estimation_for_'+str(Pilotnum)+'.pth.tar'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    print("Weight Loaded!")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# generate data fro training
train_data = generator(batch_size,H_tra,Pilotnum)
test_data = generatorXY(2000,H_val,Pilotnum)

print('==========Begin Training=========')
iter = 0
for Y,X,H in train_data:
    iter = iter+1
    # Obtain input data based on the LS channel estimation
    Hf_partial = LS_Estimation(Y, Pilotnum)
    # complex ---> real + imag
    Hf_input_train = np.zeros(shape=[batch_size,2,2,2,256],dtype=np.float32)
    Hf_input_train[:,0,:,:,:] = Hf_partial.real
    Hf_input_train[:,1,:,:,:] = Hf_partial.imag

    # Obtain label data based on the perfect channel
    Hf = np.fft.fft(H, 256) / 20 # frequency-domain channel
    Hf = np.reshape(Hf, [batch_size, 2, 2, 256], order='F')
    # complex ---> real +imag
    Hf_label_train = np.zeros(shape=[batch_size,2,2,2,256],dtype=np.float32)
    Hf_label_train[:,0,:,:,:] = Hf.real
    Hf_label_train[:,1,:,:,:] = Hf.imag

    Hf_input_train = torch.Tensor(Hf_input_train).to('cuda')
    Hf_label_train = torch.Tensor(Hf_label_train).to('cuda')
    Hf_hat_train = model(Hf_input_train)
    loss = criterion(Hf_hat_train, Hf_label_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if iter % print_freq == 0:
        # print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        # print('Iter: {}\t' 'Loss {loss:.4f}\t'.format(iter, loss=loss.item()))
        print('Completed iterations [%d]\t' % iter, 'Loss {loss:.4f}\t'.format(loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if iter % val_freq == 0:
        model.eval()
        Y,X,H = test_data
        Ns = Y.shape[0]
        # Obtain input data based on the LS channel estimation
        Hf_partial = LS_Estimation(Y, Pilotnum)
        # complex ---> real + imag
        Hf_input_test = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
        Hf_input_test[:, 0, :, :, :] = Hf_partial.real
        Hf_input_test[:, 1, :, :, :] = Hf_partial.imag

        # Obtain label data based on the perfect channel
        Hf = np.fft.fft(H, 256) / 20  # frequency-domain channel
        Hf = np.reshape(Hf, [Ns, 2, 2, 256], order='F')
        # complex ---> real +imag
        Hf_label_test = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
        Hf_label_test[:, 0, :, :, :] = Hf.real
        Hf_label_test[:, 1, :, :, :] = Hf.imag

        Hf_input_test = torch.Tensor(Hf_input_test).to('cuda')
        Hf_label_test = torch.Tensor(Hf_label_test).to('cuda')
        Hf_hat_test = model(Hf_input_test)
        average_loss = criterion(Hf_hat_test, Hf_label_test).item()
        print('NMSE %.4f' % average_loss)
        if average_loss < best_loss:
            # model save
            modelSave =  './Modelsave/FC_Estimation_for_'+str(Pilotnum)+'.pth.tar'
            # try:
            torch.save({'state_dict': model.state_dict(), }, modelSave, _use_new_zipfile_serialization=False)
            # except:
            #     torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
            print('Model saved!')
            best_loss = average_loss
        
        model.train()

    if iter>iterations:
        break