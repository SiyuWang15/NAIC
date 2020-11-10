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
num_workers = 4
print_freq = 20
val_freq = 400
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
h_dim = 2048
out_dim = 256
n_blocks =  6
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
scheduler = lr_scheduler.StepLR(optimizer, step_size = 500, gamma=0.5) 

# generate data fro training
train_data = generator(batch_size,H_tra,Pilotnum)
# test_data = generatorXY(2000,H_val,Pilotnum,mode)

print('==========Begin Training=========')
iter = 0
for Y,X,H in train_data:
    iter = iter+1
    # Obtain input data based on the LS channel estimation
    optimizer.zero_grad()

    Y_reshape = np.reshape(Y, [batch_size, 2, 2, 2, 256], order='F')
    Yp = Y_reshape[:,:,0,:,:]

    
    Yp = np.reshape(Yp, [batch_size, 2*2*256])
    Yp = torch.Tensor(Yp).to('cuda')
    # print(Yp.shape)

    H_hat_train = model(Yp)


    H_label_train =  np.zeros(shape=[batch_size,2,4,32],dtype=np.float32)
    H_label_train[:,0,:,:] = H.real
    H_label_train[:,1,:,:] = H.imag
    H_label_train = np.reshape(H_label_train , [-1, 2*4*32])
    H_label_train = torch.Tensor(H_label_train).to('cuda')

    loss = criterion(H_hat_train, H_label_train)
    loss.backward()
    optimizer.step()


    scheduler.step()
    if iter % print_freq == 0:
        # print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        # print('Iter: {}\t' 'Loss {loss:.4f}\t'.format(iter, loss=loss.item()))
        print('Completed iterations [%d]\t' % iter, 'Loss {loss:.4f}\t'.format(loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if iter % val_freq == 0:
        model.eval()
        print('lr:%.4e'%optimizer.param_groups[0]['lr']) 
        test_data = generatorXY(2000,H_val,Pilotnum,mode)
        Y_test, X_test, H_test = test_data
        Ns = Y_test.shape[0]
        # Obtain input data based on the LS channel estimation

        Y_reshape = np.reshape(Y_test, [-1, 2, 2, 2, 256], order='F')
        Yp = Y_reshape[:,:,0,:,:]
        Yp = np.reshape(Yp, [-1, 2*2*256])
        Yp = torch.Tensor(Yp).to('cuda')

        H_hat_test = model(Yp)


        H_label_test =  np.zeros(shape=[Ns,2,4,32],dtype=np.float32)
        H_label_test[:,0,:,:] = H_test.real
        H_label_test[:,1,:,:] = H_test.imag
        H_label_test = np.reshape(H_label_test , [-1, 2*4*32])
        H_label_test = torch.Tensor(H_label_test).to('cuda')


        average_loss = criterion(H_hat_test, H_label_test).item()
        print('NMSE %.4f' % average_loss)
        if average_loss < best_loss:
            # model save
            modelSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_Pilot'+str(Pilotnum)+'_mode'+str(mode)+'.pth.tar'
            # try:
            torch.save({'state_dict': model.state_dict(), }, modelSave, _use_new_zipfile_serialization=False)
            # except:
            #     torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
            print('Model saved!')
            best_loss = average_loss
        
        model.train()

    if iter>iterations:
        break