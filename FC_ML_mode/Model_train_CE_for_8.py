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
from generate_data import *
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--load_flag', type = bool, default = False)
parser.add_argument('--gpu_list',  type = str,  default='4,5,6,7', help='input gpu list')
parser.add_argument('--mode',  type = int,  default= 0, help='input mode')
parser.add_argument( )
args = parser.parse_args()


# Parameters for training
gpu_list = args.gpu_list
load_flag = args.load_flag
learning_rate = 2e-3  # bigger to train faster

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
epochs = 500

lr_threshold = 1e-5
lr_freq = 10

num_workers = 16
print_freq = 100
val_freq = 100
iterations = 10000
Pilotnum = 8
mode = args.mode # 0,1,2,-1
SNRdB = -1

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

# test_data = generator(2000,H_tra,Pilotnum)
train_dataset  = RandomDataset(H_tra,Pilot_num=Pilotnum,SNRdb=SNRdB,mode=mode)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

test_dataset  = RandomDataset(H_val,Pilot_num=Pilotnum,SNRdb=SNRdB,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2000, shuffle = False, num_workers = num_workers, drop_last = True, pin_memory = True)

print('==========Begin Training=========')
iter = 0
for epoch in range(epochs):
    print('=================================')
    print('lr:%.4e'%optimizer.param_groups[0]['lr'])
    # model training
    model.train()


    for it, (Y_train, H_train, X_train) in enumerate(train_dataloader):

        # print(Y_train)
        Y_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F').float()
        Y_input_train = Y_train[:,:,0,:,:]   # 取出接收导频信号，实部虚部*两根接收天线*256子载波
        Y_input_train = np.reshape(Y_input_train, [batch_size, 2*2*256])
 
        Ht_train = np.reshape(H_train,[batch_size,2,2,32], order='F') # time-domain channel
        Ht_label_train = np.zeros(shape=[batch_size, 2, 2, 2, 32], dtype=np.float32)
        Ht_label_train[:, 0, :, :, :] = Ht_train.real
        Ht_label_train[:, 1, :, :, :] = Ht_train.imag
        Ht_label_train = np.reshape(Ht_label_train , [batch_size, 2*4*32] , order = 'F')

        Y_input_train = torch.Tensor(Y_input_train).cuda()
        Ht_label_train = torch.Tensor(Ht_label_train).cuda()

        optimizer.zero_grad()
        Ht_hat_train = model(Y_input_train)


        loss = criterion(Ht_hat_train, Ht_label_train)

        loss.backward()
        optimizer.step()

        if it % print_freq == 0:
            print('Mode:{0}'.format(mode), 'Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, it, len(train_dataloader), loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if epoch >0:
        if epoch % lr_freq ==0:
            optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.2
        if optimizer.param_groups[0]['lr'] < lr_threshold:
            optimizer.param_groups[0]['lr'] = lr_threshold
    
    model.eval()
    print('lr:%.4e'%optimizer.param_groups[0]['lr']) 
    test_data = generatorXY(2000,H_val,Pilotnum,SNRdB, mode)
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
    H_label_test = np.reshape(H_label_test , [-1, 2*4*32], order = 'F')
    H_label_test = torch.Tensor(H_label_test).to('cuda')


    average_loss = criterion(H_hat_test, H_label_test).item()
    print('NMSE %.4f' % average_loss)
    if average_loss < best_loss:
        # model save
        modelSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_Pilot'+str(Pilotnum)+'_mode'+str(mode)+'.pth.tar'
        # try:
        # torch.save({'state_dict': model.state_dict(), }, modelSave, _use_new_zipfile_serialization=False)
        # except:
        #     torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
        print('Model saved!')
        best_loss = average_loss
    
    model.train()

'''
    # 验证集
    model.eval()
    total_loss = 0
    for  Y_test, H_test, X_test  in test_dataloader:
        N_test = Y_test.shape[0]
        # Obtain input data based on the LS channel estimation

        Y_reshape_test = np.reshape(Y_test, [-1, 2, 2, 2, 256], order='F').float()
        Yp_test = Y_reshape_test[:,:,0,:,:]
        Yp_test = np.reshape(Yp_test, [-1, 2*2*256])
        Yp_test = torch.Tensor(Yp_test).to('cuda')

        H_hat_test = model(Yp_test)

        Ht_test = np.reshape(H_test,[N_test,2,2,32], order='F') # time-domain channel
        Ht_label_test = np.zeros(shape=[N_test, 2, 2, 2, 32], dtype=np.float32)
        Ht_label_test[:, 0, :, :, :] = Ht_test.real
        Ht_label_test[:, 1, :, :, :] = Ht_test.imag
        Ht_label_test = np.reshape(Ht_label_test , [N_test, 2*4*32], order = 'F')
        Ht_label_test = torch.Tensor(Ht_label_test).cuda()    


        average_loss = criterion(H_hat_test, Ht_label_test).item()
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

'''