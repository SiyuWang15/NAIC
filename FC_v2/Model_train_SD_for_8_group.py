from utils import *
import struct
import os
import torch.nn as nn
from torch.optim import lr_scheduler
import random
import torch
import h5py
from scipy.io import loadmat
from scipy.io import savemat
import time
from Model_define_pytorch import FC_Detection, FC_Estimation, NMSELoss, DatasetFolder, DnCNN
from LS_CE import LS_Estimation
from generate_data import generator,generatorXY
import argparse
import itertools
parser = argparse.ArgumentParser()
parser.add_argument('--group_index', type = int, nargs='+', default = [0,1,2,3], help = 'input group index')
parser.add_argument('--load_SD', type = bool, default = False)
parser.add_argument('--gpu_list',  type = str,  default='4,5,6,7', help='input gpu list')
args = parser.parse_args()

group_index = args.group_index
print('group indx:',group_index)
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
val_freq = 100
# print_freq = 2
# val_freq = 10
iterations = 10000
Pilotnum = 8
freeze_CE = True
load_CE = True
load_SD = args.load_SD
group_num = 32




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
in_dim = 2048
h_dim = 4096
out_dim = 2048
n_blocks =  2
CE_model = FC_Estimation(in_dim, h_dim, out_dim, n_blocks)


# 分成16组
G = 256 // group_num
N_train_groups = len(group_index)
SD_model = []
for idx in range(N_train_groups):
    SD_model.append(FC_Detection(G*(4+8),  G*4, [512,1024,2048,2048,1024,512]))

# model = DnCNN()
# criterion = NMSELoss(reduction='mean')
# criterion_test = NMSELoss(reduction='sum')
CE_criterion = NMSELoss(reduction='mean')
CE_criterion_test = NMSELoss(reduction='sum')
criterion =  torch.nn.BCELoss(weight=None, reduction='mean')


if len(gpu_list.split(',')) > 1:
    CE_model = torch.nn.DataParallel(CE_model).cuda() # model.module
    for idx in range(N_train_groups):
        SD_model[idx] = torch.nn.DataParallel(SD_model[idx]).cuda() # model.module
else:
    CE_model = CE_model.cuda()
    for idx in range(N_train_groups):
        SD_model[idx] = SD_model[idx].cuda()


if load_CE:
    CE_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation'+ '_f2f_' +'Pilot'+str(Pilotnum)+'_'+ str(in_dim) +'_'+ str(h_dim) +'_'+ str(out_dim) +'_'+ str(n_blocks) + '.pth.tar'
    CE_model.load_state_dict(torch.load(CE_model_path)['state_dict'])
    print("CE Weight Loaded!")

if freeze_CE:
    for params in CE_model.parameters():
        params.requires_grad = False
    CE_model.eval()
    print('freeze CE!')

if load_SD:
    for idx in range(N_train_groups):
        SD_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Detection_Pilot_'+str(Pilotnum)+'_'+str(group_num)+'Group'+str(group_index[idx])+'.pth.tar'
        SD_load_model = torch.load(SD_model_path)['state_dict']
        SD_model_dict = SD_model[idx].state_dict()
        SD_state_dict = {k:v for k,v in SD_load_model.items() if k in SD_model_dict.keys()}
        SD_model_dict.update(SD_state_dict)
        SD_model[idx].load_state_dict( SD_model_dict )
        print("SD Weight Loaded!")

optimizer = []
for idx in range(N_train_groups):
    optimizer.append( torch.optim.Adam(SD_model[idx].parameters(), lr=learning_rate) )
scheduler = []
for idx in range(N_train_groups):
    scheduler.append( lr_scheduler.StepLR(optimizer[idx], step_size = 2000, gamma=0.5) )

# generate data fro training
train_data = generator(batch_size,H_tra, Pilotnum)
test_data = generatorXY(2000, H_val, Pilotnum)

print('==========Begin Training=========')
iter = 0
best_accuracy = np.ones((N_train_groups,1))*0.5

for idx in range(N_train_groups):
    SD_model[idx].train()

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
    for item in range(N_train_groups):
        optimizer[item].zero_grad()

    # 信道估计
    Hf_input_train = torch.Tensor(Hf_input_train).to('cuda')
    Hf_label_train = torch.Tensor(Hf_label_train).to('cuda')
    Hf_hat_train = CE_model(Hf_input_train)


    Y_reshape = np.reshape(Y, (-1, 2,2,2, 256), order = 'F') 
    X_reshape = np.reshape(X,  (-1, 2,512) )
    loss = []
    for item in range(N_train_groups):
        # 提取X的第index组
        X_item = X_reshape[:,:, 2*G*group_index[item]: 2*G*(group_index[item] + 1)]
        X_item = np.reshape(X_item, (-1, G*4))

        # 提取H的第index组
        Hf_hat_train_item = Hf_hat_train[:,:,:,:, G*group_index[item]: G*(group_index[item]+1)]
        Hf_hat_train_item = Hf_hat_train_item.reshape(-1, G*8)

        Hf_label_train_item = Hf_label_train[:,:,:,:,  G*group_index[item]: G*(group_index[item]+1)]
        Hf_label_train_item = Hf_label_train_item.reshape(-1, G*8)

        # 提取Y_data的第index组
        Y_data_item = np.reshape(Y_reshape[:,:,1,:,  G*group_index[item]: G*(group_index[item]+1)], (-1, G*4))
        Y_data_item = torch.Tensor(Y_data_item).to('cuda')
        

        # 信号检测
        SD_input = torch.cat((Y_data_item, Hf_hat_train_item), 1)
        SD_label = torch.Tensor(X_item).to('cuda')
        SD_output = SD_model[item](SD_input)

        loss_item = criterion(SD_output, SD_label)

        loss_item.backward()
        loss.append(loss_item)

        optimizer[item].step()
        scheduler[item].step()

    if iter % print_freq == 0:
        # print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        # print('Iter: {}\t' 'Loss {loss:.4f}\t'.format(iter, loss=loss.item()))
        print('Completed iterations [%d]\t' % iter, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for item in range(N_train_groups):
            print('Group[%d]:'%group_index[item],  'Loss {loss:.4f}\t'.format(loss=loss[item].item()))
        
    if iter % val_freq == 0:
        for idx in range(N_train_groups):
            SD_model[idx].eval()

        print('lr:%.4e'%optimizer[0].param_groups[0]['lr']) 
        Y_test,X_test,H_test = test_data


        Ns = Y_test.shape[0]
        # Obtain input data based on the LS channel estimation
        Hf_partial = LS_Estimation(Y_test, Pilotnum)
        # complex ---> real + imag
        Hf_input_test = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
        Hf_input_test[:, 0, :, :, :] = Hf_partial.real
        Hf_input_test[:, 1, :, :, :] = Hf_partial.imag

        # Obtain label data based on the perfect channel
        Hf = np.fft.fft(H_test, 256) / 20  # frequency-domain channel
        Hf = np.reshape(Hf, [Ns, 2, 2, 256], order='F')
        # complex ---> real +imag
        Hf_label_test = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
        Hf_label_test[:, 0, :, :, :] = Hf.real
        Hf_label_test[:, 1, :, :, :] = Hf.imag

        #  信道估计
        Hf_input_test = torch.Tensor(Hf_input_test).to('cuda')
        Hf_label_test = torch.Tensor(Hf_label_test).to('cuda')
        Hf_hat_test = CE_model(Hf_input_test)

        X_reshape = np.reshape(X_test, (-1, 2, 512))
        Y_reshape = np.reshape(Y_test, (-1, 2,2,2, 256), order = 'F') 

        for item in range(N_train_groups):
            #提取X的第index组

            X_temp = X_reshape[:,:,2*G*group_index[item]: 2*G*(group_index[item] + 1)]
            X_test_item = np.reshape(X_temp, (-1, G*4))

            # 提取H的第index组
            Hf_hat_test_item = Hf_hat_test[:,:,:,:, G*group_index[item]: G*(group_index[item]+1)]
            Hf_hat_test_item = Hf_hat_test_item.reshape(-1, G*8)

            Hf_label_test_item = Hf_label_test[:,:,:,:, G*group_index[item]: G*(group_index[item]+1)]
            Hf_label_test_item = Hf_label_test_item.reshape(-1, G*8)

            # 提取Y的第index组
            Y_data_test_item = np.reshape(Y_reshape[:,:,1,:,  G*group_index[item]: G*(group_index[item]+1)], (-1, G*4))
            Y_data_test_item = torch.Tensor(Y_data_test_item).to('cuda')

            SD_input = torch.cat((Y_data_test_item, Hf_hat_test_item), 1)
            SD_output = SD_model[item](SD_input)

            SD_label = torch.Tensor(X_test_item)
            X_hat_test_item = (SD_output > 0.5)*1. 
            error = X_hat_test_item.cpu() - SD_label 
            accuracy = np.array((torch.sum(torch.abs(error)<0.1)*1.)/(error.numel()*1.))

            
            if accuracy > best_accuracy[item]:
                # model save
                SD_modelSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Detection_Pilot_'+str(Pilotnum)+'_'+str(group_num)+'Group'+str(group_index[item])+'.pth.tar'
                # try:
                torch.save({'state_dict': SD_model[item].state_dict(), }, SD_modelSave, _use_new_zipfile_serialization=False)
                # except:
                #     torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
                best_accuracy[item] = accuracy
                print('Group[%d]'%group_index[item], 'accuracy: %.4f' % accuracy , 'Model Saved!!!')
            else:
                print('Group[%d]'%group_index[item], 'accuracy: %.4f' % accuracy)

        for item in range(N_train_groups):
            SD_model[item].train()
    if iter>iterations:
        break