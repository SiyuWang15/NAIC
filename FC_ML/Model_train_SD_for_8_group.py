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

parser = argparse.ArgumentParser()
parser.add_argument('--group_index', type = int, default = 0, help = 'input group index')
parser.add_argument('--group_num', type = int, default = 32, help = 'input group num')
parser.add_argument('--load_SD', type = bool, default = False)
parser.add_argument('--gpu_list',  type = str,  default='4,5,6,7', help='input gpu list')
args = parser.parse_args()
group_index = args.group_index
group_num = args.group_num
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
CE_model = FC_Estimation(2048, 4096, 4096, 2048)
# 分成16组
G = 256 // group_num
SD_model = FC_Detection(G*(4+8),  G*4, [512,512,1024,1024,512,512])
# model = DnCNN()
# criterion = NMSELoss(reduction='mean')
# criterion_test = NMSELoss(reduction='sum')
criterion =  torch.nn.BCELoss(weight=None, reduction='mean')


if len(gpu_list.split(',')) > 1:
    CE_model = torch.nn.DataParallel(CE_model).cuda() # model.module
    SD_model = torch.nn.DataParallel(SD_model).cuda() # model.module
else:
    CE_model = CE_model.cuda()
    SD_model = SD_model.cuda()


if load_CE:
    CE_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Estimation_for_'+str(Pilotnum)+'.pth.tar'
    CE_model.load_state_dict(torch.load(CE_model_path)['state_dict'])
    print("CE Weight Loaded!")

if freeze_CE:
    for params in CE_model.parameters():
        params.requires_grad = False
    CE_model.eval()
    print('freeze CE!')

if load_SD:
    SD_model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Detection_for_'+str(Pilotnum)+'_'+str(group_num)+'Group'+'0'+'.pth.tar'
    SD_load_model = torch.load(SD_model_path)['state_dict']
    SD_model_dict = SD_model.state_dict()

    SD_state_dict = {k:v for k,v in SD_load_model.items() if (k in SD_model_dict.keys() and 'output' not in k)}

    SD_model_dict.update(SD_state_dict)
    SD_model.load_state_dict( SD_model_dict )
    print("SD Weight Loaded!")

optimizer = torch.optim.Adam(SD_model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size = 2000, gamma=0.5)

# generate data fro training
train_data = generator(batch_size,H_tra, Pilotnum)
test_data = generatorXY(2000, H_val, Pilotnum)

print('==========Begin Training=========')
iter = 0
SD_model.train()
for Y,X,H in train_data:
    iter = iter+1
    # 提取X的第index组
    X_temp = np.reshape(X,  (-1, 2,512) )
    X_temp = X_temp[:,:, 2*G*group_index: 2*G*(group_index + 1)]
    X = np.reshape(X_temp, (-1, G*4))
# for idx in range(iterations):
    # Y,X,H = train_data

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

    optimizer.zero_grad()

    # 信道估计
    Hf_input_train = torch.Tensor(Hf_input_train).to('cuda')
    Hf_label_train = torch.Tensor(Hf_label_train).to('cuda')
    Hf_hat_train = CE_model(Hf_input_train)

    # 提取H的第index组
    Hf_hat_train = Hf_hat_train[:,:,:,:, G*group_index: G*(group_index+1)]
    Hf_hat_train = Hf_hat_train.reshape(-1, G*8)

    Hf_label_train = Hf_label_train[:,:,:,:, G*group_index: G*(group_index+1)]
    Hf_label_train = Hf_label_train.reshape(-1, G*8)

    # 提取Y_data的第index组
    temp = np.reshape(Y, (-1, 2,2,2, 256), order = 'F') 
    Y_data = np.reshape(temp[:,:,1,:,  G*group_index: G*(group_index+1)], (-1, G*4))
    Y_data = torch.Tensor(Y_data).to('cuda')
    

    # 信号检测
    SD_input = torch.cat((Y_data, Hf_label_train), 1)
    SD_label = torch.Tensor(X).to('cuda')
    SD_output = SD_model(SD_input)

    loss = criterion(SD_output, SD_label)

    loss.backward()
    optimizer.step()

    scheduler.step()

    if iter % print_freq == 0:
        # print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        # print('Iter: {}\t' 'Loss {loss:.4f}\t'.format(iter, loss=loss.item()))
        print('Group[%d]:'%group_index, 'Completed iterations [%d]\t' % iter, 'Loss {loss:.4f}\t'.format(loss=loss.item()),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
    if iter % val_freq == 0:
        SD_model.eval()
        print('lr:%.4e'%optimizer.param_groups[0]['lr']) 
        Y_test,X_test,H_test = test_data

        #提取X的第index组
        X_temp = np.reshape(X_test, (-1, 2, 512))
        X_temp = X_temp[:,:,2*G*group_index: 2*G*(group_index + 1)]
        X_test = np.reshape(X_temp, (-1, G*4))


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

        # 提取H的第index组
        Hf_hat_test = Hf_hat_test[:,:,:,:, G*group_index: G*(group_index+1)]
        Hf_hat_test = Hf_hat_test.reshape(-1, G*8)

        Hf_label_test = Hf_label_test[:,:,:,:, G*group_index: G*(group_index+1)]
        Hf_label_test = Hf_label_test.reshape(-1, G*8)

        # 提取Y的第index组
        temp = np.reshape(Y_test, (-1, 2,2,2, 256), order = 'F') 
        Y_data_test = np.reshape(temp[:,:,1,:,  G*group_index: G*(group_index+1)], (-1, G*4))
        Y_data_test = torch.Tensor(Y_data_test).to('cuda')

        SD_input = torch.cat((Y_data_test, Hf_hat_test), 1)
        SD_output = SD_model(SD_input)

        SD_label = torch.Tensor(X_test)
        X_hat_test = (SD_output > 0.5)*1. 
        error = X_hat_test.cpu() - SD_label 
        accuracy = (torch.sum(torch.abs(error)<0.1)*1.)/(error.numel()*1.)

        print('accuracy %.4f' % accuracy)
        if accuracy > best_accuracy:
            # model save
            SD_modelSave =  '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/FC_Detection_for_'+str(Pilotnum)+'_'+str(group_num)+'Group'+str(group_index)+'.pth.tar'
            # try:
            torch.save({'state_dict': SD_model.state_dict(), }, SD_modelSave, _use_new_zipfile_serialization=False)
            # except:
            #     torch.save({'state_dict': model.module.state_dict(), }, modelSave,_use_new_zipfile_serialization=False)
            print('Model saved!')
            best_accuracy = accuracy

        SD_model.train()
    if iter>iterations:
        break