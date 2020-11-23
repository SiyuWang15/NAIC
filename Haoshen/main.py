import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import struct
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import init
import torch.autograd as autograd
import torch.utils.data as Data
import time
import model
from model import Receiver
from functions import Regularization, NMSELoss, ConvBN, CRBlock
import channel_pre_estimation as cpe

# Read the training dataset and the testing dataset

# data1=open('X_1.bin','rb')
# X1=struct.unpack('f'*10000*1024, data1.read(4*10000*1024))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
train = 0
test = 1
num_nn = 8
num_pilot = 32
mode = '0'
modev = '0'
model_path = '../Modelsave'+mode+'_p'+str(num_pilot)+'/Rhy5_'
print('----------------------------------')
print('  is train:',train)
print('  is test:',test)
print('  num_pilot:',num_pilot)
print('  num_nn:',num_nn)
print('  training mode:',mode)
print('  validation mode:',modev)
print('  model save path:',model_path)
print('----------------------------------')
K = 256
P = num_pilot*2
pilotCarriers = np.arange(0, K, K // P)

## Load PilotValue
PilotValue_file_name = 'PilotValue_' + str(num_pilot) + '.mat'
D = loadmat(PilotValue_file_name)
PilotValue = D['P']  ## 1*Pilotnum*2
Pilot = np.zeros([K,2])
Pilot[pilotCarriers,0] = PilotValue.real
Pilot[pilotCarriers,1] = PilotValue.imag
Pilot = torch.from_numpy(Pilot).float().cuda()


# pilotCarriers1 = pilotCarriers[0:P:2]
# pilotCarriers2 = pilotCarriers[1:P:2]
# PilotValue1 = PilotValue[:,0:P:2] ## Pilot value for the first transmitting antenna
# PilotValue2 = PilotValue[:,1:P:2] ## Pilot value for the second transmitting antenna

def extract(v):
    return v.data.storage().tolist()

criterion = nn.BCELoss().cuda()
criterion_h = NMSELoss(reduction='mean')
criterion_test = NMSELoss(reduction='sum')

def Test(test_loader, R, num_test, best_ber):
    print('###############################################')
    # Pilot_batch = torch.reshape(Pilot, [1, 256, 2])
    # Pilot_batch = Pilot_batch.expand(num_test, 256, 2)
    XX_pred = []
    for i, (x_batch, y_batch, h_batch) in enumerate(test_loader):
        X_vali = x_batch.float().cuda()   # training data
        Y_vali = y_batch.float().cuda() 
        H_vali = h_batch.float().cuda()
                    
        batch_size = X_vali.shape[0]
        # H_LS_part = model.LS_Estimation(Y_vali, num_pilot)
        # H_inter = model.Interpolation_f(H_LS_part, num_pilot)
        H_inter = H_vali
        Yd = model.EXTRA_Yd(Y_vali, num_pilot)
        X_pred = []
        for i in range(num_nn):
            idx_y = model.Idx_nn(num_nn,i,'y')
            idx_p = model.Idx_nn(num_nn,i,'p')
            k,j = divmod(i,int(num_nn/2))
            pred = R[k](H_inter[:,:,:,:,idx_p], Yd[:,:,:,idx_p])
            X_pred.append(pred)
                    
        X_pred = torch.reshape(torch.stack(X_pred, dim=1),[batch_size,1024])
        X_pred[X_pred<0.5] = 0
        X_pred[X_pred>=0.5] = 1
        XX_pred.append(X_pred)
    XX_pred = torch.reshape(torch.stack(XX_pred, dim=0),[num_test,1024])
    ber_test = sum(sum(abs(XX_pred - X_vali)))/(num_test*1024)
            
    return ber_test

if train:
    print('###############################################')
    print("Loading training data ...")
    X = np.load("/data/HaoJiang_data/AI second part/training dataset/X_m"+mode+"_d12_p"+str(num_pilot)+".npy")
    Y = np.load("/data/HaoJiang_data/AI second part/training dataset/Y_m"+mode+"_d12_p"+str(num_pilot)+".npy")
    H_hat = np.load("/data/HaoJiang_data/AI second part/training dataset/H_hat_m"+mode+"_d12_p"+str(num_pilot)+"_1.npy")
    num_data = X.shape[0]
    L1 = random.sample(range(0, num_data), num_data)
    X = X[L1,:]
    Y = Y[L1,:]
    H_hat = H_hat[L1,:]
    radio = 1
    num_train = int(num_data*radio)
    X = torch.from_numpy(X[:num_train,:])
    Y = torch.from_numpy(Y[:num_train,:])
    #H_hat = cpe.channel_pre(Y, mode, num_pilot)
    H_hat = torch.from_numpy(H_hat[:num_train,:])
    
    Xv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Xv_m"+modev+"_d12_p"+str(num_pilot)+".npy")
    Yv = np.load("/data/HaoJiang_data/AI second part/validation dataset/Yv_m"+modev+"_d12_p"+str(num_pilot)+".npy")
    Hv_hat = np.load("/data/HaoJiang_data/AI second part/validation dataset/Hv_hat_m"+modev+"_d12_p"+str(num_pilot)+"_1.npy")
    num_test = Xv.shape[0]
    
    Xv = torch.from_numpy(Xv)
    Yv = torch.from_numpy(Yv)
    #Hv_hat = cpe.channel_pre(Yv, mode, num_pilot)
    Hv_hat = torch.from_numpy(Hv_hat)
    print("Num of train dataset is %s "%(num_train))
    print("Num of vaildation dataset is %s "%(num_test))
    print("Finished ...")
    print('###############################################')
    print('Training the model ...')
    
    batch_size = 2000
    epochs = 100
    r_learning_rate = 0.0001  
    optim_betas = (0.9, 0.999)
    print_interval = 1
    weight_gain = 1
    bias_gain = 0.1
    g_weight_gain = 0.1
    g_bias_gain = 0.1
    weight_decay=0.000000000000000000000000000000000000000000000000000000001
    nn = 2
    
    train_dataset = Data.TensorDataset(X,Y,H_hat)
    test_dataset = Data.TensorDataset(Xv,Yv,Hv_hat)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=2, pin_memory=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=int(num_test), shuffle=False, num_workers=2, pin_memory=True)
    
    cuda_gpu = torch.cuda.is_available()
    gpus = [0]
    #cuda_gpu = False
    R = model.ini_weights(nn, num_nn, cuda_gpu, gpus)
    model.Load_Model(R, nn, model_path)
    model.Save_Model(R, nn, model_path)
    
    r_optimizer=[]
    scheduler = []
    r_reg = [] 
    Error = []
    
    for i in range(nn):
        r_optimizer.append(optim.Adam(R[i].parameters(), lr=r_learning_rate, betas=optim_betas))
        scheduler.append(lr_scheduler.StepLR(r_optimizer[i], step_size = 20, gamma=0.25))
        r_reg.append(Regularization(R[i], weight_decay, p=2))
    
    
    best_ber = 100
    best_ber_train = 100
    # Pilot_batch = torch.reshape(Pilot, [1, 256, 2])
    # Pilot_batch = Pilot_batch.expand(batch_size, 256, 2)
    time_start=time.time()
    # ber_test = Test(test_loader, R, num_test, best_ber)
    # print(ber_test)
    R[0].train()
    R[1].train()
    for epoch in range(epochs):
        for step, (x_batch, y_batch, h_batch) in enumerate(train_loader):
            X_train = x_batch.float().cuda()   # training data
            Y_train = y_batch.float().cuda() 
            H_train = h_batch.float().cuda() 
            # H_LS_part = model.LS_Estimation(Y_train, num_pilot)
            # H_inter = model.Interpolation_f(H_LS_part, num_pilot) #Hfr = np.zeros([Ns, 2, 2, 2, 256])
            Yd = model.EXTRA_Yd(Y_train, num_pilot)               #Ydr = np.zeros([Ns, 2, 2, 256])
            e = 0
            X_pred = []
            for i in range(num_nn):
                idx_y = model.Idx_nn(num_nn,i,'y')
                idx_p = model.Idx_nn(num_nn,i,'p')
                k,j = divmod(i,int(num_nn/2))
                pred = R[k](H_train[:,:,:,:,idx_p], Yd[:,:,:,idx_p])
                X_pred.append(pred)
                error = criterion(pred, X_train[:,int(1024/num_nn)*i:int(1024/num_nn)*(i+1)])        
                R[k].zero_grad()
                r_optimizer[k].zero_grad() 
                error.backward()
                r_optimizer[k].step()  
                e +=error

            if step % 10 == 0:
                X_pred = torch.reshape(torch.stack(X_pred, dim=1),[batch_size,1024])
                X_pred[X_pred<0.5] = 0
                X_pred[X_pred>=0.5] = 1
                ber_train = sum(sum(abs(X_pred - X_train)))/(batch_size*1024)
                #H_pred = torch.reshape(torch.stack(H_pred, dim=4),[batch_size,2,2,2,512])
                #NMSE_pred = criterion_test(H_pred[:,:,:,:,0:256], H_train)
                if ber_train<best_ber_train:
                    model.Save_Model(R, nn, model_path)
                    best_ber_train = ber_train
                    
                print(" epoch: %s, step: %s, ber_train: %s)" %(epoch, step,  extract(ber_train)))

        Error.append(extract(error))
    #---------------------------------------------------------------------------------------------------------
        if epoch % print_interval == 0:
            R[0].eval()
            R[1].eval()
            with torch.no_grad():
                time_end=time.time()
                print(" epoch %s: error: %s, time: %s" %(epoch, error, time_end-time_start))
                X_pred = torch.reshape(torch.stack(X_pred, dim=1),[batch_size,1024])
                X_pred[X_pred<0.5] = 0
                X_pred[X_pred>=0.5] = 1
                ber_train = sum(sum(abs(X_pred - X_train)))/(batch_size*1024)
                #r_optimizer.param_groups[0]['lr'] /= 1.2
                #print(e, error0)
                
                ber_test = Test(test_loader, R, num_test, best_ber)
                if ber_test < best_ber:
                    model.Save_Model(R, nn, model_path)
                    best_ber = ber_test
                print(" epoch %s: ber_train: %s, ber_test: %s)" %(epoch, extract(ber_train), extract(ber_test)))
                time_start=time.time()
            


if test == 1:
    print('###############################################')
    print('Test the performance of the model training under the mode '+mode+' ...')
    print('Number of pilot:', num_pilot)
    if num_pilot == 32:
        Y_data = np.loadtxt('Y_1.csv',dtype=np.str,delimiter=",")
    elif num_pilot == 8:
        Y_data = np.loadtxt('Y_2.csv',dtype=np.str,delimiter=",")
    
    Y_data = Y_data.astype(np.float32)
    print('Y_data.shape:',Y_data.shape)
    print('Y_data.type',Y_data.dtype)
    
    # H_hat = cpe.channel_pre(Y_data, mode, num_pilot)
    H_hat = []
    for i in range(20):
        if num_pilot == 32:
            hat = cpe.channel_pre_32(Y_data[i*500:(i+1)*500,:], mode, num_pilot)
        elif num_pilot == 8:
            hat = cpe.channel_pre_8(Y_data[i*500:(i+1)*500,:], mode, num_pilot)
        
        print(i, hat.shape)
        H_hat.append(hat)
    H_hat = np.stack(H_hat,0)
    print('H_hat:',H_hat.shape)
    H_hat = H_hat.reshape([10000,2,2,2,256])
    print('H_hat:',H_hat.shape)
    
    gpus = [0]
    cuda_gpu = torch.cuda.is_available()
    #cuda_gpu = False
    nn = 2
    R = model.ini_weights(nn, num_nn, cuda_gpu, gpus)
    R = model.Load_Model(R, nn, model_path)
    R[0].eval()
    R[1].eval()
    X_data = []
    for i in range(20):
        X_batch = model.test(Y_data[i*500:(i+1)*500,:],H_hat[i*500:(i+1)*500,:], R, num_pilot, num_nn, model_path)
        X_batch = X_batch.detach().cpu().numpy()
        print(i,' X_batch:', X_batch.shape)
        X_data.append(X_batch)
    X_data = np.stack(X_data,0)   
    X_data = np.reshape(X_data,[10000,1024])
    X_0 = np.array(np.floor(X_data + 0.5), dtype=np.bool)
    print('x_data:',X_data.shape)
    print('X_0:',X_0.shape)
    
    if num_pilot == 32:
        X_0.tofile('X_pre_1.bin')
    elif num_pilot == 8:
        X_0.tofile('X_pre_2.bin')
    

# import numpy as np
# X_1_m0 = np.load("X_1_m0.npy")
# X_1_m1 = np.load("X_1_m1.npy")
# X_1_m2 = np.load("X_1_m2.npy")

# ber1 = sum(sum(abs(X_1_m0-X_1_m1)))/1024/10000
# ber2 = sum(sum(abs(X_1_m0-X_1_m2)))/1024/10000
# ber3 = sum(sum(abs(X_1_m1-X_1_m2)))/1024/10000
# print(ber1,ber2,ber3)
 