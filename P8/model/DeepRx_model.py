import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import struct
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from model.DeepRx_functions import Regularization, NMSELoss, ConvBN, CRBlock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import torch.autograd as autograd
import time


    
def ini_weights(nn, num_nn, cuda_gpu, gpus):
    R=[]
    for i in range(nn):
        R.append(Receiver(input_size=int(1024/num_nn*4), input_fre = int(1024/num_nn/2), output_size=int(1024/num_nn)))
    if(cuda_gpu):
        for i in range(nn):
            R[i] = torch.nn.DataParallel(R[i], device_ids=gpus).cuda()
    return R

class Receiver(nn.Module):
    def __init__(self, input_size, input_fre, output_size):
        super(Receiver, self).__init__()
        self.M = 12
        self.Nt = 2
        self.Nr = 2
        self.input_size = input_size
        self.input_fre = input_fre
        self.output_size = output_size
        self.H_decoder = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 32, 5)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()), 
            ("CRBlock3", CRBlock()), 
            ("conv1x1_bn", ConvBN(32, self.M*2, 1)),
            ]))
        self.Y_decoder = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 32, 5)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),  
            ("CRBlock3", CRBlock()), 
            ("conv1x1_bn", ConvBN(32, 2, 1)),          
            ]))
        self.Z_decoder = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 32, 5)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),  
            ("CRBlock3", CRBlock()), 
            ("conv1x1_bn", ConvBN(32, 2, 1)),          
            ]))
        
        
        self.fc1 = nn.Linear(3072, self.output_size)
        
        self.relu2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, H, Yd):
        H1 = H.reshape(H.size(0),2,self.Nt*self.Nr, self.input_fre)
        H1 = self.H_decoder(H1)
        H1 = H1.reshape(H.size(0),self.M*2, self.Nt*self.Nr, self.input_fre)
        H1 = H1.reshape(H.size(0), 2, self.M, self.Nr, self.Nt, self.input_fre)
        H1 = H1.permute(0,1,3,4,2,5).contiguous()
        H1 = H1.reshape(H.size(0),2, self.Nr, self.Nt*self.M, self.input_fre)
        
        Y1 = Yd.reshape(Yd.size(0),2, self.Nr, self.input_fre)
        Y1 = self.Y_decoder(Y1)
        Y1 = Y1.reshape(Yd.size(0),2, self.Nr,1, self.input_fre)
        
        Z = self.MRC(H1,Y1)
        Z = self.Z_decoder(Z)
        Z = torch.reshape(Z, [-1, Z.shape[1]*Z.shape[2]*Z.shape[3]])
        
        Z = self.fc1(Z)
        Z = torch.sigmoid(Z)
        return Z
    
    def MRC(self,H,Y):
        
        H = H.permute(0,4,1,3,2)
        Y = Y.permute(0,4,1,2,3)
        Hr = H[:,:,0,:,:]
        Hi = H[:,:,1,:,:]
        Yr = Y[:,:,0,:,:]
        Yi = Y[:,:,1,:,:]
        Hh = torch.cat([torch.cat([Hr,Hi],3), torch.cat([-Hi,Hr],3)],2)
        Yh = torch.cat([Yr,Yi],2)
        Z = torch.matmul(Hh,Yh)
        Z = Z.reshape(H.size(0),H.size(1),2,self.M*2)
        s = torch.norm(H,dim=(2,4))
        s = torch.pow(s,-2)
        s = s.reshape([H.size(0),H.size(1),1,self.M*2])
        Z = s*Z
        Z = Z.permute(0,2,3,1)
        return Z
        
        

def test(Y_test, H_hat, R, num_pilot, num_nn, model_path):
    nn = 2
    this_type_str = type(Y_test)
    if this_type_str is np.ndarray:
        print(this_type_str)
        Y_test = torch.from_numpy(Y_test).float().cuda()
    this_type_str = type(H_hat)
    if this_type_str is np.ndarray:
        print(this_type_str)
        H_hat = torch.from_numpy(H_hat).float().cuda()
        
    num_test = Y_test.shape[0]

    
    # H_LS_part = LS_Estimation(Y_test, num_pilot)
    # H_inter = Interpolation_f(H_LS_part, num_pilot)
    Yd = EXTRA_Yd(Y_test, num_pilot)
    
    # Pilot_batch = torch.reshape(Pilot, [1, 256, 2])
    # Pilot_batch = Pilot_batch.expand(num_test, 256, 2)
    # if(cuda_gpu):
    #     Y_test = Y_test.cuda()
    X_pred = []
    for i in range(num_nn):
        idx_y = Idx_nn(num_nn,i,'y')
        idx_p = Idx_nn(num_nn,i,'p')
        k,j = divmod(i,int(num_nn/2))
        pred = R[k](H_hat[:,:,:,:,idx_p], Yd[:,:,:,idx_p])
        X_pred.append(pred)
                    
    X_pred = torch.reshape(torch.stack(X_pred, dim=1),[num_test,1024])
    X_pred[X_pred<0.5] = 0
    X_pred[X_pred>=0.5] = 1
    #ber_test = sum(sum(abs(X_pred - X_test)))/(num_test*1024)
    return X_pred


def EXTRA_Yd(Y, Pilotnum):
    this_type_str = type(Y)
    if this_type_str is not np.ndarray:
        Y = Y.detach().cpu().numpy()
    Ns = Y.shape[0]
    Y = np.reshape(Y, [-1, 2, 2, 2, 256], order='F')
    Y_complex = Y[:,0,:,:,:]+1j * Y[:,1,:,:,:]
    Yd = Y_complex[:,1,:,:] # Received data signal
    
    Ydr = np.zeros([Ns, 2, 2, 256])
    Ydr[:,0,:,:] = Yd.real
    Ydr[:,1,:,:] = Yd.imag
    Ydr = torch.from_numpy(Ydr).float().cuda()
    return Ydr

def LS_Estimation(Y, Pilotnum):
    this_type_str = type(Y)
    if this_type_str is not np.ndarray:
        Y = Y.detach().cpu().numpy()
    K = 256
    P = Pilotnum*2
    Ns = Y.shape[0]

    Y = np.reshape(Y, [-1, 2, 2, 2, 256], order='F')
    Y_complex = Y[:,0,:,:,:]+1j * Y[:,1,:,:,:]
    # print(Y_complex)
    Yp = Y_complex[:,0,:,:] # Received pilot signal
    # Yp = np.reshape(Yp, [Ns, 2, 256])
    # print(Yp)
    Yp1 = Yp[:,0,:] # Received pilot signal for the first receiving antenna
    Yp2 = Yp[:,1,:] # Received pilot signal for the second receiving antenna

    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]

    ## Load PilotValue
    PilotValue_file_name = 'PilotValue_' + str(Pilotnum) + '.mat'
    D = loadmat(PilotValue_file_name)
    PilotValue = D['P']  ## 1*Pilotnum*2
    PilotValue1 = PilotValue[:,0:P:2] ## Pilot value for the first transmitting antenna
    PilotValue2 = PilotValue[:,1:P:2] ## Pilot value for the second transmitting antenna

    ## LS estimation
    Hf = np.zeros(shape=(Ns, 2, 2, 256),dtype=np.complex64)

    Hf[:, 0, 0, pilotCarriers1] = Yp1[:, pilotCarriers1]/PilotValue1
    Hf[:, 0, 1, pilotCarriers2] = Yp1[:, pilotCarriers2]/PilotValue2
    Hf[:, 1, 0, pilotCarriers1] = Yp2[:, pilotCarriers1]/PilotValue1
    Hf[:, 1, 1, pilotCarriers2] = Yp2[:, pilotCarriers2]/PilotValue2
    
    return Hf

def Interpolation_f(Hf, Pilotnum):
    Ns = Hf.shape[0]
    Hf_inter = np.zeros((Ns, 2, 2, 256), dtype=np.complex64)
    if Pilotnum == 32:
        i ,j = np.meshgrid(np.arange(256), np.arange(256))
        omega = np.exp(-2*np.pi*1j/256)
        DFT_Matrix = np.power(omega, i*j)

        inP1 = int(256 / Pilotnum)
        inP2 = int(128 / Pilotnum)
        idx1 = np.arange(0,256,inP1)
        idx2 = np.arange(inP2,256,inP1)
        DFT_1 = DFT_Matrix[idx1,0:32]
        DFT_1v = np.linalg.inv(DFT_1)
        DFT_2 = DFT_Matrix[idx2,0:32]
        DFT_2v = np.linalg.inv(DFT_2)
        hf_00 = Hf[:, 0, 0, idx1].T
        hf_01 = Hf[:, 0, 1, idx2].T
        hf_10 = Hf[:, 1, 0, idx1].T
        hf_11 = Hf[:, 1, 1, idx2].T
        ht_00 = np.dot(DFT_1v, hf_00).T
        ht_01 = np.dot(DFT_2v, hf_01).T
        ht_10 = np.dot(DFT_1v, hf_10).T
        ht_11 = np.dot(DFT_2v, hf_11).T
        Hf_inter[:, 0, 0, :] = np.fft.fft(ht_00,256)
        Hf_inter[:, 0, 1, :] = np.fft.fft(ht_01,256)
        Hf_inter[:, 1, 0, :] = np.fft.fft(ht_10,256)
        Hf_inter[:, 1, 1, :] = np.fft.fft(ht_11,256)
    elif Pilotnum == 8:
        i ,j = np.meshgrid(np.arange(256), np.arange(256))
        omega = np.exp(-2*np.pi*1j/256)
        DFT_Matrix = np.power(omega, i*j)

        inP1 = int(256 / Pilotnum)
        inP2 = int(128 / Pilotnum)
        idx1 = np.arange(0,256,inP1)
        idx2 = np.arange(inP2,256,inP1)
        idx1_inter = np.arange(0, 256, int(inP1/4))
        idx2_inter = np.arange(int(inP2/4), 256, int(inP1/4))
        DFT_1 = DFT_Matrix[idx1_inter,0:32]
        DFT_1v = np.linalg.inv(DFT_1)
        DFT_2 = DFT_Matrix[idx2_inter,0:32]
        DFT_2v = np.linalg.inv(DFT_2)
        hf_00 = Hf[:, 0, 0, idx1]
        hf_01 = Hf[:, 0, 1, idx2]
        hf_10 = Hf[:, 1, 0, idx1]
        hf_11 = Hf[:, 1, 1, idx2]
        hf_00_inter = hf_01_inter = hf_10_inter = hf_11_inter = np.zeros((Ns, 32),dtype=np.complex64)
        for sample in range(0, Ns):
            hf_00_inter[sample, :] = np.interp(idx1_inter,  idx1, hf_00[sample, :])
            hf_01_inter[sample, :] = np.interp(idx2_inter,  idx2, hf_01[sample, :])
            hf_10_inter[sample, :] = np.interp(idx1_inter,  idx1, hf_10[sample, :])
            hf_11_inter[sample, :] = np.interp(idx2_inter,  idx2, hf_11[sample, :])
        ht_00_inter = np.dot(DFT_1v, hf_00_inter.T).T
        ht_01_inter = np.dot(DFT_2v, hf_01_inter.T).T
        ht_10_inter = np.dot(DFT_1v, hf_10_inter.T).T
        ht_11_inter = np.dot(DFT_2v, hf_11_inter.T).T
        Hf_inter[:, 0, 0, :] = np.fft.fft(ht_00_inter,256)
        Hf_inter[:, 0, 1, :] = np.fft.fft(ht_01_inter,256)
        Hf_inter[:, 1, 0, :] = np.fft.fft(ht_10_inter,256)
        Hf_inter[:, 1, 1, :] = np.fft.fft(ht_11_inter,256)
    else:
        print('error!')
    
    Hfr = np.zeros([Ns, 2, 2, 2, 256])
    Hfr[:,0,:,:,:] = Hf_inter.real
    Hfr[:,1,:,:,:] = Hf_inter.imag
    Hfr = torch.from_numpy(Hfr).float().cuda()
    return Hfr

def Idx_nn(num_nn,i,s):
    k,i = divmod(i,int(num_nn/2))
    output_nn = int(1024/num_nn) #64 16
    num_fre = int(output_nn/2)   #32 8
    eac_fre = 8
    inte_pliot = int(256/32)
    if s == 'y':
        idx_y = np.arange(i*num_fre*eac_fre,(i+1)*num_fre*eac_fre,1) #256
        return idx_y
    elif s =='h':
        idx_h = np.arange(i*num_fre,(i+1)*num_fre,1)
        return idx_h
    elif s == 'p':
        idx_p = np.arange(i*num_fre,(i+1)*num_fre,1)
        return idx_p
    # idx_d1_r = np.arange(i*num_fre*eac_fre+2,(i+1)*num_fre*eac_fre,eac_fre) 
    # idx_d1_i = np.arange(i*num_fre*eac_fre+3,(i+1)*num_fre*eac_fre,eac_fre)
    # idx_d2_r = np.arange(i*num_fre*eac_fre+6,(i+1)*num_fre*eac_fre,eac_fre) 
    # idx_d2_i = np.arange(i*num_fre*eac_fre+7,(i+1)*num_fre*eac_fre,eac_fre)
    # idx_y = np.concatenate([idx_d1_r,idx_d1_i,idx_d2_r,idx_d2_i],0)
    
    
    # idx_p1_r = np.arange(i*num_fre*eac_fre+0,(i+1)*num_fre*eac_fre,eac_fre*inte_pliot) 
    # idx_p1_i = np.arange(i*num_fre*eac_fre+1,(i+1)*num_fre*eac_fre,eac_fre*inte_pliot) 
    # idx_p1 = np.concatenate([idx_p1_r,idx_p1_i],0)
    
    # idx_p2_r = np.arange(i*num_fre*eac_fre+4,(i+1)*num_fre*eac_fre,eac_fre*inte_pliot) 
    # idx_p2_i = np.arange(i*num_fre*eac_fre+5,(i+1)*num_fre*eac_fre,eac_fre*inte_pliot) 
    # idx_p2 = np.concatenate([idx_p2_r,idx_p2_i],0)
        
    # idx_y = np.concatenate([idx_y,idx_p1,idx_p2],0)
    # idx_p1r = np.arange(0,2048,eac_fre)
    # idx_p1i = np.arange(1,2048,eac_fre)
    # idx_p2r = np.arange(4,2048,eac_fre)
    # idx_p2i = np.arange(5,2048,eac_fre)
    # idx_y = np.concatenate([idx_y,idx_p1r,idx_p1i,idx_p2r,idx_p2i],0)
    # idx_y = np.unique(idx_y)
    
def Load_Model(R, nn, model_path):
    #model_path = './Modelsave/R1.pth.tar'
    #model.load_state_dict(torch.load(model_path)['state_dict'])
    for i in range(nn):
        R[i].module.load_state_dict(torch.load(model_path+"%d.pkl"%(i)))
    return R

def Save_Model(R, nn, model_path):
    #modelSave = './Modelsave/R1.pth.tar'
    for i in range(nn):
        torch.save(R[i].module.state_dict(), model_path+"%d.pkl"%(i))
    print(' Mode save to '+model_path+' !')   

# class Receiver(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Receiver, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.fc1 = nn.Linear(int(input_size*10/8), 4096)
#         self.relu1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
#         self.fc2 = nn.Linear(4096, output_size)
        
#         decoder = OrderedDict([
#             ("conv5x5_bn", ConvBN(4, 32, 5)),
#             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("CRBlock1", CRBlock()),
#             ("CRBlock2", CRBlock()),
#         ])
#         self.conv1x1_bn = ConvBN(32, 4, 1)
#         self.relu2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
#         self.decoder_feature = nn.Sequential(decoder)
#         self.sig = nn.Sigmoid()

#     def forward(self, x, p):
#         x = x.view(-1, int(self.input_size/8), 8)
#         out = torch.cat((x, p), 2)
#         out = out.view(-1,int(self.input_size*10/8))
        
        
#         out = self.relu1(self.fc1(out))
#         out = out.view(-1, 4, 32, 32)
        
#         out = self.decoder_feature(out)
#         out = self.conv1x1_bn(out)
#         out = self.relu2(out)
        
#         out = torch.reshape(out, [-1, out.shape[1]*out.shape[2]*out.shape[3]])
#         out = self.fc2(out)
#         out = self.sig(out)
#         return out

def extract(v):
    return v.data.storage().tolist()

# class Receiver(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Receiver, self).__init__()
#         #128->256->128->64->output_size=32  NN=32
#         #256->512->256->128->output_size=64  NN=16
#         #512->1024->512->256->output_size=128 NN=8
#         self.map1 = nn.Linear(256, 1024) #1152
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.map2 = nn.Linear(1024, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.map3 = nn.Linear(512, 256)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.map4 = nn.Linear(256, output_size)
#         # self.bn4 = nn.BatchNorm1d(128)
#         # self.map5 = nn.Linear(128, output_size)
# #        torch.nn.init.xavier_normal_(self.map1.weight, weight_gain)
# #        torch.nn.init.constant(self.map1.bias, bias_gain)
# #        torch.nn.init.xavier_normal_(self.map2.weight, weight_gain)
# #        torch.nn.init.constant(self.map2.bias, bias_gain)
        
#     def forward(self, x):
#         x = F.relu(self.map1(x))
#         x = self.bn1(x)
#         x = F.relu(self.map2(x))
#         x = self.bn2(x)
#         x = F.relu(self.map3(x))
#         x = self.bn3(x)
#         return torch.sigmoid(self.map4(x))  
