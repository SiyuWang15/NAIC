import torch
import torch.nn as nn
import numpy as np 
import os 
import logging
import sys
import time
sys.path.append('..')
from utils import *

from Estimators import ModeEstimator, RouteEstimator
from data import get_test_data, get_val_data


r'''This runner is only used to DO the whole test procedure, not used to train model'''

class FullRunner():
    def __init__(self, config):
        self.config = config
        self.Pn = self.config.RE.Pilotnum
    
    def get_model(self, device):
        mode_estimator = ModeEstimator()
        load_path = os.path.join(self.config.ckpt.medir, f'best_ckpt_P{self.Pn}.pth')
        mode_estimator.load_state_dict(torch.load(load_path))
        mode_estimator.to(device)
        mode_estimator.eval()
        route_estimators = []
        for mode in [0, 1, 2]:
            model = RouteEstimator(self.config.RE.in_dim, self.config.RE.out_dim, \
                    self.config.RE.h_dims, self.config.RE.act)
            load_path = os.path.join(self.config.ckpt.redir, f'best_mode_{mode}_Pn_{self.Pn}.pth')
            model.load_state_dict(torch.load(load_path))
            model.to(device)
            model.eval()
            route_estimators.append(model)
        return mode_estimator, route_estimators

    def run(self):
        device = 'cuda'
        mode_estimator, route_estimators = self.get_model(device)
        Yp, Yd = get_test_data(self.Pn)
    
    def validation(self):
        device = 'cuda'
        mode_estimator, route_estimators = self.get_model(device)
        Yp, Yd, X_label, H_label = get_val_data(self.Pn)
        Yp = torch.Tensor(Yp).to(device)
        predX = self.estimateX(Yp, Yd, mode_estimator, route_estimators)
        print(predX.shape)
        acc = (predX == X_label).astype('float32').mean()
        logging.info(acc)

    def estimateX(self, Yp, Yd, mode_estimator, route_estimators): # for a single data item
        Yp_modes = [[],[],[]]
        Yd_modes = [[],[],[]]
        X_modes = dict()
        predX = []
        cnt = [0, 0, 0]
        with torch.no_grad():
            mode = mode_estimator(Yp)
            mode = torch.argmax(mode, dim = -1)
            for i, m in enumerate(mode):
                Yd_modes[m].append(Yd[i])
                Yp_modes[m].append(Yp[i])
            for mode in range(3):
                tmp_Yd = np.stack(Yd_modes[mode], axis=0)
                tmp_Yp = torch.stack(Yp_modes[mode]).to(Yp.device)
                H_est = route_estimators[mode](tmp_Yp)
                H_est = np.asarray(H_est.cpu()) # Nsx256
                Hf_est = transfer_H(H_est)
                tmp_Yd = transfer_Y(tmp_Yd)
                X_ML, X_bits = self.MLReceiver(tmp_Yd, Hf_est)
                X_modes[mode] = X_bits
            for i in range(len(Pd)):
                thismode = mode[i]
                predX.append(X_modes[thismode][cnt[thismode]])
            predX = np.asarray(predX)
        return predX

    def MLReceiver(self, Y, H):
        Codebook = self.MakeCodebook(4)
        G, P = Codebook.shape
        assert P == 2**G

        B = G//4
        assert B*4 == G

        T = 256//B
        assert T*B == 256

        batch = H.shape[0]

        # B为分组的个数，只能为2的整倍数
        Y = np.reshape(Y , (batch, 2,  B, T))
        H = np.reshape(H , (batch, 2,  2, B, T))
        X_ML = np.zeros((batch, 2, B, T) , dtype = complex)
        X_bits = np.zeros((batch, 2, 2, B, T))

        for num in range(batch):
            if num % 100 == 0:
                print('Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for idx in range(T):
                y = Y[num, :, :, idx]
                h = H[num, :, :, :, idx]
                error = np.zeros((P, 1)) 

                for item in range(P):

                    x = np.reshape(Codebook[:, item], (2,2,B))
                    x = 0.7071 * ( 2 * x[:,0,:] - 1 ) + 0.7071j * ( 2 * x[:,1,:] - 1 )
                    for b in range(B):
                        error[item] = error[item] + np.linalg.norm( y[:,b:b+1] - np.dot(h[:,:,b], x[:,b:b+1]) )**2

                ML_idx = np.argmin(error)

                x = np.reshape(Codebook[:, ML_idx], (2,2,B))
                x_ML = 0.7071 * ( 2 * x[:,0,:] - 1 ) + 0.7071j * ( 2 * x[:,1,:] - 1 )
                X_ML[num,:,:,idx] = x_ML

        X_ML = np.reshape(X_ML, [batch, 2 , 256])
        # X_bits = np.reshape(X_bits, [batch, 2, 512])
        # Batch * TX * subcarrier *  IQ
        X_bits = np.reshape(X_ML, [batch, 2 , 256, 1])
        X_bits = np.concatenate([np.real(X_bits)>0,np.imag(X_bits)>0], 3)*1
        X_bits = np.reshape(X_bits, [batch, 1024])
        return X_ML, X_bits

    def MakeCodebook(self, G = 4):
        assert type(G) == int

        codebook = np.zeros((G, 2**G))
        for idx in range(2**G):
            n = idx
            for i in range(G):
                r = n % 2
                codebook[G -1- i, idx] = r
                n = n//2

        return codebook