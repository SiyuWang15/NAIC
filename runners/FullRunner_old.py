import torch
import torch.nn as nn
import numpy as np 
import os 
import logging
import math
import multiprocessing as mp
import sys
import time
sys.path.append('..')
from utils import *

from Estimators import ModeEstimator, RouteEstimator, CNNRouteEstimator, ResNetRouteEstimator
from data import get_test_data, get_val_data, get_simple_val_data


r'''This runner is only used to DO the whole test procedure, not used to train model'''

class FullRunner():
    def __init__(self, config):
        self.config = config
        self.Pn = self.config.RE.Pilotnum
    
    def get_model(self, device):
        mode_estimator = ModeEstimator(out_dim=self.config.ME.out_dim)
        load_path = os.path.join(self.config.ckpt.medir, f'P{self.Pn}.pth')
        mode_estimator.load_state_dict(torch.load(load_path))
        mode_estimator.to(device)
        mode_estimator.eval()
        route_estimators = []
        for mode in [0, 1, 2]:
            if self.config.RE.model == 'CNN':
                model = CNNRouteEstimator(self.config.RE.in_nc,  out_dim = self.config.RE.out_dim)
            else:
                model = RouteEstimator(self.config.RE.in_dim, self.config.RE.out_dim, \
                    self.config.RE.h_dims, self.config.RE.act)
            load_path = os.path.join(self.config.ckpt.redir, f'mode_{mode}_Pn_{self.Pn}.pth')
            model.load_state_dict(torch.load(load_path))
            model.to(device)
            model.eval()
            route_estimators.append(model)
        return mode_estimator, route_estimators

    def run(self):
        if self.config.log.run_mode == 'validation':
            self.validation()
        elif self.config.log.run_mode == 'testing':
            self.test()
    
    def simple_run(self):
        RE_dir = '/data/siyu/NAIC/workspace/CNNY2HEstimator/1114-23-32-23/mode_0_Pn_32/checkpoints/best.pth'
        # RE_dir = '/data/siyu/NAIC/workspace/CNNY2HEstimator/1112-12-59-18/mode_0_Pn_32/checkpoints/best.pth'
        route_estimator = ResNetRouteEstimator(34)
        # route_estimator = CNNRouteEstimator(self.config.RE.in_nc, out_dim=self.config.RE.out_dim)
        route_estimator.load_state_dict(torch.load(RE_dir))
        Yp, Yd, X_label, H_label = get_simple_val_data(0, 32)
        Yp = torch.Tensor(Yp)
        H_pred = route_estimator(Yp).detach().numpy()
        nmse = self.NMSE(H_pred, H_label)
        Yd = transfer_Y(Yd)
        Hf = transfer_H(H_pred)
        _, pred_X = self.MLReceiver(Yd, Hf)
        acc = (pred_X == X_label).astype('float32').mean()
        logging.info(f'acc: {acc}, nmse: {nmse}')
    
    def simple_test(self):
        RE_dir = '/data/siyu/NAIC/workspace/CNNY2HEstimator/1113-16-57-39/mode_0_Pn_8/checkpoints/best.pth'
        route_estimator = ResNetRouteEstimator(34)
        route_estimator.load_state_dict(torch.load(RE_dir))
        Yp, Yd = get_test_data(8)
        Yp = np.reshape(Yp, [len(Yp), 2, 16, 32], order = 'F')
        Yp = torch.Tensor(Yp)
        predH = route_estimator(Yp).detach().numpy()
        predX = self.estimateX(Yd, predH)
        predX = np.array(np.floor(predX+0.5), dtype = np.bool)
        dp = os.path.join(self.config.log.log_dir, 'X_pre_2.bin')
        predX.tofile(dp)


    def test(self):
        device = 'cuda'
        mode_estimator, route_estimators = self.get_model(device)
        Yp, Yd = get_test_data(self.Pn)
        Yp = torch.Tensor(Yp).to(device)
        predH = self.estimateH(Yp, mode_estimator, route_estimators)
        predX = self.estimateX(Yd, predH)
        predX = np.array(np.floor(predX+0.5), dtype = np.bool)
        if self.Pn == 32:
            dp = os.path.join(self.config.log.log_dir, 'X_pre_1.bin')
        elif self.Pn == 8:
            dp = os.path.join(self.config.log.log_dir, 'X_pre_2.bin')
        predX.tofile(dp)
        logging.info(f'results save in {dp}')

    def validation(self):
        device = 'cuda'
        description = r'SNRdb:9~11, mode ratio consistent with Y_1 and Y_2'
        logging.info(description)
        mode_estimator, route_estimators = self.get_model(device)
        Yp, Yd, X_label, H_label = get_val_data(self.Pn) # H_label: Nsx256
        Yp4RE = np.reshape(Yp, [len(Yp), 2, 16, 32], order = 'F')
        Yp4RE = torch.Tensor(Yp4RE).to(device)
        Yp4ME = torch.Tensor(Yp.copy()).to(device)
        predH = self.estimateH(Yp4ME, Yp4RE, mode_estimator, route_estimators) # Nsx256
        logging.info(f'nmse of predicted H: {self.NMSE(predH, H_label)}')
        predX = self.estimateX(Yd, predH)
        acc = (predX == X_label).astype('float32').mean()
        logging.info(acc)

    def NMSE(self, H_pred, H_label):
        mse = np.power(H_pred-H_label, 2).sum()
        norm = np.power(H_label, 2).sum()
        return mse / norm
    
    def estimateH(self, Yp4ME, Yp4RE, mode_estimator, route_estimators):
        Yp_modes = [[],[],[]]
        H_modes = [[],[],[]]
        predH = []
        cnt = [0,0,0]
        with torch.no_grad():
            modes = mode_estimator(Yp4ME)
            modes = torch.argmax(modes, dim = -1)
            modes = np.asarray(modes.cpu())
            for i, mode in enumerate(modes):
                # if mode == 1: 
                #     mode = 0
                #     modes[i] = 0
                Yp_modes[mode].append(Yp4RE[i])
            for mode in [0,1,2]:
                logging.info(f'number of Y in mode {mode}: {len(Yp_modes[mode])}')
            for mode in [0, 1, 2]:
                if len(Yp_modes[mode]) == 0:
                    continue
                tmp_Yp = torch.stack(Yp_modes[mode]).to(Yp4RE.device)
                H_est = route_estimators[mode](tmp_Yp)
                H_est = np.asarray(H_est.cpu()) # Nsx256
                H_modes[mode] = H_est
            
            for i in range(len(Yp4RE)):
                mode = modes[i]
                predH.append(H_modes[mode][cnt[mode]])
                cnt[mode] += 1
            predH = np.asarray(predH)
        return predH

    def estimateX(self, Yd, H_est):
        # Yd: Nsx1024
        # H_est: Nsx256
        Yd = transfer_Y(Yd)
        Hf_est = transfer_H(H_est)
        _, predX = self.MLReceiver(Yd, Hf_est)
        return predX
    
    def MLReceiver(self, Y, H, num_workers = 16):
        Codebook = self.MakeCodebook(4)
        G, P = Codebook.shape
        assert P == 2**G

        B = G//4
        assert B*4 == G

        T = 256//B
        assert T*B == 256
        N_s = H.shape[0]
        # 创建多进程
        q = mp.Manager().dict()
        Processes = []
        batch = math.floor(N_s * 1. / num_workers)

        for i in range(num_workers):
            if i < num_workers - 1:
                Y_single = Y[  i*batch : (i+1)*batch, ...]
                H_single = H[  i*batch : (i+1)*batch, ...]
                P = mp.Process( target = self.MLReceiver_single_process, args = (q, i, Y_single, H_single,Codebook))
                # P.start()
                Processes.append(P)
            else:
                Y_single = Y[  i*batch :, ...]
                H_single = H[  i*batch :, ...]
                P = mp.Process( target = self.MLReceiver_single_process, args = (q, i, Y_single, H_single,Codebook))
                # P.start()
                Processes.append(P)


        for i in range(num_workers):
            Processes[i].start()

        for i in range(num_workers):
            Processes[i].join()


        X_ML = np.zeros((N_s, 2, 256) , dtype = complex)
        X_bits = np.zeros((N_s, 1024))

        for label,v in q.items():
            ml, bits = v
            if label < num_workers - 1:
                X_ML[  label*batch : (label+1)*batch, :, :] = ml
                X_bits[  label*batch : (label+1)*batch, :] = bits
            else:
                X_ML[  label*batch: , :, :] = ml
                X_bits[  label*batch: , :] = bits

        return X_ML, X_bits
    
    def MLReceiver_single_process(self, q, label, Y, H, Codebook):
        G, P = Codebook.shape
        B = G//4
        T = 256//B
        batch = H.shape[0]

        # B为分组的个数，只能为2的整倍数
        Y = np.reshape(Y , (batch, 2,  B, T))
        H = np.reshape(H , (batch, 2,  2, B, T))
        X_ML = np.zeros((batch, 2, B, T) , dtype = complex)
        X_bits = np.zeros((batch, 2, 2, B, T))

        for num in range(batch):
            if num % 100 == 0:
                print('P{0}'.format(label),': Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

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
        X_bits = np.reshape(X_ML, [batch, 2 , 256, 1])
        X_bits = np.concatenate([np.real(X_bits)>0,np.imag(X_bits)>0], 3)*1
        X_bits = np.reshape(X_bits, [batch, 1024])
        q[label] = (X_ML, X_bits)
        # print('P{0} completed!'.format(label))

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