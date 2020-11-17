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
from data import get_test_data, get_val_data, get_simple_val_data, get_Pilot


r'''This runner is only used to DO the whole test procedure, not used to train model'''

class FullRunner():
    def __init__(self, config):
        self.config = config
        self.Pn = self.config.RE.Pilotnum

    def run(self):
        if self.config.log.run_mode == 'validation':
            self.validation()
        elif self.config.log.run_mode == 'testing':
            self.test()
        
    def get_models(self, device):
        modelkv = {'cnn': CNNRouteEstimator,  'resnet': ResNetRouteEstimator, 'mlp': RouteEstimator}
        assert len(self.config.ckpt.redir) == len(self.config.ckpt.models)
        models = []
        for modelname, ckpt_fp in zip(self.config.ckpt.models, self.config.ckpt.redir):
            if modelname == 'cnn':
                M = CNNRouteEstimator()
            elif modelname == 'resnet':
                M = ResNetRouteEstimator(34)
            fp = os.path.join('/data/siyu/NAIC/workspace/CNNY2HEstimator/', ckpt_fp, 'mode_0_Pn_8/checkpoints/best.pth')
            M.load_state_dict(torch.load(fp))
            M.to(device)
            M.eval()
            models.append(M)
        return models
    
    def validation(self):
        device = 'cuda'
        logging.info('boosting')
        route_estimators = self.get_models(device)
        Yp, Yd, X_label, H_label = get_simple_val_data(mode = 0, Pn = self.Pn) # Yp, Yd Nsx1024
        Pilot = get_Pilot(self.Pn)
        predXds = []
        accXp = []

        for i, model in enumerate(route_estimators):
            cnn = True if (self.config.ckpt.models[i] == 'cnn' or 'resnet') else False
            predXp, predXd, predH = self.single_validation(model, Yp.copy(), Yd.copy(), device, cnn)
            acc = (predXp == Pilot).astype('float32').mean(-1)
            nmse = self.NMSE(predH, H_label)
            acc_avg = acc.mean()
            logging.info(f'model {self.config.ckpt.models[i]} performance: acc {acc_avg} || nmse {nmse}')
            predXds.append(predXd)
            accXp.append(acc)
    
    def single_validation(self, route_estimator, Yp, Yd, device, cnn):
        Yp2 = Yp.copy()
        if cnn is True:
            Yp = np.reshape(Yp, [len(Yp), 2, 16, 32], order = 'F')
        Yp = torch.Tensor(Yp).to(device)
        predH = route_estimator(Yp).detach().cpu().numpy()
        Yd = transfer_Y(Yd)
        Hf = transfer_H(predH)
        _, predXd = self.MLReceiver(Yd, Hf)
        Yp2 = transfer_Y(Yp2)
        _, predXp = self.MLReceiver(Yp2, Hf)
        return predXp, predXd, predH
        # acc = (pred_X == X_label).astype('float32').mean()
        # accp = (predXp == Pilot).astype()
        # logging.info(f'Results of validation at {self.config.boosting[i]} || {self.config.ckpt.redir[i]} \n acc: {acc}, nmse: {nmse}')
    
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


    def NMSE(self, H_pred, H_label):
        mse = np.power(H_pred-H_label, 2).sum()
        norm = np.power(H_label, 2).sum()
        return mse / norm
    
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
            # if num % 100 == 0:
            #     print('P{0}'.format(label),': Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

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