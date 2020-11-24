import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
import os 
import shutil
import logging
import math
import multiprocessing as mp
import sys
import time
sys.path.append('..')
from utils import *
from func import SoftMLReceiver as MLReceiver
from Estimators import CNN_Estimation, FC_ELU_Estimation, NMSELoss, ResNet34, Densenet, ResNet101
from data import get_test_data, get_val_data


r'''This runner is only used to DO the whole test procedure, not used to train model'''

class EnsembleRunner():
    def __init__(self, config):
        self.config = config
        self.Pn = config.Pn
        self.mode = config.mode
        self.FCconf = config.FC
        self.device = 'cuda'
        self.N = 4000
        self.get_data()
    
    def get_data(self):
        N = self.N
        D = np.load('/data/siyu/NAIC/dataset/evaluation.npy', allow_pickle = True).item()
        self.X, self.Y, self.H = D['x'], D['y'], D['h'] # numpy Nx1024, Nxx2x2x2x256, Nx2x4x32
        self.X, self.Y, self.H = self.X[:N], self.Y[:N], self.H[:N]
    
    def run(self):
        models = self.get_model()
        X_jh = np.load('/data/siyu/NAIC/dataset/X_jh.npy', allow_pickle=True)[:self.N]
        
        logging.info(f'model jianghao || acc : {(X_jh == self.X).mean()}')
        batch_sizes = [500]*10
        X_collect = []
        Ht_collect = []
        for i, model in enumerate(models):
            acc, nmse, predx, predh = self.evaluate_fc_cnn(model[0], model[1], batch_sizes[i])
            X_collect.append(predx)
            Ht_collect.append(predh)
            logging.info(f'model {self.modelnames[i]} || acc: {acc:.7f} || nmse: {nmse:.5f}')
        X_collect.append(X_jh)
        X_collect = np.stack(X_collect, axis=0)
        # Ht_collect = np.stack(Ht_collect, axis=0)
        tag = 1 if self.Pn == 32 else 2
        # np.save(os.path.join(self.config.log_dir, f'X_pre_{tag}.npy'), X_collect)
        thre = int((len(self.modelnames) + 1)/2)
        predx = (X_collect.sum(axis = 0) > thre)
        X_collect = np.concatenate([X_collect, np.expand_dims(predx, axis = 0)], axis = 0)
        acc = (predx == self.X).mean()
        logging.info(f'Final accuracy: {acc}')
        self.similarity(X_collect)
        

    def similarity(self, x):
        modelnames = self.modelnames
        modelnames.append('jianghao')
        info = []
        for i in range(-1, len(modelnames)):
            if i == -1:

                s = '{:<15}'.format('similarity')
                for j in range(len(modelnames)):
                    s += f'\t{modelnames[j]:>15}'
            else:
                s = f'{modelnames[i]:<15}'
                for j in range(len(modelnames)):
                    s += f'\t{(x[i, :] == x[j, :]).mean():.13f}'    
            info.append(s)
        f = open(os.path.join(self.config.log_dir, 'similarity'), 'w')
        for s in info:
            f.write(s + '\n')

    def get_model(self):
        modelnames = ['resnet18', 'resnet34']
        # modelnames = ['resnet18', 'resnet18']
        use_fc = [1]*10
        # ckpts = [
        #     # '/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1123-14-42-37/checkpoints/best.pth',
        #     '/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-20-57-33/checkpoints/best.pth',
        #     '/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-02-00-34/checkpoints/best.pth',
        #     '/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_0_Pn_8/EMA/1122-21-06-48/checkpoints/best_ema.pth'
        # ]
        # ckpts = [
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1123-21-43-21/checkpoints/epoch50.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1123-21-43-21/checkpoints/epoch30.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-02-00-34/checkpoints/epoch290.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-02-00-34/checkpoints/epoch280.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-02-00-34/checkpoints/epoch270.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1120-14-53-06/checkpoints/epoch280.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1120-14-53-06/checkpoints/best.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-02-00-34/checkpoints/best.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1123-21-43-21/checkpoints/best.pth',
        #     'workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-02-00-34/checkpoints/best.pth'
        # ]
        ckpts = [
            '/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_0_Pn_8/EMA/1121-15-28-33/checkpoints/best.pth', 
            '/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/1122-20-57-33/checkpoints/best.pth'
        ]
        models = []
        for i in range(len(modelnames)):
            state_dict = torch.load(ckpts[i])
            if modelnames[i] == 'densenet':
                CNN = Densenet()
                CNN = nn.DataParallel(CNN)
            elif modelnames[i] == 'resnet18':
                CNN = CNN_Estimation()
            elif modelnames[i] == 'resnet34':
                CNN = ResNet34()
            CNN.load_state_dict(state_dict['cnn'])
            CNN.eval()
            if use_fc[i]:
                FC = FC_ELU_Estimation(self.FCconf.in_dim, self.FCconf.h_dim, self.FCconf.out_dim, self.FCconf.n_blocks)
                FC.load_state_dict(state_dict['fc'])
                FC.eval()
                models.append((FC, CNN))
            else:
                models.append((CNN))
        self.modelnames = modelnames
        return models

    def evaluate_fc_cnn(self, FC, CNN, bs):
        device = 'cuda'
        FC.to(self.device)
        CNN.to(self.device)
        predXs = []
        predHts = []
        N = int(len(self.X) / bs)
        with torch.no_grad():
            for i in range(N):
                Y = self.Y[i*bs: (i+1)*bs] # bsx2x2x2x256
                Yp4fc = Y[:, :, 0, :, :].reshape(bs, -1)
                Hf = FC(torch.Tensor(Yp4fc).to(device)).reshape(bs, 2, 4, 256)
                Yp4cnn = Y[:, :, 0, :, :]
                Yd = Y[:, :, 1, :, :]
                net_input = torch.cat([torch.Tensor(Yd).to(device), torch.Tensor(Yp4cnn).to(device), Hf], dim = 2)
                Ht = CNN(net_input).reshape(bs, 2, 4, 32).detach().cpu().numpy()
                predHts.append(Ht.copy())
                Ht = Ht[:, 0, :, :] + Ht[:, 1, :, :] * 1j
                Hf_pred = np.fft.fft(Ht, 256) / 20.
                Hf_pred = np.reshape(Hf_pred, [bs, 2, 2, 256], order = 'F')
                Yd = Yd[:, 0, :, :] + 1j * Yd[:, 1, :, :]
                _, predX = MLReceiver(Yd, Hf_pred)
                predXs.append(predX)
                # print(f'[{i+1}]/[{N}] complete.')
            predXs = np.concatenate(predXs, axis = 0)
            predHts = np.concatenate(predHts, axis = 0)
            nmse = self.NMSE(predHts, self.H)
            acc = (predXs == self.X).mean()
        return acc, nmse, predXs, predHts
        
    def NMSE(self, H_pred, H_label):
        mse = np.power(H_pred - H_label, 2).sum()
        norm = np.power(H_label, 2).sum()
        return mse / norm


    def evaluate_v3(self, model):
        pass