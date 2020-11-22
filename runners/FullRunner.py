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
from Estimators import CNN_Estimation, FC_ELU_Estimation, NMSELoss
from data import get_test_data, get_val_data


r'''This runner is only used to DO the whole test procedure, not used to train model'''

class FullRunner():
    def __init__(self, config):
        self.config = config
        self.Pn = config.Pn
        self.mode = config.mode
        self.FCconf = config.FC
    
    def get_model(self, device):
        CNN = CNN_Estimation()
        FC = FC_ELU_Estimation(self.FCconf.in_dim, self.FCconf.h_dim, self.FCconf.out_dim, self.FCconf.n_blocks)
        if self.config.model == 'cnn':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/CNN',\
             self.config.resume, 'checkpoints/best.pth')
        elif self.config.model == 'ema':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/EMA',\
             self.config.resume, 'checkpoints/best_ema.pth')

        shutil.copy(fp, os.path.join(self.config.log_dir, 'best.pth'))

        logging.info(f'loading state dicts from [{fp}]')
        state_dicts = torch.load(fp)
        FC.load_state_dict(state_dicts['fc'])
        CNN.load_state_dict(state_dicts['cnn'])
        FC.to(device)
        CNN.to(device)
        return FC, CNN

    def run(self):
        if self.config.run_mode == 'validation':
            self.validation()
        elif self.config.run_mode == 'testing':
            self.test()
        
    def NMSE(self, H_pred, H_label):
        mse = np.power(H_pred - H_label, 2).sum()
        norm = np.power(H_label, 2).sum()
        return mse / norm
    
    def validation(self):
        device = 'cuda'
        FC, CNN = self.get_model(device)
        FC.eval()
        CNN.eval()
        logging.info('Model Constructed.')
        val_set = get_val_data(self.Pn, self.mode)
        val_loader = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False, num_workers=16)
        logging.info('Data Loaded.')
        predXs = []
        predHts = []
        Hlabels = []
        Xlabels = []
        with torch.no_grad():
            for i, (Yp4fc, Yp4cnn, Yd, X_label, H_label) in enumerate(val_loader):
                # print(Yp4fc.dtype, Yp4cnn.dtype, Yd.dtype, X_label.dtype, H_label.dtype)
                # assert 1==0
                bs = len(Yp4fc)
                Yp4fc = Yp4fc.to(device)
                Hf = FC(Yp4fc).reshape(bs, 2, 4, 256)
                if not self.config.use_yp:
                    cnn_input = torch.cat([Yd.to(device), Hf], dim = 2)
                else:
                    cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim=2)
                Ht = CNN(cnn_input)
                Ht = Ht.reshape(bs, 2, 4, 32).detach().cpu().numpy()
                predHts.append(Ht.copy())
                Ht = Ht[:, 0, :, :] + 1j * Ht[:, 1, :, :]
                Hf_pred = np.fft.fft(Ht, 256) / 20.
                Hf_pred = np.reshape(Hf_pred, [bs, 2, 2, 256], order = 'F')
                Yd = Yd.numpy()
                Yd = Yd[:, 0, :, :] + 1j * Yd[:, 1, :, :]
                _, predX = MLReceiver(Yd, Hf_pred)
                predXs.append(predX)
                Hlabels.append(H_label.numpy())
                Xlabels.append(X_label.numpy())
                logging.info(f'[{i+1}]/[{len(val_loader)}] complete.')
            predXs = np.concatenate(predXs, axis = 0)
            predHts = np.concatenate(predHts, axis = 0)
            Hlabels = np.concatenate(Hlabels, axis = 0)
            Xlabels = np.concatenate(Xlabels, axis = 0)
            acc = (predXs == Xlabels).astype('float32').mean()
            nmse = self.NMSE(predHts, Hlabels)
            logging.info(f'validation on {self.config.resume}, acc: {acc:.5f}, nmse: {nmse:.5f}')

    def test(self):
        device = 'cuda'
        FC, CNN = self.get_model(device)
        FC.eval()
        CNN.eval()
        Y = get_test_data(self.Pn) # 10000x2x2x256
        bs = 500
        predXs = []
        for i in range(int(len(Y) / bs)):
            YY = Y[i*bs:(i+1)*bs, :]
            Yp = YY[:, :, 0, :, :]
            Yp4cnn = Yp.copy()
            Yp = Yp.reshape(bs, 2*2*256)
            Yd = YY[:, :, 1, :, :]
            Hf = FC(torch.Tensor(Yp).to(device))
            Hf = Hf.reshape(bs, 2, 4, 256)
            if self.config.use_yp :
                net_input = torch.cat([torch.Tensor(Yd).to(device), torch.Tensor(Yp4cnn).to(device), Hf], dim =2)
            else:
                net_input = torch.cat([torch.Tensor(Yd).to(device), Hf], dim=2)
            Ht = CNN(net_input)
            Ht = Ht.reshape(bs, 2, 4, 32).detach().cpu().numpy()
            Ht = Ht[:, 0, :, :] + 1j*Ht[:, 1, :, :]
            Hf_pred = np.fft.fft(Ht, 256) / 20.
            Hf_pred = np.reshape(Hf_pred, [bs, 2, 2, 256], order = 'F')
            Yd = Yd[:, 0, :, :] + 1j * Yd[:, 1, :, :]
            _, predX = MLReceiver(Yd, Hf_pred)
            predXs.append(predX)
            logging.info(f'[{(i+1)*bs}]/[{len(Y)}] complete!')
        predXs = np.concatenate(predXs, axis = 0)
        predXs = np.array(np.floor(predXs+0.5), dtype = np.bool)
        tag = 1 if self.Pn == 32 else 2
        dp = os.path.join(self.config.log_dir, f'X_pre_{tag}.bin')
        predXs.tofile(dp)
        logging.info(f'Complete! Results saved at {dp}')
