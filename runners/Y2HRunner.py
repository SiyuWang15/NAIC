r'''
    This is training runner for channel estimation. FC for coarse estimation and CNN for finer estimation.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
import os
# import sys
# sys.path.append('..')
from Estimators import CNN_Estimation, FC_ELU_Estimation, NMSELoss, ResNet34
from data import get_YH_data_random 
from utils import *

class Y2HRunner():
    def __init__(self, config, cnn:bool=False):
        self.config = config
        self.mode = config.mode
        self.Pn = config.Pn
    
    def get_optimizer(self, parameters, lr):
        if self.config.train.optimizer == 'adam':
            return optim.Adam(parameters, lr = lr)
        elif self.config.train.optimizer == 'sgd':
            return optim.SGD(parameters, lr = lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.train.optimizer))

    def run(self):
        if self.config.model == 'fc':
            self.train_FC()
        elif self.config.model == 'cnn':
            self.train_CNN()
    
    def get_dataloader(self):
        train_set, val_set = get_YH_data_random(self.mode, self.Pn)
        train_loader = DataLoader(train_set, batch_size=self.config.train.train_batch_size, \
            shuffle = True, num_workers = 16, drop_last = False)
        val_loader = DataLoader(val_set, batch_size=self.config.train.val_batch_size, \
            shuffle=False, num_workers=16, drop_last=False)
        return train_loader, val_loader

    def train_FC(self):
        device = 'cuda'
        train_loader, val_loader = self.get_dataloader()
        logging.info('Data Loaded!')
        FC = FC_ELU_Estimation(self.config.FC.in_dim, self.config.FC.h_dim, self.config.FC.out_dim, self.config.FC.n_blocks)
        if not self.config.train.FC_resume == 'None':
            fp = os.path.join(f'./workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/FC', self.config.train.FC_resume, 'checkpoints/best.pth')
            try:
                FC.load_state_dict(torch.load(fp))
            except:
                FC.load_state_dict(torch.load(fp)['fp'])
            logging.info(f'Loading state dict of FC from {self.config.train.FC_resume}')
        FC.to(device)

        criterion = NMSELoss()
        optimizer = self.get_optimizer(FC.parameters(), self.config.train.fc_lr)
        best_nmse = 1000.

        logging.info('Everything prepared well, start to train...')

        for epoch in range(self.config.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] learning rate: {current_lr:.4e}')

            FC.train()
            for it, (Yp4fc, _, _, _, H_label) in enumerate(train_loader):
                bs = len(Yp4fc)
                optimizer.zero_grad()
                # 真实的频域信道，获取标签
                Hf_label = process_H(H_label).to(device)
                # 第一层网络输入
                Yp4fc = Yp4fc.to(device)
                Hf = FC(Yp4fc).reshape(bs, 2, 4, 256)
                # 计算loss
                loss = criterion(Hf, Hf_label)
                loss.backward()
                optimizer.step()
                if it % self.config.print_freq == 0:
                    # print(nmse)
                    logging.info(f'Mode:{self.mode} || Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}')
            FC.eval()
            with torch.no_grad():
                Hf_list = []
                Hflabel_list = []
                for Yp4fc, _, _, _, H_label in val_loader:
                    bs = Yp4fc.shape[0]
                    # 真实的频域信道，获取标签
                    H_label = process_H(H_label)

                    Yp4fc = Yp4fc.to(device)
                    Hf = FC(Yp4fc).reshape(bs, 2, 4, 256).cpu()
                    Hf_list.append(Hf)
                    Hflabel_list.append(H_label)
                Hf = torch.cat(Hf_list, dim = 0)
                Hflabel = torch.cat(Hflabel_list, dim = 0)
                loss = criterion(Hf, Hflabel)

                fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                torch.save({'fc': FC.state_dict()}, fp)
                if loss < best_nmse:
                    torch.save({'fc': FC.state_dict()}, os.path.join(self.config.ckpt_dir, 'best.pth'))
                    best_nmse = loss.item()
                logging.info(f'{fp} saved!')
                logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] || NMSE {loss.item():.5f}, best nmse: {best_nmse:.5f}')

    def train_CNN(self):
        device = 'cuda'
        train_set, val_set = get_YH_data_random(self.mode, self.Pn)
        train_loader, val_loader = self.get_dataloader()
        logging.info('Data Loaded!')

        FC = FC_ELU_Estimation(self.config.FC.in_dim, self.config.FC.h_dim, self.config.FC.out_dim, self.config.FC.n_blocks)
        FC.to(device)
        if self.config.cnnmodel == 'base':
            CNN = CNN_Estimation().to(device)
        elif self.config.cnnmodel == 'resnet34':
            CNN = ResNet34().to(device)
        else:
            raise NotImplementedError
        
        if not self.config.train.CNN_resume == 'None':
            fp = os.path.join(f'./workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/CNN', \
                self.config.train.CNN_resume, 'checkpoints/best.pth')
            state_dicts = torch.load(fp)
            CNN.load_state_dict(state_dicts['cnn'])
            FC.load_state_dict(state_dicts['fc'])
            logging.info(f'load state dicts of CNN and FC from {fp}.')
        else:
            assert not self.config.train.FC_resume == 'None'
            fp = os.path.join(f'./workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/FC', self.config.train.FC_resume, 'checkpoints/best.pth')
            try:
                FC.load_state_dict(torch.load(fp)['fc'])
            except:
                FC.load_state_dict(torch.load(fp))
            logging.info(f'Loading state dict of FC from {self.config.train.FC_resume}, random initialize CNN')
        
        if self.config.train.freeze_FC:
            for param in FC.parameters():
                param.requires_grad = False
            logging.info('Freeze FC layer.')
            FC.eval()
        
        best_nmse = 1000.
        
        for k, v in CNN.named_parameters():
            if k == 'linear.weight' or k == 'linear.bias':
                logging.info('do not freeze linear layer.')
                v.requires_grad = True
            else:
                v.requires_grad = False

        criterion = NMSELoss()
        optimizer_CNN = self.get_optimizer(filter(lambda p: p.requires_grad,  CNN.parameters()), self.config.train.cnn_lr)
        symbol = not self.config.train.freeze_FC
        if symbol:
            optimizer_FC = self.get_optimizer(FC.parameters(), self.config.train.fc_lr)
            
        logging.info('Everything prepared well, start to train...')
        for epoch in range(self.config.n_epochs):
            CNN.eval()
            FC.eval()
            with torch.no_grad():
                Ht_list = []
                Hlabel_list = []
                for Yp4fc, Yp4cnn, Yd, X, H_label in val_loader:
                    bs = Yp4fc.shape[0]
                    Yp4fc = Yp4fc.to(device)
                    Hf = FC(Yp4fc)
                    Yp4fc = Yp4fc.cpu() # release gpu memory
                    Hf = Hf.reshape(bs, 2, 4, 256)
                    if self.config.use_yp:
                        cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim = 2)
                    else:
                        cnn_input = torch.cat([Yd.to(device), Hf], dim = 2)
                    Ht = CNN(cnn_input).reshape(bs, 2, 4, 32).cpu()
                    # Ht = CNN(cnn_input).reshape(bs, 2, 32).cpu()
                    Ht_list.append(Ht)
                    Hlabel_list.append(H_label.float())
                Ht = torch.cat(Ht_list, dim = 0)
                Hlabel = torch.cat(Hlabel_list, dim = 0)
                loss = criterion(Ht, Hlabel)
                if loss < best_nmse:
                    best_nmse = loss.item()
                    state_dicts = {
                        'cnn': CNN.state_dict(),
                        'fc': FC.state_dict(),
                        'epoch_num': epoch
                    }
                    torch.save(state_dicts, os.path.join(self.config.ckpt_dir, f'best.pth'))
                if epoch % self.config.save_freq == 0:
                    fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                    state_dicts = {
                        'cnn': CNN.state_dict(),
                        'fc': FC.state_dict()
                    }
                    torch.save(state_dicts, fp)
                    logging.info(f'{fp} saved.')
                logging.info(f'Validation Epoch [{epoch}]/[{self.config.n_epochs}] || NMSE {loss.item():.5f}, best nmse: {best_nmse:.5f}')
                
                
            current_lr = optimizer_CNN.param_groups[0]['lr']
            if symbol:
                current_fc_lr = optimizer_FC.param_groups[0]['lr']
                logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] cnn learning rate: {current_lr:.4e}, fc learning rate: {current_fc_lr:.4e}')
            else:
                logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] learning rate: {current_lr:.4e}')
            # model training
            CNN.train()
            if symbol:
                FC.train()

            for it, (Yp4fc, Yp4cnn, Yd, X, H_label) in enumerate(train_loader): # H_train: bsx2(real and imag)x4x32
                bs = Yp4fc.shape[0]
                optimizer_CNN.zero_grad()
                Yp4fc = Yp4fc.to(device)
                Hf = FC(Yp4fc)
                Yp4fc = Yp4fc.cpu() # release gpu memory
                Hf = Hf.reshape(bs, 2, 4, 256)
                if self.config.use_yp:
                    cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim = 2)
                else:
                    cnn_input = torch.cat([Yd.to(device), Hf], dim = 2)
                Ht = CNN(cnn_input).reshape(bs, 2, 4, 32)
                # Ht = CNN(cnn_input).reshape(bs, 2, 32)
                H_label = H_label.to(device)

                loss = criterion(Ht, H_label)
                loss.backward()
                optimizer_CNN.step()
                if symbol:
                    optimizer_FC.step()

                if it % self.config.print_freq == 0:
                    # print(nmse)
                    logging.info(f'CNN Mode:{self.mode} || Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}')

            if epoch >0:
                if epoch % self.config.lr_decay ==0:
                    optimizer_CNN.param_groups[0]['lr'] =  optimizer_CNN.param_groups[0]['lr'] * 0.5
                    if symbol:
                        optimizer_FC.param_groups[0]['lr'] =  optimizer_FC.param_groups[0]['lr'] * 0.5
            
                if optimizer_CNN.param_groups[0]['lr'] < self.config.lr_threshold:
                    optimizer_CNN.param_groups[0]['lr'] = self.config.lr_threshold
                    if symbol:
                        optimizer_FC.param_groups[0]['lr'] =  self.config.lr_threshold

            