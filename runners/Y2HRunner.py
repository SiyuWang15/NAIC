import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
import os
import sys
sys.path.append('..')
from Estimators import CNN_Estimation, FC_ELU_Estimation, NMSELoss, ResNet34, ResNet50
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

    def train_FC(self):
        device = 'cuda'
        train_set, val_set = get_YH_data_random(self.mode, self.Pn)
        train_loader = DataLoader(
            train_set, 
            batch_size=self.config.train.train_batch_size, 
            shuffle=True, 
            num_workers=8, 
            drop_last=False
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=self.config.train.val_batch_size, 
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        logging.info('Data Loaded!')

        FC = FC_ELU_Estimation(self.config.FC.in_dim, self.config.FC.h_dim, self.config.FC.out_dim, self.config.FC.n_blocks)
        if not self.config.train.FC_resume == 'None':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/FC', self.config.train.FC_resume, 'checkpoints/best.pth')
            FC.load_state_dict(torch.load(fp))
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
            for it, (Y_train, X_train, H_train) in enumerate(train_loader):
                batch_size = len(Y_train)
                optimizer.zero_grad()
                # 真实的频域信道，获取标签
                Hf_train_label = process_H(H_train)
                # 第一层网络输入
                Y_input_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F')
                Y_input_train = Y_input_train.float()
                Yp_train = Y_input_train[:,:,0,:,:].reshape(batch_size, 2*2*256).to(device)
                Hf_train_output = FC(Yp_train).reshape(batch_size, 2, 4, 256)

                # 计算loss
                loss = criterion(Hf_train_output, Hf_train_label.to(device))

                loss.backward()
                optimizer.step()

                if it % self.config.print_freq == 0:
                    # print(nmse)
                    logging.info(f'Mode:{self.mode} || Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}')
            FC.eval()
            with torch.no_grad():
                for Y_test, X_test, H_test in val_loader:
                    Ns = Y_test.shape[0]
                    # 真实的频域信道，获取标签
                    Hf_test_label = process_H(H_test)

                    # 第一层网络输入
                    Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
                    Y_input_test = Y_input_test.float()

                    Yp_test = Y_input_test[:,:,0,:,:] # 取出接收导频信号，实部虚部*2*256
                    Yp_test = Yp_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
                    Yp_test = Yp_test.to(device)
                    Hf_test_output = FC(Yp_test)
                    # 第一级网络输出
                    Hf_test_output = Hf_test_output.reshape(Ns, 2, 4, 256)

                    # 计算loss
                    loss = criterion(Hf_test_output, Hf_test_label.cuda())

                    
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
        train_loader = DataLoader(
            train_set, 
            batch_size=self.config.train.train_batch_size, 
            shuffle=True, 
            num_workers=8, 
            drop_last=False
        )
        val_loader = DataLoader(
            val_set, 
            shuffle=False,
            batch_size=self.config.train.val_batch_size, 
            num_workers=8,
            drop_last=False
        )
        logging.info('Data Loaded!')
        FC = FC_ELU_Estimation(self.config.FC.in_dim, self.config.FC.h_dim, self.config.FC.out_dim, self.config.FC.n_blocks)
        FC.to(device)
        if self.config.train.FC_resume is not 'None':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/FC', self.config.train.FC_resume, 'checkpoints/best.pth')
            FC.load_state_dict(torch.load(fp)['fc'])
            logging.info(f'Loading state dict of FC from {self.config.train.FC_resume}')
        if self.config.train.freeze_FC:
            for param in FC.parameters():
                param.requires_grad = False
            logging.info('Freeze FC layer.')
            FC.eval()
        
        if self.config.cnnmodel == 'base':
            CNN = CNN_Estimation().to(device)
        elif self.config.cnnmodel == 'resnet34':
            CNN = ResNet34().to(device)
        elif self.config.cnnmodel == 'resnet50':
            CNN = ResNet50().to(device)
        if not self.config.train.CNN_resume == 'None':
            CNN.load_state_dict(torch.load(self.config.train.CNN_resume))
        
        best_nmse = 1000.
        criterion = NMSELoss()
        optimizer_CNN = self.get_optimizer(CNN.parameters(), self.config.train.cnn_lr)
        symbol = not self.config.train.freeze_FC
        if symbol:
            optimizer_FC = self.get_optimizer(FC.parameters(), self.config.train.fc_lr)
            
        logging.info('Everything prepared well, start to train...')
        for epoch in range(self.config.n_epochs):
            print('=================================')
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

            for it, (Y_train, X_train, H_train) in enumerate(train_loader): # H_train: bsx2(real and imag)x4x32
                batch_size = Y_train.shape[0]
                optimizer_CNN.zero_grad()

                # 第一层网络输入
                Y_input_train = np.reshape(Y_train, [batch_size, 2, 2, 2, 256], order='F')
                Y_input_train = Y_input_train.float()
                Yp_train = Y_input_train[:,:,0,:,:]

                #输入网络
                Yp_train = Yp_train.reshape(batch_size, 2*2*256) # 取出接收导频信号，实部虚部*2*256

                Yp_train =  Yp_train.to(device)

                Hf_train_output = FC(Yp_train)
                # 第一级网络输出 以及 第二级网络输入
                Hf_train_output = Hf_train_output.reshape(batch_size, 2, 4, 256)
                
                Hf_input_train = Hf_train_output
                Yd_input_train = Y_input_train[:,:,1,:,:] 
                Yd_input_train = Yd_input_train.cuda()

                net_input = torch.cat([Yd_input_train, Hf_input_train], 2) # bsx2x6x256
                
                Ht_train_refine = CNN(net_input)

                #第二级网络输出
                Ht_train_refine = Ht_train_refine.reshape(batch_size, 2, 4, 32)

                # 第二级标签
                # Ht_train_label = torch.zeros([batch_size, 2, 4, 32], dtype=torch.float32)
                # Ht_train_label[:, 0, :, :] = H_train.real.float()
                # Ht_train_label[:, 1, :, :] = H_train.imag.float()
                Ht_train_label = H_train.float()

                # 计算loss
                loss = criterion(Ht_train_refine, Ht_train_label.cuda())
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

            CNN.eval()
            FC.eval()
            with torch.no_grad():

                for Y_test, X_test, H_test in val_loader:
                    Ns = Y_test.shape[0]

                    # 第一层网络输入
                    Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
                    Y_input_test = Y_input_test.float()
                    Yp_test = Y_input_test[:,:,0,:,:]

                    #输入网络
                    Yp_test = Yp_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
                    Yp_test =  Yp_test.cuda()
                    Hf_test_output = FC(Yp_test)
                    # 第一级网络输出 以及 第二级网络输入
                    Hf_test_output = Hf_test_output.reshape(Ns, 2, 4, 256)
                    
                    Hf_input_test = Hf_test_output
                    Yp_input_test = Y_input_test[:,:,1,:,:] 
                    Yp_input_test = Yp_input_test.cuda()

                    net_input = torch.cat([Yp_input_test, Hf_input_test], 2)
                    # net_input = torch.reshape(net_input, [Ns, 1, 12, 256])
                    
                    Ht_test_refine = CNN(net_input)

                    #第二级网络输出
                    Ht_test_refine = Ht_test_refine.reshape(Ns, 2, 4, 32)

                    # 第二级标签
                    # Ht_test_label = torch.zeros([Ns, 2, 4, 32], dtype=torch.float32)
                    # Ht_test_label[:, 0, :, :] = H_test.real.float()
                    # Ht_test_label[:, 1, :, :] = H_test.imag.float()
                    Ht_test_label = H_test.float()
                    # 计算loss
                    loss = criterion(Ht_test_refine, Ht_test_label.cuda())

                    fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                    state_dicts = {
                        'cnn': CNN.state_dict(),
                        'fc': FC.state_dict()
                    }
                    torch.save(state_dicts, fp)
                    if loss < best_nmse:
                        torch.save(state_dicts, os.path.join(self.config.ckpt_dir, 'best.pth'))
                        best_nmse = loss
                    logging.info(f'{fp} saved.')
                    logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] || NMSE {loss.item():.5f}, best nmse: {best_nmse:.5f}')