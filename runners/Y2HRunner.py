import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
import os
import sys
sys.path.append('..')
from Estimators import CNN_Estimation, FC_ELU_Estimation, NMSELoss, ResNet34, ResNet50, Denoise_Resnet18
from data import get_YH_data_random
from utils import *
from func import MLReceiver
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
    
    def get_dataloader(self):
        train_set, val_set = get_YH_data_random(self.mode, self.Pn)
        train_loader = DataLoader(train_set, batch_size=self.config.train.train_batch_size, \
            shuffle = True, num_workers = 16, drop_last = False)
        val_loader = DataLoader(val_set, batch_size=self.config.train.val_batch_size, \
            shuffle=False, num_workers=16, drop_last=False)
        return train_loader, val_loader
    
    def evaluation(self, Ht, Yd, X):
        # Ht bsx2x4x32 must be on cpu
        X = X.numpy()
        Ht = Ht.numpy()
        Ht = Ht[:, 0, :, :] + 1j * Ht[:, 1, :, :]
        Hf = np.fft.fft(Ht, 256) / 20.
        Hf = np.reshape(Hf, [len(Ht), 2, 2, 256], order = 'F')
        Yd = Yd.numpy()
        Yd = Yd[:, 0, :, :] + 1j*Yd[:, 1, :, :]
        _, predX = MLReceiver(Yd, Hf)
        acc = (X == predX).astype('float32').mean()
        return acc

    def run(self):
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
        elif self.config.cnnmodel == 'resnet50':
            CNN = ResNet50().to(device)
        if not self.config.train.CNN_resume == 'None':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/CNN', \
                self.config.train.CNN_resume, 'checkpoints/best.pth')
            state_dicts = torch.load(fp)
            CNN.load_state_dict(state_dicts['cnn'])
            FC.load_state_dict(state_dicts['fc'])
            logging.info(f'load state dicts of CNN and FC from {fp}.')
        else:
            assert not self.config.train.FC_resume == 'None'
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/FC', self.config.train.FC_resume, 'checkpoints/best.pth')
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
        
        denoiser = Denoise_Resnet18()
        denoiser.to(device)
        fp = f'/data/siyu/NAIC/workspace/denoiser/mode_{self.mode}_Pn_{self.Pn}/{self.config.train.denoiser_resume}/checkpoints/best.pth'
        denoiser.load_state_dict(torch.load(fp)['denoiser'])
        denoiser.eval()
        for param in denoiser.parameters():
            param.requires_grad = False
        logging.info('Freeze Denoiser.')

        best_nmse = 1000.
        criterion = NMSELoss()
        optimizer_CNN = self.get_optimizer(CNN.parameters(), self.config.train.cnn_lr)
        
        logging.info('Everything prepared well, start to train...')
        for epoch in range(self.config.n_epochs):
            CNN.eval()
            FC.eval()
            with torch.no_grad():
                Ht_list = []
                Hlabel_list = []
                X_label = []
                Yd_list = []
                for Y, Ylabel, Yp4fc, X, H_label in val_loader:
                    bs = Y.shape[0]
                    dY = denoiser(Y.to(device)).reshape(bs, 1, 8, 256).detach().cpu().numpy() # this is on cpu
                    denoise_mse = nn.MSELoss()(torch.Tensor(dY), Ylabel)
                    # print(f'denoise mse: {denoise_mse}')
                    Yp4cnn, Yd = extract(dY) # bsx2x2x256
                    Hf = FC(Yp4fc.to(device))
                    Hf = Hf.reshape(bs, 2, 4, 256)

                    cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim = 2)

                    Ht = CNN(cnn_input).reshape(bs, 2, 4, 32).cpu()
                    # Ht = CNN(cnn_input).reshape(bs, 2, 32).cpu()
                    Ht_list.append(Ht)
                    Hlabel_list.append(H_label.float())
                    Yd_list.append(Yd)
                    X_label.append(X)
                Ht = torch.cat(Ht_list, dim = 0)
                Hlabel = torch.cat(Hlabel_list, dim = 0)
                Yd = torch.cat(Yd_list, dim = 0)
                Xlabel = torch.cat(X_label, dim = 0)
                loss = criterion(Ht, Hlabel)
                if loss < best_nmse:
                    best_nmse = loss.item()
                    state_dicts = {
                        'cnn': CNN.state_dict(),
                        'fc': FC.state_dict(),
                        'epoch_num': epoch
                    }
                    torch.save(state_dicts, os.path.join(self.config.ckpt_dir, f'best.pth'))
                logging.info(f'Validation Epoch [{epoch}]/[{self.config.n_epochs}] || NMSE {loss.item():.5f}, best nmse: {best_nmse:.5f}')
                if epoch % self.config.save_freq == 0:
                    fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                    state_dicts = {
                        'cnn': CNN.state_dict(),
                        'fc': FC.state_dict()
                    }
                    torch.save(state_dicts, fp)
                    logging.info(f'{fp} saved.')
                if epoch % self.config.eval_freq == 0 and epoch > 0:
                    acc = self.evaluation(Ht, Yd, Xlabel)
                    logging.info(f'Evaluation Epoch [{epoch}], acc: {acc:.5f}')
                
                
            current_lr = optimizer_CNN.param_groups[0]['lr']
            
            logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] learning rate: {current_lr:.4e}')
            # model training
            CNN.train()

            for it, (Y, Ylabel, Yp4fc, X, H_label) in enumerate(train_loader): # H_train: bsx2(real and imag)x4x32
                bs = Yp4fc.shape[0]
                dY = denoiser(Y.to(device)).reshape(bs, 1, 8, 256).detach().cpu().numpy()
                optimizer_CNN.zero_grad()
                Yp4fc = Yp4fc.to(device)
                Hf = FC(Yp4fc)
                Yp4fc = Yp4fc.cpu() # release gpu memory
                Hf = Hf.reshape(bs, 2, 4, 256)
                Yp4cnn, Yd = extract(dY)

                cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim = 2)
                Ht = CNN(cnn_input).reshape(bs, 2, 4, 32)
                H_label = H_label.to(device)
                loss = criterion(Ht, H_label)
                loss.backward()
                optimizer_CNN.step()
                if it % self.config.print_freq == 0:
                    # print(nmse)
                    logging.info(f'CNN Mode:{self.mode} || Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}')

            if epoch % self.config.lr_decay == 0 and epoch > 0:
            # if epoch % self.config.lr_decay == 0:
                for param_group in optimizer_CNN.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, self.config.lr_threshold)