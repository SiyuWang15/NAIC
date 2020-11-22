import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
import os
import sys
sys.path.append('..')
from Estimators import Denoise_Unet, Denoise_Resnet18
from data import get_denoise_data
from utils import *

class DenoiserRunner():
    def __init__(self, config):
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
        train_set, val_set = get_denoise_data(self.mode, self.Pn)
        train_loader = DataLoader(train_set, batch_size=self.config.train.train_batch_size, \
            shuffle = True, num_workers = 16, drop_last = False)
        val_loader = DataLoader(val_set, batch_size=self.config.train.val_batch_size, \
            shuffle=False, num_workers=16, drop_last=False)
        return train_loader, val_loader

    def run(self):
        device = 'cuda'
        train_loader, val_loader = self.get_dataloader()

        if self.config.model == 'unet':
            model = Denoise_Unet().to(device)
        elif self.config.model == 'resnet18':
            model = Denoise_Resnet18().to(device)
        else:
            raise NotImplementedError
        if not self.config.train.resume == 'None':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/denoiser/{self.config.train.resume}/mode_{self.mode}_Pn__{self.Pn}/checkpoints/best.pth')
            model.load_state_dict(torch.load(fp))
            logging.info(f'Loading state dict from {fp}')
        
        best_mse = 1000.
        criterion = nn.MSELoss(reduction='mean')
        optimizer = self.get_optimizer(model.parameters(), self.config.train.lr)
        
        logging.info('Everything prepared well, start to train...')
        for epoch in range(self.config.n_epochs):
            model.eval()
            with torch.no_grad():
                losses = []
                for Y, Y_label in val_loader:
                    Y = Y.to(device)
                    Y_label = Y_label.to(device)
                    dY = model(Y).reshape(len(Y), 1, 8, 256)
                    loss = criterion(dY, Y_label)
                    losses.append(loss)
                loss = sum(losses)/ len(losses)
                logging.info(f'Validation epoch [{epoch}] || MSE: {loss:.5f}, best MSE: {best_mse:.5f}')
                if loss < best_mse:
                    best_mse = loss.item()
                    state_dicts = {
                        'denoiser': model.state_dict(),
                        'epoch_num': epoch, 
                        'mse': best_mse
                    }
                    torch.save(state_dicts, os.path.join(self.config.ckpt_dir, f'best.pth'))
                if epoch % self.config.save_freq == 0:
                    fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                    state_dicts = {
                        'denoiser': model.state_dict()
                    }
                    torch.save(state_dicts, fp)
                    logging.info(f'{fp} saved.')

            model.train()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] learning rate: {current_lr:.4e}')
            for it, (Y, Y_label) in enumerate(train_loader):
                optimizer .zero_grad()
                Y = Y.to(device)
                Y_label = Y_label.to(device)
                dY = model(Y).reshape(len(Y), 1, 8, 256)
                loss = criterion(dY, Y_label)
                loss.backward()
                optimizer.step()
                if it % self.config.print_freq == 0:
                    logging.info(f'Model: {self.config.model} || Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}')
            if epoch % self.config.lr_decay == 0 and epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, 1e-5)