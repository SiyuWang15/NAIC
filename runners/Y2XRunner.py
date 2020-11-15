r'''to be implemented'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
import logging
import sys
import os
sys.path.append('..')
from Estimators import Y2XEstimator
from data import get_YX_data

class Y2XRunner():
    def __init__(self, config):
        self.config = config
        self.mode = config.OFDM.mode
        self.Pn = config.OFDM.Pilotnum
    
    def get_optimizer(self, parameters):
        if self.config.train.optimizer == 'adam':
            return torch.optim.Adam(parameters, lr = self.config.train.lr)
        elif self.config.train.optimizer == 'sgd':
            return torch.optim.SGD(parameters, lr = self.config.train.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.train.optimizer))
    
    def run(self):
        device = 'cuda'
        model = Y2XEstimator(self.config.model.in_ch, self.config.model.out_ch, self.config.model.numblocks).to(device)
        train_set, val_set = get_YX_data(self.mode, self.Pn)
        train_loader = DataLoader(
            train_set, 
            batch_size=self.config.train.batch_size, 
            shuffle=True, 
            num_workers=8, 
            drop_last=False
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=self.config.train.val_batch_size, 
            num_workers=8,
            drop_last=False,
            shuffle=False
        )
        best_acc = 0.0
        optimizer = self.get_optimizer(model.parameters())
        logging.info('Everything prepared well, start to train...')

        for epoch in range(self.config.train.n_epochs):
            it = 0
            model.train()
            for Y, X in train_loader:
                Y = Y.to(device) # bsx4x16x32
                X = X.to(device) # bsx2x16x32
                X_logits = model(Y)
                loss = nn.BCELoss(reduction='mean')(X_logits, X)
                acc = ((X_logits > 0.5).int() == X).float().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if it % self.config.log.print_freq == 0:
                    logging.info(f"Iter/Epoch [{it}]/[{epoch}] || mode {self.mode} || Pn {self.Pn} || loss: {loss.item():.5f}, acc: {acc:.5f}")
                it += 1
            val_N = 0
            val_cnt = 0
            losses = []
            model.eval()
            for Y, X in val_loader:
                Y = Y.to(device)
                X_logits = model(Y).cpu().detach()
                loss = nn.BCELoss(reduction='mean')(X_logits, X).numpy()
                X_pred = (X_logits > 0.5).numpy()
                X = X.numpy()
                val_cnt += (X_pred == X).sum()
                val_N += X.shape[0]
                losses.append(loss.item())
            loss = np.asarray(losses).mean()
            acc = val_cnt / (float(val_N)*1024)
            logging.info(f"Epoch {epoch} || loss: {loss:.5f}  acc: {acc:.5f}")
            fp = os.path.join(self.config.log.ckpt_dir, f'epoch{epoch}.pth')
            torch.save(model.state_dict(), fp)
            logging.info(f'checkpoint save at {fp}')
