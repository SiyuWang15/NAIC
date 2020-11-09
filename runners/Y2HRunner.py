import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
import logging
import sys
sys.path.append('..')
from Estimators import RouteEstimator
from get_data import get_YH_data

class Y2HRunner():
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
    
    def NMSE(self, H_pred, H_label):
        mse = nn.MSELoss(reduction='sum')(H_pred, H_label)
        norm = torch.pow(H_label, 2).sum()
        return mse / norm


    def run(self):
        train_set, val_set = get_YH_data(self.mode, self.Pn, self.config.model.Hdom)
        logging.info('Data Loaded!')
        # outdim = 2*2*32 if self.config.model.Hdom == 'time' else 256
        # assert self.config.model.out_dim == outdim  'out dimension not consistent with H domain'
        device = 'cuda'
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
            drop_last=False
        )
        model = RouteEstimator(self.config.model.in_dim, self.config.model.out_dim, self.config.model.h_dims, \
            self.config.model.act).to(device)
        optimizer = self.get_optimizer(model.parameters())
        best_nmse = 1000.

        logging.info('Everything prepared well, start to train...')

        for epoch in range(self.config.train.n_epochs):
            it = 0
            model.train()
            for Yp, H_label in train_loader: # H_label: bsx256
                it += 1
                Yp = Yp.to(device)
                H_label = H_label.to(device)

                H_pred = model(Yp)
                loss = nn.MSELoss(reduction='mean')(H_pred, H_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                if it % self.config.log.print_freq == 0:
                    logging.info('Iter{}/Epoch{} || MSE Loss: {}'.format(it, \
                        epoch, self.NMSE(H_pred, H_label).cpu().item()))
                        
            with torch.no_grad():
                model.eval()
                H_labels = []
                H_preds = []
                for Yp, H_label in val_loader:
                    Yp = Yp.to(device)
                    H_label = H_label.to(device)
                    H_pred = model(Yp)
                    H_preds.append(H_pred)
                H_labels = torch.cat(H_labels, 0)
                H_preds = torch.cat(H_preds, 0)
                nmse = self.NMSE(H_preds, H_labels).item()
                torch.save(model.state_dict(), os.path.join(self.config.log.ckpt_dir, 'epoch{}.pth'.format(epoch)))
                if nmse < best_nmse:
                    best_nmse = nmse
                    torch.save(model.state_dict(), self.config.log.ckpt_dir, 'best.pth')
                
                logging.info('Validation Epoch {} || nmse: {}, best nmse: {}'.format( \
                    epoch, nmse.cpu().item(), best_nmse))
                    