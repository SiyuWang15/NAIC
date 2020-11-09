import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
import logging
import sys
import os 
sys.path.append('..')
from Estimators import ModeEstimator
from data import get_Yp_modes

class Y2ModeRunner():
    def __init__(self, config):
        self.config = config
    
    def get_optimizer(self, parameters):
        if self.config.train.optimizer == 'adam':
            return torch.optim.Adam(parameters, lr = self.config.train.lr)
        elif self.config.train.optimizer == 'sgd':
            return torch.optim.SGD(parameters, lr = self.config.train.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.train.optimizer))
    
    def run(self):
        model = ModeEstimator()
        device = 'cuda'
        model.to(device)
        optimizer = self.get_optimizer(model.parameters())

        train_set, val_set = get_Yp_modes(self.config.OFDM.Pilotnum)
        train_dataloader = DataLoader(
            dataset=train_set, 
            batch_size=128,
            shuffle=False,
            num_workers=8,
            drop_last=True,
            pin_memory=True)
        val_dataloader = DataLoader(
            dataset=val_set, 
            batch_size=128,
            shuffle=False,
            num_workers=8,
            drop_last=True,
            pin_memory=True
        )
        best_acc = 0.0
        for epoch in range(self.config.train.n_epochs):
            it = 0
            model.eval()
            preds = []
            modes = []
            for (Yp, mode) in val_dataloader:
                Yp = Yp.float().to(device)
                mode = mode.to(device)
                pred = model(Yp)
                pred = torch.argmax(pred, dim = 1, keepdim=False)
                preds.append(pred)
                modes.append(mode)
            preds = torch.cat(preds, 0)
            modes = torch.cat(modes, 0)
            acc = (preds == modes).float().mean()
            logging.info('epoch: {} || Acc: {}'.format(epoch, acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(self.config.log.ckpt_dir, \
                    'best_ckpt_P{}.pth'.format(self.config.OFDM.Pilotnum)))


            model.train()
            for (Yp, mode) in train_dataloader:
                it += 1
                Yp = Yp.float().to(device)
                label = mode.to(device)
                pred = model(Yp).float()

                loss = nn.CrossEntropyLoss()(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = torch.argmax(pred, 1, keepdim=False)
                acc = (pred == label).float().mean()
                if it % self.config.log.print_freq == 0:
                    logging.info('iter: {} || loss: {}, acc: {}'.format(it, loss.item(), acc.item()))