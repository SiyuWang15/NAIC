import torch
import torch.nn as nn
import numpy as np
import yaml
from argparse import Namespace
import logging
from datetime import datetime
import os
from data import get_data, get_H, get_val_data
from models import MLP
from utils import set_logger, seed_everything

def main(config):
    device = 'cuda'
    set_logger(config)
    logging.info(config)
    seed_everything()

    model = MLP(config.model.in_dim,  config.model.out_dim, config.model.h_dim, config.model.n_blocks)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    
    H_train, H_val = get_H()

    data = get_data(config.train.batch_size, H_train, device)
    it = 0
    for Y, X in data:
        it += 1
        Y = Y[:, :, 1, :, :].reshape(-1, 1024)
        pred = model(Y)
        loss = nn.MSELoss()(pred, X)
        acc = ((pred > 0.5) == X).sum() / float(config.train.batch_size * 128)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if it % config.log.print_freq == 0:
            logging.info('iter: {} || loss: {}, acc: {}'.format(it, loss.cpu().item(), acc.cpu().item()))
        
        if it % config.log.val_freq == 0:
            val_y, val_x = get_val_data(config.train.val_batch_size, H_train, device)
            val_y = val_y[:, :, 1, :, :].reshape(-1, 1024)
            pred = model(val_y) 
            loss = nn.MSELoss()(pred, val_x)
            pred = pred > 0.5
            acc = (pred == val_x).sum() / float(config.train.val_batch_size * 128)
            logging.info('iter: {}, validation || loss: {}, accuracy: {}'.format(it, loss.cpu().item(), acc.cpu().item()))
            print('----log at {}'.format(config.log.log_dir))



if __name__ == "__main__":
    config = yaml.load(open('config.yml'))
    for k, v in config.items():
        config[k] = Namespace(**v)
    config = Namespace(**config)
    now = datetime.now()
    config.log.log_dir = os.path.join('workspace', config.log.log_prefix, now.strftime('%H-%M-%S'))
    os.makedirs(config.log.log_dir)
    main(config)