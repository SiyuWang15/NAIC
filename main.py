import torch
import torch.nn as nn
import numpy as np
import yaml
from argparse import Namespace
import logging
from datetime import datetime
import os
from data import get_data, get_H, get_val_data
from models import MLP, ComplexMLP
from utils import set_logger, seed_everything, bit_err

def main(config):
    device = 'cuda'
    set_logger(config)
    logging.info(config)
    description = '''Description: random H, reserve all pilot y, but not pilot x.'''
    logging.info(description)
    seed_everything()

    model = MLP(config.model.in_dim,  config.model.out_dim, config.model.h_dim, config.model.n_blocks) \
        if config.model.n_blocks != -1 else ComplexMLP(config.model.in_dim, config.model.out_dim, config.model.h_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    
    H_train, H_val = get_H()

    data = get_data(config.train.batch_size, H_train, device, config)
    it = 0
    model.train()
    for Y, X in data:
        it += 1
        # Y = Y[:, :, 1, :, :].reshape(-1, 1024)
        pred = model(Y)
        loss = nn.MSELoss()(pred, X)
        err = bit_err(X, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if it % config.log.print_freq == 0:
            logging.info('iter: {} || loss: {}, acc: {}'.format(it, loss.cpu().item(), err))
        
        if it % config.log.val_freq == 0:
            model.eval()
            val_y, val_x = get_val_data(config.train.val_batch_size, H_val, device, config)
            # val_y = val_y[:, :, 1, :, :].reshape(-1, 1024)
            pred = model(val_y) 
            loss = nn.MSELoss()(pred, val_x)
            err = bit_err(val_x, pred)
            logging.info('iter: {}, validation || loss: {}, accuracy: {}'.format(it, loss.cpu().item(), err))
            print('----log at {}'.format(config.log.log_dir))
            model.train()
        if it > config.train.n_iters:
            logging.info('{}-iter training Complete. Log save at {}'.format(it, config.log.log_dir))


if __name__ == "__main__":
    config = yaml.load(open('config.yml'))
    for k, v in config.items():
        config[k] = Namespace(**v)
    config = Namespace(**config)
    now = datetime.now()
    config.log.log_dir = os.path.join('workspace', config.log.log_prefix, now.strftime('%H-%M-%S'))
    os.makedirs(config.log.log_dir)
    main(config)