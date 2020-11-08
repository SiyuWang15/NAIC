import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from argparse import Namespace
import logging
from datetime import datetime
import os
from DataLoader import get_data
from models import MLP, ComplexMLP
from utils import set_logger, seed_everything, bit_err, arg_parser

def main(config):
    device = 'cuda'
    set_logger(config)
    logging.info(config)
    # description = '''Description:. '''
    # logging.info(description)
    seed_everything()

    model = MLP(config.model.in_dim,  config.model.out_dim, config.model.h_dim, config.model.n_blocks) \
        if config.model.n_blocks != -1 else ComplexMLP(config.model.in_dim, config.model.out_dim, config.model.h_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    train_dataset, val_dataset = get_data(config.train.random, config.slice, config.model.out_dim)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True)
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    logging.info('{} data loader is ready.'.format('random' if config.train.random else 'fixed'))
    model.train()
    for epoch in range(config.train.n_epochs):
        it = 0
        model.train()
        for Y, X in train_dataloader:
            Y = Y.float().to(device)
            X = X.float().to(device)
            it += 1
            pred = model(Y)
            loss = nn.BCELoss()(pred, X)
            err = bit_err(X, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % config.log.print_freq == 0:
                logging.info('iter/epoch: {}/{} || loss: {}, bit err: {}'.format(it, epoch, loss.cpu().item(), err))
        
        model.eval()
        preds = []
        val_xs = []
        for val_y, val_x in val_dataloader:
            val_y = val_y.float().to(device)
            val_x = val_x.float().to(device)
            pred = model(val_y)
            preds.append(pred)
            val_xs.append(val_x)
        pred = torch.cat(preds, dim = 0)
        val_x = torch.cat(val_xs, dim = 0)
        loss = nn.BCELoss()(pred, val_x)
        err = bit_err(val_x, pred)
        logging.info('epoch: {}, validation || loss: {}, bit err: {}'.format(epoch, loss.cpu().item(), err))
        ckpt_path = os.path.join(config.log.ckpt_dir, '{}.pth'.format(epoch))
        torch.save(model.state_dict(), ckpt_path)
        logging.info('checkpoint saved at {}'.format(ckpt_path))


if __name__ == "__main__":
    args = arg_parser()
    config = yaml.load(open('config.yml'))
    for k, v in config.items():
        config[k] = Namespace(**v)
    config = Namespace(**config)
    now = datetime.now()
    config.log.log_dir = os.path.join('workspace', config.log.log_prefix, now.strftime('%H-%M-%S'))
    config.log.ckpt_dir = os.path.join(config.log.log_dir, 'checkpoints')
    os.makedirs(config.log.ckpt_dir)
    config.slice = args.slice
    main(config)