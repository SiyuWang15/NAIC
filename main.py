import torch
import torch.nn as nn
import numpy as np 
import logging
from data import get_data, get_H, get_val_data
from models import MLP
from utils import set_logger, seed_everything

in_dim = 1024
out_dim = 1024
h_dim = 2048
batch_size = 1024
val_batch_size = 256
lr = 0.002

val_freq = 20
print_freq = 5
log = 'training3.log'

def main():
    device = 'cuda'
    set_logger(log)
    seed_everything()
    model = MLP(in_dim,  out_dim, h_dim, 2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    H_train, H_val = get_H()

    data = get_data(batch_size, H_train, device)
    it = 0
    for Y, X in data:
        it += 1
        Y = Y[:, :, 1, :, :].reshape(-1, 1024)
        pred = model(Y)
        loss = nn.MSELoss()(pred, X)
        acc = ((pred > 0.5) == X).sum() / float(batch_size * 1024)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if it % print_freq == 0:
            logging.info('iter: {} || loss: {}, acc: {}'.format(it, loss.cpu().item(), acc.cpu().item()))
        
        if it % val_freq == 0:
            val_y, val_x = get_val_data(val_batch_size, H_train, device)
            val_y = val_y[:, :, 1, :, :].reshape(-1, 1024)
            pred = model(val_y) 
            loss = nn.MSELoss()(pred, val_x)
            pred = pred > 0.5
            acc = (pred == val_x).sum() / float(val_batch_size * 1024)
            logging.info('iter: {}, validation || loss: {}, accuracy: {}'.format(it, loss.cpu().item(), acc.cpu().item()))



if __name__ == "__main__":
    main()