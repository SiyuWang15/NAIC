import torch
import torch.nn as nn
import numpy as np 
import logging
from data import get_data, get_H, get_val_data
from models import MLP
from utils import set_logger, seed_everything


in_dim = 2048
out_dim = 1024
h_dim = 1024
batch_size = 1024
val_batch_size = 10000
lr = 0.02

val_freq = 100
print_freq = 100



def main():
    device = 'cuda'
    set_logger()
    seed_everything()
    model = MLP(in_dim,  out_dim, in_dim, 5)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    H_train, H_val = get_H()

    data = get_data(batch_size, H_train, device)
    it = 0
    for Y, X in data:
        it += 1
        pred = model(Y)
        loss = - torch.log(pred + 1e-4).sum(dim = 1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if it % print_freq == 0:
            logging.info('iter: {} || loss: {}'.format(it, loss.cpu().item()))
        
        if it % val_freq == 0:
            val_y, val_x = get_val_data(val_batch_size, H_val, device)
            pred = model(val_y) 
            loss = - torch.log(pred + 1e-4).sum(dim = 1).mean()
            pred = pred > 0.5
            acc = (pred == val_x).sum() / float(val_batch_size * 1024)
            logging.info('iter: {}, validation || loss: {}, accuracy: {}'.format(it, loss.cpu().item(), acc.cpu().item()))



if __name__ == "__main__":
    main()