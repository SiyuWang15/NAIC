import numpy as np 
import torch
from H_utils import *
import struct

Pilotnums = [32, 8]
modes = [0,1,2]
SNRdbs = [8, 9, 10, 11, 12]
# H_ind = 0

def get_H(H_path = 'dataset/H.npy', ratio = 0.9):
    H = np.load(H_path)
    H = H[:, 1, :, :] + 1j*H[:, 0, :, :]
    np.save('dataset/H_data.npy', H)
    split = int(ratio * 320000)
    H_train = H[:split, :, :]
    H_val = H[split:, :, :]
    return H_train, H_val

def get_data(batch_size, H_stack, device, config):
    while True:
        input_labels = []
        input_samples = []
        for row in range(batch_size):
            bits0 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
            bits1 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
            X = [bits0, bits1]
            H_ind = np.random.randint(0, len(H_stack))
            HH = H_stack[H_ind]
            mode = config.OFDM.mode if config.OFDM.mode != -1 else np.random.choice(modes, 1)[0]
            SNRdb = config.OFDM.SNRdb if config.OFDM.SNRdb != -1 else np.random.choice(SNRdbs, 1)[0]
            Pilotnum = config.OFDM.Pilotnum if config.OFDM.Pilotnum != -1 else np.random.choice(Pilotnums, 1)[0]
            YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20
            # YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
            XX = np.concatenate((bits0[:config.model.out_dim // 2], bits1[:config.model.out_dim // 2]), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = torch.Tensor(np.asarray(input_samples)).to(device)
        batch_x = torch.Tensor(np.asarray(input_labels)).to(device)
        yield (batch_y, batch_x)

def get_val_data(batch_size, H_stack, device, config):
    input_labels = []
    input_samples = []
    for row in range(batch_size):
        bits0 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
        bits1 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
        X = [bits0, bits1]
        H_ind = np.random.randint(0, len(H_stack))
        HH = H_stack[H_ind]
        mode = config.OFDM.mode if config.OFDM.mode != -1 else np.random.choice(modes, 1)[0]
        SNRdb = config.OFDM.SNRdb if config.OFDM.SNRdb != -1 else np.random.choice(SNRdbs, 1)[0]
        Pilotnum = config.OFDM.Pilotnum if config.OFDM.Pilotnum != -1 else np.random.choice(Pilotnums, 1)[0]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20
        # YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        XX = np.concatenate((bits0[:config.model.out_dim // 2], bits1[:config.model.out_dim // 2]), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = torch.Tensor(np.asarray(input_samples)).to(device)
    batch_x = torch.Tensor(np.asarray(input_labels)).to(device)
    return (batch_y, batch_x)

