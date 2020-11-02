import numpy as np 
import torch
from H_utils import *
import struct

Pilotnums = [32, 8]
modes = [0,1,2]
SNRdbs = [8, 9, 10, 11, 12]
# H_ind = 1

def get_H(H_path = 'dataset/H_data.npy', ratio = 0.9):
    H = np.load(H_path)
    split = int(ratio * len(H))
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
            XX = np.concatenate((bits0, bits1), 0)[:config.model.out_dim]
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
        XX = np.concatenate((bits0, bits1), 0)[:config.model.out_dim]
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = torch.Tensor(np.asarray(input_samples)).to(device)
    batch_x = torch.Tensor(np.asarray(input_labels)).to(device)
    return (batch_y, batch_x)

def make_X():
    X = np.random.binomial(n=1, p=0.5, size = (320000, 1024))
    X = np.asarray(X, dtype = bool)
    np.save('dataset/X_bin.npy', X)

def get_train_data():
    X = np.load('dataset/X_bin.npy')
    H_ind = 0


if __name__ == "__main__":
    make_X()

# def make_dataset(Pilotnum = 32):
#     H_train, H_val = get_H()
#     X_train = []
#     Y_train = []
#     X_val = []
#     Y_val = []
#     for i in range(len(H_train)):
#         bits0 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
#         bits1 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
#         X = [bits0, bits1]
#         H = H_train[i]
#         Y = MIMO(X, H, 12, 0, Pilotnum) / 20
#         X = np.concatenate((bits0, bits1), 0)
#         X_train.append(X)
#         Y_train.append(Y)
#         if (i+1) % 10000 == 0:
#             print(i+1)
#     X_train = np.asarray(X_train, dtype = bool)
#     Y_train = np.asarray(Y_train)
#     np.save('dataset/train{}.npy'.format(Pilotnum), {'X': X_train, 'Y': Y_train})
#     print('train dataset saved!')
#     for i in range(len(H_val)):
#         bits0 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
#         bits1 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
#         X = [bits0, bits1]
#         H = H_val[i]
#         Y = MIMO(X, H, 12, 0, Pilotnum) / 20
#         X = np.concatenate((bits0, bits1), 0)
#         X_val.append(X)
#         Y_val.append(Y)
#         if (i+1) % 1000 == 0:
#             print(i+1)
#     X_val = np.asarray(X_val, dtype = bool)
#     Y_val = np.asarray(Y_val)
#     np.save('dataset/val{}.npy'.format(Pilotnum), {'X': X_val, 'Y': Y_val})
#     print('val dataset saved!')