import logging
import numpy as np 
from numpy import random
import torch
import os 
import yaml
import shutil
from datetime import datetime
import argparse

# logging
def set_logger(config):
    log = os.path.join(config.log.log_dir, 'training.log')
    level = getattr(logging, 'INFO', None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(log)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
    logging.info('log at: {}'.format(log))

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def bit_err(x_true, x_prob):
    x_pred = x_prob > 0.5
    err = 1 - np.asarray((x_pred == x_true).cpu(), dtype = 'float32').mean()
    return err

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice', default=0, type = int, help='use which slice of X as label')
    return parser

def get_config(fp):
    config = yaml.load(open(fp))
    for k, v in config.items():
        config[k] = argparse.Namespace(**v)
    config = argparse.Namespace(**config)
    return config

def to_numpy(x):
    if not (type(x) is np.ndarray):
        return np.array(x)

def flattern_Y(Y): # Nsx2x2x2x256 or Nsx2x2x256
    # for 2x2x2x256, imag or real, Yp and Yd, 2, 256
    # for 2x2x256 (only Yp or Yd), imag or real, 2, 256
    YY = to_numpy(Y)
    YY = np.reshape(YY, [len(YY), -1], order = 'F')
    assert YY.shape[-1] == 2048 or YY.shape[-1] == 1024
    return YY

def fuck_Y(Y): # Nsx2048 or Nsx1024
    YY = to_numpy(Y)
    assert YY.shape[-1] == 2048 or YY.shape[-1] == 1024
    if YY.shape[-1] == 2048:
        YY = np.reshape(YY, [-1, 2, 2, 2, 256], order = 'F') 
        print('Yp and Yd are all here.')
        return YY
    if YY.shape[-1] == 1024:
        YY = np.reshape(YY, [-1, 2, 2, 256], order = 'F')
        print('only Yp or Yd')
        return YY

def flatter_H(H):  # only for time domain
    # H: Nsx2x4x32 imag or real, 4 routes, 32 dimension in time domain
    HH = to_numpy(H)
    return np.reshape(HH, [len(HH), -1])


def fuck_H(H): # only for time domain
    # H: Nsx256
    HH = to_numpy(H)
    return np.reshape(HH, [len(HH), 2, 4, 32])

def FFT_H(H_time):
    H_freq = np.reshape(H_time, [len(H_time), 2, 2, 2, 32]) # imag or real, sender, receiver, 32 dimension in time domain
    H_freq = H_freq[:, 1, :, :, :] + H_freq[:, 0, :, :, :] * 1j
    H_freq = np.fft.fft(H_freq, 256) / 20
    
    pass

def IFFT_H(H_freq):
    pass