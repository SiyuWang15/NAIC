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
    config = yaml.safe_load(open(fp))
    for k, v in config.items():
        config[k] = argparse.Namespace(**v)
    config = argparse.Namespace(**config)
    return config

def to_numpy(x):
    if not (type(x) is np.ndarray):
        return np.array(x)
    else:
        return x

def flattern_Y(Y): # Nsx2x2x2x256 or Nsx2x2x256
    # for 2x2x2x256, imag or real, Yp and Yd, 2, 256
    # for 2x2x256 (only Yp or Yd), imag or real, 2, 256
    YY = to_numpy(Y)
    YY = np.reshape(YY, [len(YY), -1], order = 'F')
    assert YY.shape[-1] == 2048 or YY.shape[-1] == 1024
    return YY

def transfer_Y(Y): # Nsx2048 or Nsx1024  
    # return YY: Nsx2x256 complex number can be fed into MLReceiver
    # return YY: Nsx2x2x256 complex number should extract data first and then feed into Receiver
    YY = to_numpy(Y)
    assert YY.shape[-1] == 2048 or YY.shape[-1] == 1024
    if YY.shape[-1] == 2048:
        YY = np.reshape(YY, [-1, 2, 2, 2, 256], order = 'F') 
        # print('Yp and Yd are all here.')
        return real2complex(YY, 'Y')
    if YY.shape[-1] == 1024:
        YY = np.reshape(YY, [-1, 2, 2, 256], order = 'F')
        # print('only Yp or Yd')
        return real2complex(YY, 'Yd')

def flattern_H(H):  # only for time domain
    # H: Nsx2x4x32 imag or real, 4 routes, 32 dimension in time domain
    # This is only for siyu's Hdata since siyu didn't reshape H via order 'F'
    HH = to_numpy(H)
    return np.reshape(HH, [len(HH), -1])


def transfer_H(H):
    # H: Nsx256 complex number 
    # For siyu's Hdata
    # transfer time domain H which can be fed into MIMO  TOOOOO frequency domain H which can be fed into MLReceiver. 
    HH = to_numpy(H)
    HH = np.reshape(HH, [len(HH), 2, 4, 32])
    HH = real2complex(HH, 'H')
    Hf = np.fft.fft(HH, 256) / 20.
    Hf = np.reshape(Hf, (len(HH), 2, 2, 256), order = 'F')
    return Hf

def real2complex(D, tag):
    # D: Yp&Yd: Nsx2x2x256 or Y: Nsx2x2x2x256 or H: Nsx2x4x32
    if tag == 'H':
        return D[:, 1, :, :] + 1j * D[:, 0, :, :]
    if tag == 'Yp' or tag == 'Yd':
        return D[:, 0, :, :] + 1j * D[:, 1, :, :]
    if tag == 'Y':
        return D[:, 0, :, :, :] + 1j * D[:, 1, :, :, :]