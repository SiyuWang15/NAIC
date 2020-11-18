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
    log = os.path.join(config.log_dir, 'training.log')
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
    parser.add_argument('--mode', default = 0, type=int)
    parser.add_argument('--Pn', default = 32, type=int)
    parser.add_argument('--time', default = 'default', type=str)
    parser.add_argument('--runner', default = 'y2h', type = str)
    parser.add_argument('--run_mode', default = 'none', type = str)
    return parser

def get_config(fp):
    config = yaml.safe_load(open(fp))
    for k, v in config.items():
        if type(v) is dict:
            config[k] = argparse.Namespace(**v)
    config = argparse.Namespace(**config)
    return config

def to_numpy(x):
    if not (type(x) is np.ndarray):
        return np.array(x)
    else:
        return x


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
    

def process_H(H): # bs x 2 x 4 x 32
    # 真实的频域信道，获取标签
    batch_size = len(H)
    Hf_train = np.array(H)[:, 0, :] + 1j * np.array(H)[:, 1, :]
    Hf_train = np.fft.fft(Hf_train, 256)/20 # 4*256
    Hf_train_label = torch.zeros([batch_size, 2, 4, 256], dtype=torch.float32)
    Hf_train_label[:, 0, :, :] = torch.tensor(Hf_train.real, dtype = torch.float32)
    Hf_train_label[:, 1, :, :] = torch.tensor(Hf_train.imag, dtype = torch.float32)
    return Hf_train_label

def process_Y(Y): # bsx2x2x256
    return real2complex(Y, 'Yd')