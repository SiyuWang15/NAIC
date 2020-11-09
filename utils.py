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