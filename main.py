import logging
import yaml
import os 
from datetime import datetime
from argparse import Namespace
from utils import set_logger, seed_everything, get_config

from runners import Y2HRunner, Y2XRunner, Y2ModeRunner, FullRunner

def run_y2h():
    config = get_config('./configs/y2h_config.yml')
    now = datetime.now()
    config.log.log_prefix = 'Y2HEstimator/mode_{}_Pn_{}'.format(config.OFDM.mode, config.OFDM.Pilotnum)
    config.log.log_dir = os.path.join('workspace', config.log.log_prefix, now.strftime('%H-%M-%S'))
    config.log.ckpt_dir = os.path.join(config.log.log_dir, 'checkpoints')
    os.makedirs(config.log.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2HRunner(config)
    runner.run()

def run_y2mode():
    config = get_config('./configs/y2mode_config.yml')
    now = datetime.now()
    config.log.log_prefix = 'Mode_estimator/Pn_{}'.format(config.OFDM.Pilotnum)
    config.log.log_dir = os.path.join('workspace', config.log.log_prefix, now.strftime('%H-%M-%S'))
    config.log.ckpt_dir = config.log.log_dir
    os.makedirs(config.log.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2ModeRunner(config)
    runner.run()

def run_full():
    config = get_config('./configs/full_config.yml')
    now = datetime.now()
    config.log_prefix = 'Full_runner'
    config.log_dir = os.path.join('workspace', config.log_prefix, now.strftime('%H-%M-%S'))
    os.makedirs(config.log_dir)
    runner = FullRunner(config)
    runner.validation()

if __name__ == "__main__":
    run_full()