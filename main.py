import logging
import yaml
import os 
from datetime import datetime
from argparse import Namespace
from utils import set_logger, seed_everything, get_config, arg_parser
from multiprocessing import Pool

from runners import Y2HRunner, FullRunner,  EMAY2HRunner, SDCERunner

def run_y2h(args):
    config = get_config(f'./configs/y2h_config_{args.run_mode}.yml')
    config.model = args.run_mode
    if config.model == 'fc':
        config.log_prefix = f'workspace/ResnetY2HEstimator/mode_{config.mode}_Pn_{config.Pn}/FC'
    elif config.model == 'cnn':
        config.log_prefix = f'workspace/ResnetY2HEstimator/mode_{config.mode}_Pn_{config.Pn}/CNN'
    config.log_dir = os.path.join(config.log_prefix, args.time)
    config.ckpt_dir = os.path.join(config.log_dir, 'checkpoints')
    if not os.path.isdir(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2HRunner(config)
    runner.run()

def run_sdce(args):
    config = get_config(f'./configs/y2h_config_cesd.yml')
    config.log_prefix = f'workspace/ResnetY2HEstimator/mode_{config.mode}_Pn_{config.Pn}/SDCE'
    config.log_dir = os.path.join(config.log_prefix, args.time)
    config.ckpt_dir = os.path.join(config.log_dir, 'checkpoints')
    if not os.path.isdir(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = SDCERunner(config)
    runner.run()

def run_ema(args):
    config = get_config(f'./configs/y2h_config_ema.yml')
    assert config.model == 'ema'
    config.log_prefix = f'workspace/ResnetY2HEstimator/mode_{config.mode}_Pn_{config.Pn}/EMA'
    config.log_dir = os.path.join(config.log_prefix, args.time)
    config.ckpt_dir = os.path.join(config.log_dir, 'checkpoints')
    if not os.path.isdir(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = EMAY2HRunner(config)
    runner.run()

def run_full(args):
    config = get_config('./configs/full_config.yml')
    config.run_mode = args.runner
    config.Pn = args.Pn
    config.log_dir = os.path.join('workspace', config.run_mode, f'mode_{config.mode}_Pn_{config.Pn}', args.time)
    if not os.path.isdir(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = FullRunner(config)
    runner.run()

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    if args.runner == 'y2h':
        run_y2h(args)
    elif args.runner == 'validation':
        run_full(args)
    elif args.runner == 'testing':
        run_full(args)
    elif args.runner == 'ema':
        run_ema(args)
    elif args.runner == 'sdce':
        run_sdce(args)
    else:
        raise NotImplementedError