import logging
import yaml
import os 
from datetime import datetime
from argparse import Namespace
from utils import set_logger, seed_everything, get_config, arg_parser
from multiprocessing import Pool

from runners import Y2HRunner, FullRunner

def run_y2h(args):
    config = get_config('./configs/y2h_config.yml')
    config.OFDM.mode = args.mode
    config.OFDM.Pilotnum = args.Pn
    config.log.log_prefix = f'workspace/MLPY2HEstimator/{args.log_prefix}'
    config.log.log_dir = os.path.join(config.log.log_prefix, f'mode_{args.mode}_Pn_{args.Pn}')
    config.log.ckpt_dir = os.path.join(config.log.log_dir, 'checkpoints')
    os.makedirs(config.log.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2HRunner(config, cnn=False)
    runner.run()

def run_y2h_cnn(args):
    config = get_config('./configs/y2h_cnn_config.yml')
    config.OFDM.mode = args.mode
    config.OFDM.Pilotnum = args.Pn
    config.log.log_prefix = f'workspace/CNNY2HEstimator/{args.log_prefix}'
    config.log.log_dir = os.path.join(config.log.log_prefix, f'mode_{args.mode}_Pn_{args.Pn}')
    config.log.ckpt_dir = os.path.join(config.log.log_dir, 'checkpoints')
    os.makedirs(config.log.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2HRunner(config, cnn=True)
    runner.run()

def run_y2mode(args):
    config = get_config('./configs/y2mode_config.yml')
    config.OFDM.Pilotnum = args.Pn
    config.log.log_prefix = f'workspace/ModeEstimator/{args.log_prefix}'
    config.log.log_dir = os.path.join(config.log.log_prefix, f'Pn_{args.Pn}')
    config.log.ckpt_dir = os.path.join(config.log.log_dir, 'checkpoints')
    os.makedirs(config.log.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2ModeRunner(config)
    runner.run()

def run_y2x(args):
    config = get_config('./configs/y2x_config.yml')
    config.OFDM.Pilotnum = args.Pn
    config.log.log_prefix = f'workspace/Y2XEstimator/{args.log_prefix}'
    config.log.log_dir = os.path.join(config.log.log_prefix, f'mode_{args.mode}_Pn_{args.Pn}')
    config.log.ckpt_dir = os.path.join(config.log.log_dir, 'checkpoints')
    os.makedirs(config.log.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2XRunner(config)
    runner.run()


def run_full(args):
    config = get_config('./configs/full_config.yml')
    config.RE.Pilotnum = args.Pn
    config.log.run_mode = args.runner # validation or testing
    config.log.log_dir = os.path.join('workspace', config.log.run_mode, args.log_prefix, f'Pn_{args.Pn}')
    os.makedirs(config.log.log_dir)
    set_logger(config)
    logging.info(config)
    if config.log.run_mode == 'validation':
        # logging.info(f'validating on Pn={args.Pn}')
        runner = FullRunner(config)
        runner.simple_run()
    elif config.log.run_mode == 'testing':
        logging.info(f'testing on Pn={args.Pn}')
        runner = FullRunner(config)
        runner.simple_test()

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    if args.runner == 'y2mode':
        run_y2mode(args)
    elif args.runner == 'y2h':
        run_y2h(args)
    elif args.runner == 'y2hcnn':
        run_y2h_cnn(args)
    elif args.runner == 'validation':
        run_full(args)
    elif args.runner == 'testing':
        run_full(args)
    elif args.runner == 'y2x':
        run_y2x(args)