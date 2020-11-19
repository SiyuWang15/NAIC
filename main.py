import logging
import yaml
import os 
from datetime import datetime
from argparse import Namespace
from utils import set_logger, seed_everything, get_config, arg_parser
from multiprocessing import Pool

from runners import Y2HRunner, FullRunner

def run_y2h(args):
    config = get_config(f'./configs/y2h_config_{args.run_mode}.yml')
    if config.model == 'fc':
        config.log_prefix = f'workspace/ResnetY2HEstimator/mode_{config.mode}_Pn_{config.Pn}/FC'
    elif config.model == 'cnn':
        config.log_prefix = f'workspace/ResnetY2HEstimator/mode_{config.mode}_Pn_{config.Pn}/CNN'
    config.log_dir = os.path.join(config.log_prefix, args.time)
    config.ckpt_dir = os.path.join(config.log_dir, 'checkpoints')
    os.makedirs(config.ckpt_dir)
    set_logger(config)
    logging.info(config)
    runner = Y2HRunner(config)
    runner.run()


def run_full(args):
    config = get_config('./configs/full_config.yml')
    config.run_mode = args.runner
    config.Pn = args.Pn
    config.log_dir = os.path.join('workspace', config.run_mode, f'mode_{config.mode}_Pn_{config.Pn}', args.time)
    os.makedirs(config.log_dir)
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