#!/usr/bin/env python

import os
import os.path as osp
import time
import torch

import click
import mmcv

from mmcv import Config
from mmcv.utils import get_logger
from mmdet.utils import collect_env
from mmdet.apis import (
    inference_detector,
    init_detector,
    show_result_pyplot,
    init_random_seed,
    set_random_seed,
)

from neudc.apis import 


@click.command()
@click.option(
    '--config_path',
    '-c',
    default='configs/config.py',
    type=str,
    help='Input config path.',
)
@click.option(
    '--save_images',
    '-s',
    default=False,
    type=bool,
    is_flag=True,
    help='Save images from video or not',
)
@click.option(
    '--debug',
    '-d',
    default=False,
    type=bool,
    is_flag=True,
    help='Draw predicted boxes on images or not',
)
def main(
    config_path: str,
    save_images: bool,
    debug: bool,
) -> None:

    # read a config
    cfg = Config.fromfile(config_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if cfg.get('launcher', None) is None:
        distributed = False
    
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.dataset.output_path))
    # dump config
    cfg.dump(osp.join(cfg.dataset.output_path, osp.basename(config_path)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.dataset.output_path, f'{timestamp}.log')
    logger = get_logger(
        log_file=log_file,
        name='mmdet',
    )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(cfg.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {cfg.deterministic}')
    set_random_seed(seed, deterministic=cfg.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config_path)

    # build the model
    model = init_detector(
        cfg.model.mmconfig,
        cfg.model.mmcheckpoint,
        cfg.model.device,
    )

    # Convert the model into evaluation mode
    model.eval()

    logger.info(f'Model:\n{model}')

    


if __name__ == '__main__':
    main()
