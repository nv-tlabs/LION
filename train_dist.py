# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# ---------------------------------------------------------------

import importlib
import argparse
from loguru import logger
from comet_ml import Experiment
import torch
import numpy as np
import os
import sys
import torch.distributed as dist
from torch.multiprocessing import Process
from default_config import cfg as config
from utils import exp_helper, io_helper
from utils import utils


@logger.catch(onerror=lambda _: sys.exit(1), reraise=False)
def main(args, config):
    # -- trainer -- #
    logger.info('use trainer: {}', config.trainer.type)
    trainer_lib = importlib.import_module(config.trainer.type)
    Trainer = trainer_lib.Trainer

    if config.set_detect_anomaly:
        # attention: this makes thing slow
        torch.autograd.set_detect_anomaly(True)
        logger.info(
            '\n\n' + '!'*30 + '\nWARNING: ths set_detect_anomaly is turned on, it can slow down the training! \n' + '!'*30)

    # -- command init -- #
    comet_key = config.comet_key
    _, writer = utils.common_init(args.global_rank,
                                  config.trainer.seed, config.save_dir, comet_key)
    trainer = Trainer(config, args)
    writer.add_hparams(config.to_dict(), vars(args))
    nparam = utils.count_parameters_in_M(trainer.model)
    logger.info('param size = %fM ' % nparam)
    writer.log_other('nparam', nparam)

    if args.global_rank == 0:
        trainer.set_writer(writer)
        writer.set_model_graph('{}'.format(trainer.model), overwrite=True)
        if len(config.bash_name) > 0 and os.path.exists(config.bash_name):
            writer.log_asset(config.bash_name)
        if len(config.bash_name) > 0 and os.path.exists(os.path.join(config.save_dir, config.bash_name.split('/')[-1])):
            writer.log_asset(os.path.join(
                config.save_dir, config.bash_name.split('/')[-1]))
    ckpt_dir = os.path.join(config.save_dir, 'checkpoints')
    snapshot_file = os.path.join(config.save_dir, 'checkpoints', 'snapshot')

    # -- check if prev saved ckpt exist -- #
    if os.path.exists(ckpt_dir) and os.path.exists(snapshot_file):
        logger.info(
            '[Detect saved snapshot at the checkpoint dir] resume from preemption!!! ')
        args.resume = True
        args.pretrained = os.path.join(
            config.save_dir, 'checkpoints', 'snapshot')
    else:
        logger.info('not find any checkpoint: {}, (exist={}), or snapshot {}, (exist={})',
                    ckpt_dir, os.path.exists(ckpt_dir), snapshot_file, os.path.exists(snapshot_file))

    # -- prepare -- #
    if args.resume or args.eval_generation:
        if args.pretrained is not None:
            trainer.start_epoch = trainer.resume(
                args.pretrained, eval_generation=args.eval_generation)
        else:
            raise NotImplementedError
    elif args.pretrained is not None:
        logger.info('Resuming training from {}; if you dont want resume training, edit the cmt to change the exp name',
                args.pretrained)
        trainer.resume(args.pretrained)

    if not args.eval_generation:
        trainer.train_epochs()
    else:
        logger.info('[skip_sample]={}', args.skip_sample)

        save_file = None
        if not args.skip_nll:
            trainer.eval_nll(trainer.step, ntest=args.ntest, save_file=True)
        logger.info('save as : {}', save_file)
        # vis sampled output
        if not args.skip_sample:
            trainer.vis_sample(num_vis=8, writer=trainer.writer,
                               step=trainer.step, include_pred_x0=False,
                               save_file=save_file)
            trainer.eval_sample(trainer.step)
        logger.info('done')

    # make all nodes wait for rank 0 to finish saving the files
    # if args.distributed:
    #    dist.barrier()


def get_args():
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--exp_root', type=str, default='../exp',
                        help='location of the results')
    # parser.add_argument('--save', type=str, default='exp',
    #                     help='id used for storing intermediate results')
    # parser.add_argument('--recont_with_local_prior', type=bool, default=False,
    #                    help='eval nll with local prior sampled from normal distribution')
    parser.add_argument('--skip_sample', type=int, default=0,
                        help='only eval nll, no sampling')
    parser.add_argument('--skip_nll', type=int, default=0,
                        help='skip eval nll ')
    # data
    parser.add_argument('--ntest', type=str, default=None,
                        help='number of samples in eval_nll, if None, eval the whole val set')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nvae-diff/data',
                        help='location of the data corpus')
    # DDP.
    parser.add_argument('--autocast_train', action='store_true', default=True,
                        help='This flag enables FP16 in training.')
    parser.add_argument('--autocast_eval', action='store_true', default=True,
                        help='This flag enables FP16 in evaluation.')
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--config', type=str,
                        help='The configuration file.', default='none')
    parser.add_argument("opt",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--eval_generation',
                        default=False, action='store_true')
    parser.add_argument('--pretrained',
                        default=None,
                        type=str,
                        help="Pretrained cehckpoint")

    args = parser.parse_args()

    # update config
    if args.eval_generation or args.resume:
        logger.info('[pretrained]: {}', args.pretrained)
        args.config = os.path.dirname(args.pretrained) + '/../cfg.yml'
        config.merge_from_file(args.config)
    elif args.config != 'none':
        logger.info('load config: {}', args.config)
        cur_exp_name = config.exp_name
        cur_hash = config.hash

        config.merge_from_file(args.config)
        config.exp_name = cur_exp_name  # not following the exp name here
        config.hash = cur_hash  # not following the exp name here
    config.merge_from_list(args.opt)

    # Create log_name
    EXP_ROOT = args.exp_root  # os.environ.get('EXP_ROOT', '../exp/')
    if config.exp_name == '' or config.exp_name == 'none':
        config.hash = io_helper.hash_str('%s' % config) + 'h'
        cfg_file_name = exp_helper.get_expname(config)
    else:
        cfg_file_name = config.exp_name

    # Currently save dir and log_dir are the same
    if args.eval_generation:
        config.save_dir = config.log_dir = config.log_name = os.path.dirname(
            args.config)
        if config.trainer.type == 'ddim':
            tag = 'eval_ddim'
        else:
            tag = 'eval'
        cfg_file_name += f'/{tag}/'
        config.log_name += f'/{tag}/'
        config.save_dir += f'/{tag}/'
        config.log_dir += f'/{tag}/'
    else:
        config.log_name = os.path.join(EXP_ROOT, cfg_file_name)
        config.save_dir = os.path.join(EXP_ROOT, cfg_file_name)
        config.log_dir = os.path.join(EXP_ROOT, cfg_file_name)
    os.makedirs(config.log_dir, exist_ok=True)

    # save config and log
    if args.global_rank == 0 and not args.eval_generation:
        logger.add(config.log_dir + '/train.log')
        logger.info('EXP_ROOT: {} + exp name: {}, save dir: {}', EXP_ROOT,
                    cfg_file_name, config.save_dir)
        saved_cfg = os.path.join(config.log_dir, 'cfg.yml')
        with open(saved_cfg, 'w') as file:
            file.write(config.dump())
        logger.info('save config at {}', saved_cfg)
    elif args.eval_generation:
        logger.add(config.log_dir + '/eval_gen.log')
    logger.info('log dir: {}', config.log_dir)

    return args, config


if __name__ == '__main__':
    args, config = get_args()
    args.ntest = int(args.ntest) if args.ntest is not None else None
    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            logger.info('In Rank={}', rank)
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_size = global_size
            args.global_rank = global_rank
            logger.info('Node rank %d, local proc %d, global proc %d' %
                        (args.node_rank, rank, global_rank))
            p = Process(target=utils.init_processes,
                        args=(global_rank, global_size, main, args, config))
            p.start()
            processes.append(p)

        for p in processes:
            logger.info('join {}', args.local_rank)
            p.join()
    else:
        # for debugging
        args.distributed = False
        args.global_size = 1
        utils.init_processes(0, size, main, args, config)
    logger.info('should end now')
    # if args.distributed:
    #    logger.info('destroy_process_group')
    #    dist.destroy_process_group()
