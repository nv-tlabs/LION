# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import time
import os
import numpy as np
from loguru import logger
from math import isnan
from calmsize import size as calmsize


def parse_cfg_str(cfg_str):
    """ parse a string into a dict 
    string format: k1=v1,k2=v2,k3=v3
    """
    cfg_list = cfg_str.split('-')
    cfg_list = [c for c in cfg_list if len(c) > 0]
    cfg_expand_list = []
    for c in cfg_list:
        k, v = c.split('=')
        cfg_expand_list.append(k)
        cfg_expand_list.append(v)
    return cfg_expand_list

    ##cfg_dict = {}
    # if cfg_str == '':
    # return cfg_dict
    ##cfg_str_list = cfg_str.split(',')
    # for p in cfg_str_list:
    ##    kvs = p.split('=')
    ##    assert(len(kvs) == 2), f'wrong format, expect k1=v1 for {p}'
    ##    k, v = kvs
    ##    cfg_dict[k] = v
    # return cfg_dict


def readable_size(num_bytes: int) -> str:
    return '' if isnan(num_bytes) else '{:.1f}'.format(calmsize(num_bytes))


class ExpTimer(object):
    def __init__(self, num_epoch, start_epoch=0):
        self.cur_epoch = start_epoch
        self.num_epoch = num_epoch
        self.time_list = []

    def tic(self):
        self.last_tic = time.time()

    def toc(self):
        self.time_list.append(time.time() - self.last_tic)
        self.cur_epoch += 1

    def hours_left(self):
        if len(self.time_list) == 0:
            return 0
        num_epoch_left = self.num_epoch - self.cur_epoch
        mean_epoch_time = np.array(self.time_list).mean()
        hours_left = (mean_epoch_time * num_epoch_left) / 3600.0  # hours
        return hours_left

    def print(self):
        logger.info('est: {:.1}h', self.hours_left)


def format_e(n):
    if n == 0:
        return '0'
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E-0')[1]


def get_evalname(config):
    # generate tag for the generated samples
    tag = ''
    if config.ddpm.model_var_type != 'fixedlarge':
        tag += config.ddpm.model_var_type

    if not config.ddpm.ema:
        tag += 'noema'
    tag += f"s{config.trainer.seed}"
    if config.data.te_max_sample_points != 2048:
        tag += 'N%d' % config.data.te_max_sample_points
    if config.eval_ddim_step > 0:
        tag += 'ddim%d_%s%.1f' % (
            config.eval_ddim_step,
            config.sde.ddim_skip_type,
            config.sde.ddim_kappa)
    githash = os.popen('git rev-parse HEAD').read().strip()[:5]
    logger.info('git hash: {}', githash)
    tag += f"H{githash}"
    return tag


def get_expname(config):
    if config.exp_name == '' or config.exp_name == 'none':
        cate = config.data.cates if type(
            config.data.cates) is str else config.data.cates[0]
        cfg_file_name = ''
        if config.data.type == 'datasets.neuralspline_datasets':
            cfg_file_name += 'ns'
        cfg_file_name += '%s/' % cate
        if len(config.hash):
            cfg_file_name += '%s_' % config.hash
        cfg_file_name += f"{config.trainer.type.split('.')[-1].split('_')[0]}_"
        if len(config.cmt):
            cfg_file_name += config.cmt + '_'

        cfg_file_name += 'B%d' % config.data.batch_size

        if config.data.tr_max_sample_points != 2048:
            cfg_file_name += 'N%d' % config.data.tr_max_sample_points
        run_time = time.strftime('%m%d')
        cfg_file_name = run_time + '/' + cfg_file_name

    else:
        cfg_file_name = config.exp_name
    return cfg_file_name
