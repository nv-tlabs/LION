# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" src: ddim/model/ema.py 
implement the EMA model 
usage: 
    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
    ema_helper.register(model)
    ema_helper.load_state_dict(states[-1])
    ema_helper.ema(model)

after optimizer.step()
    ema_helper.update(model)

copied and modified from 
    https://github.com/NVlabs/LSGM/blob/5eae2f385c014f2250c3130152b6be711f6a3a5a/util/ema.py
"""

import warnings
import torch
from torch.optim import Optimizer
from loguru import logger
import torch.nn as nn
import os


class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        logger.info('[EMA] apply={}', self.apply_ema)
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        for group in self.optimizer.param_groups:
            ema, params = {}, {}
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(
                    params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        # logger.info('state size: {}', len(self.state))
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """ This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            warnings.warn(
                'swap_parameters_with_ema was called when there is no EMA weights.')
            return
        logger.info('swap with ema')
        count_no_found = 0
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    # logger.info('no swap for i={}, param shape={}', i, p.shape)
                    continue
                if p not in self.optimizer.state:
                    count_no_found += 1
                    # logger.info('no found i={}, {}/{} p {}', i,
                    #            count_no_found, len(group['params']), p.shape)
                    continue
                # if count_no_found > 100:
                #    logger.info('found: i={}, p={}', i, p.shape)
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data = ema.detach()
