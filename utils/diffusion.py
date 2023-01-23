# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
"""
copied and modified from source:
    https://github.com/NVlabs/LSGM/blob/5eae2f385c014f2250c3130152b6be711f6a3a5a/diffusion_discretized.py
"""
from loguru import logger
import time
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np


def extract(input, t, shape):
    B = t.shape[0]
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def make_beta_schedule(schedule, start, end, n_timestep):
    if schedule == "cust":  # airplane
        b_start = start
        b_end = end
        time_num = n_timestep
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(
            b_start, b_end, warmup_time, dtype=np.float64)
        betas = torch.from_numpy(betas)

        #betas = torch.zeros(n_timestep, dtype=torch.float64) + end
        #n_timestep_90 = int(n_timestep*0.9)
        # betas_0 = torch.linspace(start,
        #                       end,
        #                       n_timestep_90,
        #                       dtype=torch.float64)
        #betas[:n_timestep_90] = betas_0

    elif schedule == "quad":
        betas = torch.linspace(start**0.5,
                               end**0.5,
                               n_timestep,
                               dtype=torch.float64)**2
    elif schedule == 'linear':
        betas = torch.linspace(start, end, n_timestep, dtype=torch.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(start, end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(start, end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = end * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / (torch.linspace(
            n_timestep, 1, n_timestep, dtype=torch.float64))
    else:
        raise NotImplementedError(schedule)
    return betas


class VarianceSchedule(Module):
    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', 'cust')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode
        beta_start = self.beta_1
        beta_end = self.beta_T
        assert (beta_start <= beta_end), 'require beta_start < beta_end '

        logger.info('use beta: {} - {}', beta_1, beta_T)
        tic = time.time()
        # betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        betas = make_beta_schedule(mode, beta_start, beta_end, num_steps)
        # elif mode == 'customer':
        # beta_0 = 10−5 and beta_T = 0.008 for 90% step, beta_T=0.0088
        ##     num_steps_90 = int(0.9 * num_steps)
        # logger.info('use beta_0=1e-5 and beta_T=0.008 '
        ##         'for {} step and 0.008 for the rest',
        # num_steps_90)
        ##     betas_sub = torch.linspace(1e-5, 0.008, steps=num_steps_90)
        ##     betas_full = torch.zeros(num_steps) + 0.008
        ##     betas_full[:num_steps_90] = betas_sub
        ##     betas = betas_full

        # betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        #alphas = 1 - betas
        #log_alphas = torch.log(alphas)
        # for i in range(1, log_alphas.size(0)):  # 1 to T
        #    log_alphas[i] += log_alphas[i - 1]
        #alpha_bars = log_alphas.exp()

        #sigmas_flex = torch.sqrt(betas)
        #sigmas_inflex = torch.zeros_like(sigmas_flex)
        # for i in range(1, sigmas_flex.size(0)):
        #    sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        #sigmas_inflex = torch.sqrt(sigmas_inflex)
        #sqrt_recip_alphas_cumprod = torch.rsqrt(alpha_bars)
        #sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / alpha_bars - 1)

        #self.register_buffer('betas', betas)
        #self.register_buffer('alphas', alphas)
        #self.register_buffer('alpha_bars', alpha_bars)
        #self.register_buffer('sigmas_flex', sigmas_flex)
        #self.register_buffer('sigmas_inflex', sigmas_inflex)
        #self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        # self.register_buffer('sqrt_recipm1_alphas_cumprod',
        #    sqrt_recipm1_alphas_cumprod)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (
            1 - alphas_cumprod)
        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod",
                      torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod",
                      torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod",
                      torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        if len(posterior_variance) > 1:
            self.register("posterior_log_variance_clipped",
                          torch.log(
                              torch.cat((posterior_variance[1].view(
                                  1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)
                          )
        else:
            self.register("posterior_log_variance_clipped",
                          torch.log(posterior_variance[0].view(-1)))
        self.register("posterior_mean_coef1",
                      (betas * torch.sqrt(alphas_cumprod_prev) /
                       (1 - alphas_cumprod)))
        self.register("posterior_mean_coef2",
                      ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) /
                       (1 - alphas_cumprod)))
        logger.info('built beta schedule: t={:.2f}s', time.time() - tic)

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def all_sample_t(self):
        if self.num_steps > 20:
            step = 50
        else:
            step = 1
        ts = np.arange(0, self.num_steps, step)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (
            1 - flexibility)
        return sigmas
