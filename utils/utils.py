# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
"""copied and modified from https://github.com/NVlabs/LSGM/blob/5eae2f385c014f2250c3130152b6be711f6a3a5a/util/utils.py"""
from loguru import logger
from comet_ml import Experiment, ExistingExperiment
import wandb as WB
import os
import math
import shutil
import json
import time
import sys
import types
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
USE_COMET = int(os.environ.get('USE_COMET', 1))
USE_TFB = int(os.environ.get('USE_TFB', 0))
USE_WB = int(os.environ.get('USE_WB', 0))
print(f'utils/utils.py: USE_COMET={USE_COMET}, USE_WB={USE_WB}')

class PixelNormal(object):
    def __init__(self, param, fixed_log_scales=None):
        size = param.size()
        C = size[1]
        if fixed_log_scales is None:
            self.num_c = C // 2
            # B, 1 or 3, H, W
            self.means = param[:, :self.num_c, :, :]
            self.log_scales = torch.clamp(
                param[:, self.num_c:, :, :], min=-7.0)        # B, 1 or 3, H, W
            raise NotImplementedError
        else:
            self.num_c = C
            # B, 1 or 3, H, W
            self.means = param
            # B, 1 or 3, H, W
            self.log_scales = view4D(fixed_log_scales, size)

    def get_params(self):
        return self.means, self.log_scales, self.num_c

    def log_prob(self, samples):
        B, C, H, W = samples.size()
        assert C == self.num_c

        log_probs = -0.5 * torch.square(self.means - samples) * torch.exp(-2.0 *
                                                                          self.log_scales) - self.log_scales - 0.9189385332  # -0.5*log(2*pi)
        return log_probs

    def sample(self, t=1.):
        z, rho = sample_normal_jit(
            self.means, torch.exp(self.log_scales)*t)  # B, 3, H, W
        return z

    def log_prob_discrete(self, samples):
        """
        Calculates discrete pixel probabilities.
        """
        # samples should be in [-1, 1] already
        B, C, H, W = samples.size()
        assert C == self.num_c

        centered = samples - self.means
        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = torch.distributions.Normal(0, 1).cdf(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = torch.distributions.Normal(0, 1).cdf(min_in)
        log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
        log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.999, log_one_minus_cdf_min,
                                                                            torch.log(torch.clamp(cdf_delta, min=1e-12))))

        assert log_probs.size() == samples.size()
        return log_probs

    def mean(self):
        return self.means


class DummyGradScalar(object):
    def __init__(self, *args, **kwargs):
        pass

    def scale(self, input):
        return input

    def update(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, x):
        pass

    def step(self, opt):
        opt.step()

    def unscale_(self, x):
        return x


def get_opt(params, cfgopt, use_ema, other_cfg=None):
    if cfgopt.type == 'adam':
        optimizer = optim.Adam(params,
                               lr=float(cfgopt.lr),
                               betas=(cfgopt.beta1, cfgopt.beta2),
                               weight_decay=cfgopt.weight_decay)
    elif cfgopt.type == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=float(cfgopt.lr),
                                    momentum=cfgopt.momentum)
    elif cfgopt.type == 'adamax':
        from utils.adamax import Adamax
        logger.info('[Optimizer] Adamax, lr={}, weight_decay={}, eps={}',
                    cfgopt.lr, cfgopt.weight_decay, 1e-4)
        optimizer = Adamax(params, float(cfgopt.lr),
                           weight_decay=args.weight_decay, eps=1e-4)

    else:
        assert 0, "Optimizer type should be either 'adam' or 'sgd'"
    if use_ema:
        logger.info('use_ema')
        ema_decay = 0.9999
        from .ema import EMA
        optimizer = EMA(optimizer, ema_decay=ema_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: 1.0)  # constant lr
    scheduler_type = getattr(cfgopt, "scheduler", None)
    if scheduler_type is not None and len(scheduler_type) > 0:
        logger.info('get scheduler_type: {}', scheduler_type)
        if scheduler_type == 'exponential':
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        elif scheduler_type == 'step':
            step_size = int(getattr(cfgopt, "step_epoch", 500))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=step_size,
                                                  gamma=decay)
        elif scheduler_type == 'linear':  # use default setting from shapeLatent
            start_epoch = int(getattr(cfgopt, 'sched_start_epoch', 200*1e3))
            end_epoch = int(getattr(cfgopt, 'sched_end_epoch', 400*1e3))
            end_lr = float(getattr(cfgopt, 'end_lr', 1e-4))
            start_lr = cfgopt.lr

            def lambda_rule(epoch):
                if epoch <= start_epoch:
                    return 1.0
                elif epoch <= end_epoch:
                    total = end_epoch - start_epoch
                    delta = epoch - start_epoch
                    frac = delta / total
                    return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
                else:
                    return end_lr / start_lr
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda_rule)

        elif scheduler_type == 'lambda':  # linear':
            step_size = int(getattr(cfgopt, "step_epoch", 2000))
            final_ratio = float(getattr(cfgopt, "final_ratio", 0.01))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.5))
            duration_ratio = float(getattr(cfgopt, "duration_ratio", 0.45))

            def lambda_rule(ep):
                lr_l = 1.0 - min(
                    1,
                    max(0, ep - start_ratio * step_size) /
                    float(duration_ratio * step_size)) * (1 - final_ratio)
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda_rule)

        elif scheduler_type == 'cosine_anneal_nocycle':
            ## logger.info('scheduler_type: {}', scheduler_type)
            assert(other_cfg is not None)
            final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
            eta_min = float(cfgopt.lr) * final_lr_ratio
            eta_max = float(cfgopt.lr)

            total_epoch = int(other_cfg.trainer.epochs)
            ##getattr(cfgopt, "step_epoch", 2000)
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.6))
            T_max = total_epoch * (1 - start_ratio)

            def lambda_rule(ep):
                curr_ep = max(0., ep - start_ratio * total_epoch)
                lr = eta_min + 0.5 * (eta_max - eta_min) * (
                    1 + np.cos(np.pi * curr_ep / T_max))
                lr_l = lr / eta_max
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda_rule)

        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step'"
    return optimizer, scheduler


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAvgrageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


class DummyDDP(nn.Module):
    def __init__(self, model):
        super(DummyDDP, self).__init__()
        self.module = model

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


# def create_exp_dir(path, scripts_to_save=None):
#    if not os.path.exists(path):
#        os.makedirs(path, exist_ok=True)
#    print('Experiment dir : {}'.format(path))
#
#    if scripts_to_save is not None:
#        if not os.path.exists(os.path.join(path, 'scripts')):
#            os.mkdir(os.path.join(path, 'scripts'))
#        for script in scripts_to_save:
#            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
#            shutil.copyfile(script, dst_file)
#

# class Logger(object):
#    def __init__(self, rank, save):
#        # other libraries may set logging before arriving at this line.
#        # by reloading logging, we can get rid of previous configs set by other libraries.
#        from importlib import reload
#        reload(logging)
#        self.rank = rank
#        if self.rank == 0:
#            log_format = '%(asctime)s %(message)s'
#            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
#            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
#            fh.setFormatter(logging.Formatter(log_format))
#            logging.getLogger().addHandler(fh)
#            self.start_time = time.time()
#
#    def info(self, string, *args):
#        if self.rank == 0:
#            elapsed_time = time.time() - self.start_time
#            elapsed_time = time.strftime(
#                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
#            if isinstance(string, str):
#                string = elapsed_time + string
#            else:
#                logging.info(elapsed_time)
#            logging.info(string, *args)

def flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()} \
        if isinstance(dd, dict) else {prefix: dd}


class Writer(object):
    def __init__(self, rank=0, save=None, exp=None, wandb=False):
        self.rank = rank
        self.exp = None
        self.wandb = False
        self.meter_dict = {}
        if self.rank == 0:
            self.exp = exp
            if USE_TFB and save is not None:
                logger.info('init TFB: {}', save)
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=save, flush_secs=20)
            else:
                logger.info('Not init TFB')
                self.writer = None
            if self.exp is not None and save is not None:
                with open(os.path.join(save, 'url.txt'), 'a') as f:
                    f.write(self.exp.url)
                    f.write('\n')
            self.wandb = wandb
        else:
            logger.info('rank={}, init writer as a blackhole', rank)

    def set_model_graph(self, *args, **kwargs):
        if self.rank == 0 and self.exp is not None:
            self.exp.set_model_graph(*args, **kwargs)

    @property
    def url(self):
        if self.exp is not None:
            return self.exp.url
        else:
            return 'none'

    def add_hparams(self, cfg, args):  # **kwargs):
        if self.exp is not None:
            self.exp.log_parameters(flatten_dict(cfg))
            self.exp.log_parameters(flatten_dict(args))
        if self.wandb:
            WB.config.update(flatten_dict(cfg))
            WB.config.update(flatten_dict(args))

    def avg_meter(self, name, value, step=None, epoch=None):
        if self.rank == 0:
            if name not in self.meter_dict:
                self.meter_dict[name] = AvgrageMeter()
            self.meter_dict[name].update(value)

    def upload_meter(self, step=None, epoch=None):
        for name, value in self.meter_dict.items():
            self.add_scalar(name, value.avg, step=step, epoch=epoch)
        self.meter_dict = {}

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            if 'step' in kwargs:
                self.writer.add_scalar(*args,
                                       global_step=kwargs['step'])
            else:
                self.writer.add_scalar(*args, **kwargs)

        if self.exp is not None:
            self.exp.log_metric(*args, **kwargs)
        if self.wandb:
            name = args[0]
            v = args[1]
            WB.log({name: v})

    def log_model(self, name, path):
        pass

    def log_other(self, name, value):
        if self.rank == 0 and self.exp is not None:
            self.exp.log_other(name, value)
        # if self.rank == 0 and self.exp is not None:
        #    self.exp.log_model(name, path)

    def watch(self, model):
        if self.wandb:
            WB.watch(model)

    def log_points_3d(self, scene_name, points, step=0):  # *args, **kwargs):
        if self.rank == 0 and self.exp is not None:
            self.exp.log_points_3d(*args, **kwargs)
        if self.wandb:
            WB.log({"point_cloud": WB.Object3D(points)})

    def add_figure(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_image(*args, **kwargs)
            self.writer.flush()
        if self.exp is not None:
            name, img, i = args
            if isinstance(img, Image.Image):
                # logger.debug('log PIL Imgae: {}, {}', name, i)
                self.exp.log_image(img, name, step=i)
            elif type(img) is str:
                # logger.debug('log str image: {}, {}: {}', name, i, img)
                self.exp.log_image(img, name, step=i)
            elif torch.is_tensor(img):
                if img.shape[0] in [3, 4] and len(img.shape) == 3:  # 3,H,W
                    img = img.permute(1, 2, 0).contiguous()  # 3,H,W -> H,W,3
                if img.max() < 100:  # [0-1]
                    ndarr = img.mul(255).add_(0.5).clamp_(
                        0, 255).to('cpu')  # .squeeze()
                    ndarr = ndarr.numpy().astype(np.uint8)
                    # .reshape(-1, ndarr.shape[-1]))
                    im = Image.fromarray(ndarr)
                    self.exp.log_image(im, name, step=i)
                else:
                    im = img.to('cpu').numpy()
                    self.exp.log_image(im, name, step=i)

            elif isinstance(img, (np.ndarray, np.generic)):
                if img.shape[0] == 3 and len(img.shape) == 3:  # 3,H,W
                    img = img.transpose(1, 2, 0)
                self.exp.log_image(img, name, step=i)
        if self.wandb and torch.is_tensor(img) and self.rank == 0:
            ## print(img.shape, img.max(), img.type())
            WB.log({name: WB.Image(img.numpy())})

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_histogram(*args, **kwargs)
        if self.exp is not None:
            name, value, step = args
            self.exp.log_histogram_3d(value, name, step)
            # *args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:  # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.close()

    def log_asset(self, *args, **kwargs):
        if self.exp is not None:
            self.exp.log_asset(*args, **kwargs)


def common_init(rank, seed, save_dir, comet_key=''):
    # we use different seeds per gpu. But we sync the weights after model initialization.
    logger.info('[common-init] at rank={}, seed={}', rank, seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True

    # prepare logging and tensorboard summary
    #logging = Logger(rank, save_dir)
    logging = None
    if rank == 0:
        if os.path.exists('.comet_api'):
            comet_args = json.load(open('.comet_api', 'r'))
            exp = Experiment(display_summary_level=0,
                             disabled=USE_COMET == 0,
                             **comet_args)
            exp.set_name(save_dir.split('exp/')[-1])
            exp.set_cmd_args()
            exp.log_code(folder='./models/')
            exp.log_code(folder='./trainers/')
            exp.log_code(folder='./utils/')
            exp.log_code(folder='./datasets/')
        else:
            exp = None

        if os.path.exists('.wandb_api'):
            wb_args = json.load(open('.wandb_api', 'r'))
            wb_dir = '../exp/wandb/' if not os.path.exists(
                '/workspace/result') else '/workspace/result/wandb/'
            if not os.path.exists(wb_dir):
                os.makedirs(wb_dir)
            WB.init(
                project=wb_args['project'],
                entity=wb_args['entity'],
                name=save_dir.split('exp/')[-1],
                dir=wb_dir
            )
            wandb = True
        else:
            wandb = False
    else:
        exp = None
        wandb = False
    writer = Writer(rank, save_dir, exp, wandb)
    logger.info('[common-init] DONE')

    return logging, writer


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)

    return stride


def get_cout(cin, stride):
    if stride == 1:
        cout = cin
    elif stride == -1:
        cout = cin // 2
    elif stride == 2:
        cout = 2 * cin

    return cout


def kl_balancer_coeff(num_scales, groups_per_scale, fun='square'):
    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_per_scale[num_scales - i - 1])
                          for i in range(num_scales)], dim=0).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)],
                          dim=0).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat(
            [np.sqrt(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1])
             for i in range(num_scales)],
            dim=0).cuda()
    elif fun == 'square':
        coeff = torch.cat(
            [np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * torch.ones(groups_per_scale[num_scales - i - 1])
             for i in range(num_scales)], dim=0).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


def rec_balancer(rec_all, rec_coeff=1.0, npoints=None):
    # layer depth increase, alpha_i increase, 1/alpha_i decrease; kl_coeff decrease
    # the rec with more points should have higher loss
    min_points = min(npoints)
    coeff = []
    rec_loss = 0
    assert(len(rec_all) == len(npoints))
    for ni, n in enumerate(npoints):
        c = rec_coeff*np.sqrt(n/min_points)
        rec_loss += rec_all[ni] * c
        coeff.append(c)  # the smallest points' loss weight is 1

    return rec_loss, coeff, rec_all


def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    # layer depth increase, alpha_i increase, 1/alpha_i decrease; kl_coeff decrease
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)
        # kl = ( sum * kl / alpha )
        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_per_group_vada(all_log_q, all_neg_log_p):
    assert(len(all_log_q) == len(all_neg_log_p)
           ), f'get len={len(all_log_q)} and {len(all_neg_log_p)}'

    kl_all_list = []
    kl_diag = []
    for log_q, neg_log_p in zip(all_log_q, all_neg_log_p):
        kl_diag.append(torch.mean(
            torch.sum(neg_log_p + log_q, dim=[2, 3]), dim=0))
        kl_all_list.append(torch.sum(neg_log_p + log_q,
                           dim=[1, 2, 3]))  # sum over D,H,W

    # kl_all = torch.stack(kl_all, dim=1)   # batch x num_total_groups
    kl_vals = torch.mean(torch.stack(kl_all_list, dim=1),
                         dim=0)   # mean per group

    return kl_all_list, kl_vals, kl_diag


def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
    # return max(min(max_kl_coeff * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)


def log_iw(decoder, x, log_q, log_p, crop=False):
    recon = reconstruction_loss(decoder, x, crop)
    return - recon - log_q + log_p


def reconstruction_loss(decoder, x, crop=False):

    recon = decoder.log_p(x)
    if crop:
        recon = recon[:, :, 2:30, 2:30]

    if isinstance(decoder, DiscMixLogistic):
        return - torch.sum(recon, dim=[1, 2])  # summation over RGB is done.
    else:
        return - torch.sum(recon, dim=[1, 2, 3])


def vae_terms(all_log_q, all_eps):

    # compute kl
    kl_all = []
    kl_diag = []
    log_p, log_q = 0., 0.
    for log_q_conv, eps in zip(all_log_q, all_eps):
        log_p_conv = log_p_standard_normal(eps)
        kl_per_var = log_q_conv - log_p_conv
        kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
        kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
        log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
        log_p += torch.sum(log_p_conv, dim=[1, 2, 3])
    return log_q, log_p, kl_all, kl_diag


def sum_log_q(all_log_q):
    log_q = 0.
    for log_q_conv in all_log_q:
        log_q += torch.sum(log_q_conv, dim=[1, 2, 3])

    return log_q


def cross_entropy_normal(all_eps):

    cross_entropy = 0.
    neg_log_p_per_group = []
    for eps in all_eps:
        neg_log_p_conv = - log_p_standard_normal(eps)
        neg_log_p = torch.sum(neg_log_p_conv, dim=[1, 2, 3])
        cross_entropy += neg_log_p
        neg_log_p_per_group.append(neg_log_p_conv)

    return cross_entropy, neg_log_p_per_group


def tile_image(batch_image, n, m=None):
    if m is None:
        m = n
    assert n * m == batch_image.size(0)
    channels, height, width = batch_image.size(
        1), batch_image.size(2), batch_image.size(3)
    batch_image = batch_image.view(n, m, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)  # n, height, n, width, c
    batch_image = batch_image.contiguous().view(channels, n * height, m * width)
    return batch_image


def average_gradients_naive(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            if param.requires_grad:
                param.grad.data /= size
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        if isinstance(params, types.GeneratorType):
            params = [p for p in params]

        size = float(dist.get_world_size())
        grad_data = []
        grad_size = []
        grad_shapes = []
        # Gather all grad values
        for param in params:
            if param.requires_grad:
                if param.grad is not None:
                    grad_size.append(param.grad.data.numel())
                    grad_shapes.append(list(param.grad.data.shape))
                    grad_data.append(param.grad.data.flatten())
        grad_data = torch.cat(grad_data).contiguous()

        # All-reduce grad values
        grad_data /= size
        dist.all_reduce(grad_data, op=dist.ReduceOp.SUM)

        # Put back the reduce grad values to parameters
        base = 0
        i = 0
        for param in params:
            if param.requires_grad and param.grad is not None:
                param.grad.data = grad_data[base:base +
                                            grad_size[i]].view(grad_shapes[i])
                base += grad_size[i]
                i += 1


def average_params(params, is_distributed):
    """ parameter averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            param.data /= size
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def broadcast_params(params, is_distributed):
    if is_distributed:
        for param in params:
            dist.broadcast(param.data, src=0)


def num_output(dataset):
    if dataset in {'mnist',  'omniglot'}:
        return 28 * 28
    elif dataset == 'cifar10':
        return 3 * 32 * 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return 3 * size * size
    elif dataset == 'ffhq':
        return 3 * 256 * 256
    else:
        raise NotImplementedError


def get_input_size(dataset):
    if dataset in {'mnist', 'omniglot'}:
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    elif dataset.startswith('shape'):
        return 1  # 2048
    else:
        raise NotImplementedError


def get_bpd_coeff(dataset):
    n = num_output(dataset)
    return 1. / np.log(2.) / n


def get_channel_multiplier(dataset, num_scales):
    if dataset in {'cifar10', 'omniglot'}:
        mult = (1, 1, 1)
    elif dataset in {'celeba_256', 'ffhq', 'lsun_church_256'}:
        if num_scales == 3:
            mult = (1, 1, 1)        # used for prior at 16
        elif num_scales == 4:
            mult = (1, 2, 2, 2)     # used for prior at 32
        elif num_scales == 5:
            mult = (1, 1, 2, 2, 2)  # used for prior at 64
    elif dataset == 'mnist':
        mult = (1, 1)
    else:
        mult = (1, 1)
        # raise NotImplementedError

    return mult


def get_attention_scales(dataset):
    if dataset in {'cifar10', 'omniglot'}:
        attn = (True, False, False)
    elif dataset in {'celeba_256', 'ffhq', 'lsun_church_256'}:
        # attn = (False, True, False, False) # used for 32
        attn = (False, False, True, False, False)  # used for 64
    elif dataset == 'mnist':
        attn = (True, False)
    else:
        raise NotImplementedError

    return attn


def change_bit_length(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x


def view4D(t, size, inplace=True):
    """
     Equal to view(-1, 1, 1, 1).expand(size)
     Designed because of this bug:
     https://github.com/pytorch/pytorch/pull/48696
    """
    if inplace:
        return t.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1).expand(size)
    else:
        return t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(size)


def get_arch_cells(arch_type, use_se):
    if arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_dec'] = {
            'conv_branch': ['mconv_e6k5g0'], 'se': use_se}
        arch_cells['up_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se}
        arch_cells['normal_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {
            'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_dec'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['up_dec'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish2':
        arch_cells = dict()
        arch_cells['normal_enc'] = {
            'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['down_enc'] = {
            'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['normal_dec'] = {
            'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['up_dec'] = {'conv_branch': [
            'res_bnswish_x2'], 'se': use_se}
        arch_cells['normal_pre'] = {
            'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['down_pre'] = {
            'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['normal_post'] = {
            'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': [
            'res_bnswish_x2'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv_attn':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish', ], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['down_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['normal_dec'] = {'conv_branch': [
            'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['up_dec'] = {'conv_branch': [
            'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['normal_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {
            'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv_attn_half':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_enc'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_dec'] = {'conv_branch': [
            'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['up_dec'] = {'conv_branch': [
            'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['normal_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': [
            'res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {
            'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    else:
        raise NotImplementedError

    return arch_cells


def get_arch_cells_denoising(arch_type, use_se, apply_sqrt2):
    if arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc_diff'] = {
            'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
        arch_cells['down_enc_diff'] = {
            'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
        arch_cells['normal_dec_diff'] = {
            'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
        arch_cells['up_dec_diff'] = {
            'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
    elif arch_type == 'res_ho':
        arch_cells = dict()
        arch_cells['normal_enc_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
        arch_cells['down_enc_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
        arch_cells['normal_dec_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
        arch_cells['up_dec_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
    elif arch_type == 'res_ho_p1':
        arch_cells = dict()
        arch_cells['normal_enc_diff'] = {
            'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
        arch_cells['down_enc_diff'] = {
            'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
        arch_cells['normal_dec_diff'] = {
            'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
        arch_cells['up_dec_diff'] = {
            'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
    elif arch_type == 'res_ho_attn':
        arch_cells = dict()
        arch_cells['normal_enc_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
        arch_cells['down_enc_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
        arch_cells['normal_dec_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
        arch_cells['up_dec_diff'] = {
            'conv_branch': ['res_gnswish_x2'], 'se': use_se}
    else:
        raise NotImplementedError

    for k in arch_cells:
        arch_cells[k]['apply_sqrt2'] = apply_sqrt2

    return arch_cells


def groups_per_scale(num_scales, num_groups_per_scale):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
    return g


#class PositionalEmbedding(nn.Module):
#    def __init__(self, embedding_dim, scale):
#        super(PositionalEmbedding, self).__init__()
#        self.embedding_dim = embedding_dim
#        self.scale = scale
#
#    def forward(self, timesteps):
#        assert len(timesteps.shape) == 1
#        timesteps = timesteps * self.scale
#        half_dim = self.embedding_dim // 2
#        emb = math.log(10000) / (half_dim - 1)
#        emb = torch.exp(torch.arange(half_dim) * -emb)
#        emb = emb.to(device=timesteps.device)
#        emb = timesteps[:, None] * emb[None, :]
#        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#        return emb
#
#
#class RandomFourierEmbedding(nn.Module):
#    def __init__(self, embedding_dim, scale):
#        super(RandomFourierEmbedding, self).__init__()
#        self.w = nn.Parameter(torch.randn(
#            size=(1, embedding_dim // 2)) * scale, requires_grad=False)
#
#    def forward(self, timesteps):
#        emb = torch.mm(timesteps[:, None], self.w * 2 * 3.14159265359)
#        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#
#
#def init_temb_fun(embedding_type, embedding_scale, embedding_dim):
#    if embedding_type == 'positional':
#        temb_fun = PositionalEmbedding(embedding_dim, embedding_scale)
#    elif embedding_type == 'fourier':
#        temb_fun = RandomFourierEmbedding(embedding_dim, embedding_scale)
#    else:
#        raise NotImplementedError
#
#    return temb_fun


def symmetrize_image_data(images):
    return 2.0 * images - 1.0


def unsymmetrize_image_data(images):
    return (images + 1.) / 2.


def normalize_symmetric(images):
    """
    Normalize images by dividing the largest intensity. Used for visualizing the intermediate steps.
    """
    b = images.shape[0]
    m, _ = torch.max(torch.abs(images).view(b, -1), dim=1)
    images /= (m.view(b, 1, 1, 1) + 1e-3)

    return images


@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    # 5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]
    return x.div(5.).tanh_().mul(5.)


@torch.jit.script
def soft_clamp(x: torch.Tensor, a: torch.Tensor):
    return x.div(a).tanh_().mul(a)


class SoftClamp5(nn.Module):
    def __init__(self):
        super(SoftClamp5, self).__init__()

    def forward(self, x):
        return soft_clamp5(x)


def override_architecture_fields(args, stored_args, logging):
    # list of architecture parameters used in NVAE:
    architecture_fields = ['arch_instance', 'num_nf', 'num_latent_scales', 'num_groups_per_scale',
                           'num_latent_per_group', 'num_channels_enc', 'num_preprocess_blocks',
                           'num_preprocess_cells', 'num_cell_per_cond_enc', 'num_channels_dec',
                           'num_postprocess_blocks', 'num_postprocess_cells', 'num_cell_per_cond_dec',
                           'decoder_dist', 'num_x_bits', 'log_sig_q_scale', 'latent_grad_cutoff',
                           'progressive_output_vae', 'progressive_input_vae', 'channel_mult']

    # backward compatibility
    """ We have broken backward compatibility. No need to se these manually
    if not hasattr(stored_args, 'log_sig_q_scale'):
        logging.info('*** Setting %s manually ****', 'log_sig_q_scale')
        setattr(stored_args, 'log_sig_q_scale', 5.)

    if not hasattr(stored_args, 'latent_grad_cutoff'):
        logging.info('*** Setting %s manually ****', 'latent_grad_cutoff')
        setattr(stored_args, 'latent_grad_cutoff', 0.)

    if not hasattr(stored_args, 'progressive_input_vae'):
        logging.info('*** Setting %s manually ****', 'progressive_input_vae')
        setattr(stored_args, 'progressive_input_vae', 'none')

    if not hasattr(stored_args, 'progressive_output_vae'):
        logging.info('*** Setting %s manually ****', 'progressive_output_vae')
        setattr(stored_args, 'progressive_output_vae', 'none')
    """

    for f in architecture_fields:
        if not hasattr(args, f) or getattr(args, f) != getattr(stored_args, f):
            logging.info('Setting %s from loaded checkpoint', f)
            setattr(args, f, getattr(stored_args, f))


def init_processes(rank, size, fn, args, config):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    logger.info('set MASTER_PORT: {}, MASTER_PORT: {}', os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    
    # if args.num_proc_node == 1:  # try to solve the port occupied issue
    #     import socket
    #     import errno
    #     a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     for p in range(6010, 6030):
    #         location = (args.master_address, p)  # "127.0.0.1", p)
    #         try:
    #             a_socket.bind((args.master_address, p))
    #             logger.debug('set port as {}', p)
    #             os.environ['MASTER_PORT'] = '%d' % p
    #             a_socket.close()
    #             break
    #         except socket.error as e:
    #             a = 0
    #             # if e.errno == errno.EADDRINUSE:
    #             #    # logger.debug("Port {} is already in use", p)
    #             # else:
    #             #    logger.debug(e)

    logger.info('init_process: rank={}, world_size={}', rank, size)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args, config)
    logger.info('barrier: rank={}, world_size={}', rank, size)
    dist.barrier()
    logger.info('skip destroy_process_group: rank={}, world_size={}', rank, size)
    # dist.destroy_process_group()
    logger.info('skip destroy fini')


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape, device='cuda') * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y, device='cuda')


def trace_df_dx_hutchinson(f, x, noise, no_autograd):
    """
    Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
    """
    if no_autograd:
        # the following is compatible with checkpointing
        torch.sum(f * noise).backward()
        # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
        jvp = x.grad
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        x.grad = None
    else:
        jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

    return trJ


def calc_jacobian_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, g2_t, var_N_t, args):
    """
    Calculates Jabobian regularization loss. For reference implementations, see
    https://github.com/facebookresearch/jacobian_regularizer/blob/master/jacobian/jacobian.py or
    https://github.com/cfinlay/ffjord-rnode/blob/master/lib/layers/odefunc.py.
    """
    # eps_t_jvp = eps_t.detach()
    # eps_t_jvp = eps_t.detach().requires_grad_()
    if args.no_autograd_jvp:
        raise NotImplementedError(
            "We have not implemented no_autograd_jvp for jacobian reg.")

    jvp_ode_func_norms = []
    alpha = torch.sigmoid(dae.mixing_logit.detach())
    for _ in range(args.jac_reg_samples):
        noise = sample_gaussian_like(eps_t)
        jvp = torch.autograd.grad(
            pred_params, eps_t, noise, create_graph=True)[0]

        if args.sde_type in ['geometric_sde', 'vpsde', 'power_vpsde']:
            jvp_ode_func = alpha * (noise * torch.sqrt(var_t) - jvp)
            if not args.jac_kin_reg_drop_weights:
                jvp_ode_func = f_t / torch.sqrt(var_t) * jvp_ode_func
        elif args.sde_type in ['sub_vpsde', 'sub_power_vpsde']:
            sigma2_N_t = (1.0 - m_t ** 2) ** 2 + m_t ** 2
            jvp_ode_func = noise * torch.sqrt(var_t) / (1.0 - m_t ** 4) - (
                (1.0 - alpha) * noise * torch.sqrt(var_t) / sigma2_N_t + alpha * jvp)
            if not args.jac_kin_reg_drop_weights:
                jvp_ode_func = f_t * (1.0 - m_t ** 4) / \
                    torch.sqrt(var_t) * jvp_ode_func
        elif args.sde_type in ['vesde']:
            jvp_ode_func = (1.0 - alpha) * noise * \
                torch.sqrt(var_t) / var_N_t + alpha * jvp
            if not args.jac_kin_reg_drop_weights:
                jvp_ode_func = 0.5 * g2_t / torch.sqrt(var_t) * jvp_ode_func
        else:
            raise ValueError("Unrecognized SDE type: {}".format(args.sde_type))

        jvp_ode_func_norms.append(jvp_ode_func.view(
            eps_t.size(0), -1).pow(2).sum(dim=1, keepdim=True))

    jac_reg_loss = torch.cat(jvp_ode_func_norms, dim=1).mean()
    # jac_reg_loss = torch.mean(jvp_ode_func.view(eps_t.size(0), -1).pow(2).sum(dim=1))
    return jac_reg_loss


def calc_kinetic_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, g2_t, var_N_t, args):
    """
    Calculates kinetic regularization loss. For a reference implementation, see
    https://github.com/cfinlay/ffjord-rnode/blob/master/lib/layers/wrappers/cnf_regularization.py
    """
    # eps_t_kin = eps_t.detach()

    alpha = torch.sigmoid(dae.mixing_logit.detach())
    if args.sde_type in ['geometric_sde', 'vpsde', 'power_vpsde']:
        ode_func = alpha * (eps_t * torch.sqrt(var_t) - pred_params)
        if not args.jac_kin_reg_drop_weights:
            ode_func = f_t / torch.sqrt(var_t) * ode_func
    elif args.sde_type in ['sub_vpsde', 'sub_power_vpsde']:
        sigma2_N_t = (1.0 - m_t ** 2) ** 2 + m_t ** 2
        ode_func = eps_t * torch.sqrt(var_t) / (1.0 - m_t ** 4) - (
            (1.0 - alpha) * eps_t * torch.sqrt(var_t) / sigma2_N_t + alpha * pred_params)
        if not args.jac_kin_reg_drop_weights:
            ode_func = f_t * (1.0 - m_t ** 4) / torch.sqrt(var_t) * ode_func
    elif args.sde_type in ['vesde']:
        ode_func = (1.0 - alpha) * eps_t * torch.sqrt(var_t) / \
            var_N_t + alpha * pred_params
        if not args.jac_kin_reg_drop_weights:
            ode_func = 0.5 * g2_t / torch.sqrt(var_t) * ode_func
    else:
        raise ValueError("Unrecognized SDE type: {}".format(args.sde_type))

    kin_reg_loss = torch.mean(ode_func.view(
        eps_t.size(0), -1).pow(2).sum(dim=1))
    return kin_reg_loss


def different_p_q_objectives(iw_sample_p, iw_sample_q):
    assert iw_sample_p in ['ll_uniform', 'drop_all_uniform', 'll_iw', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw',
                           'drop_sigma2t_uniform']
    assert iw_sample_q in ['reweight_p_samples', 'll_uniform', 'll_iw']
    # Removed assert below. It may be stupid, but user can still do it. It may make sense for debugging purposes.
    # assert iw_sample_p != iw_sample_q, 'It does not make sense to use the same objectives for p and q, but train ' \
    #                                    'with separated q and p updates. To reuse the p objective for q, specify ' \
    #                                    '"reweight_p_samples" instead (for the ll-based objectives, the ' \
    #                                    'reweighting factor will simply be 1.0 then)!'
    # In these cases, we reuse the likelihood-based p-objective (either the uniform sampling version or the importance
    # sampling version) also for q.
    if iw_sample_p in ['ll_uniform', 'll_iw'] and iw_sample_q == 'reweight_p_samples':
        return False
    # In these cases, we are using a non-likelihood-based objective for p, and hence definitly need to use another q
    # objective.
    else:
        return True


def decoder_output(dataset, logits, fixed_log_scales=None):
    if dataset in {'cifar10', 'celeba_64', 'celeba_256', 'imagenet_32', 'imagenet_64', 'ffhq',
                   'lsun_bedroom_128', 'lsun_bedroom_256', 'mnist', 'omniglot',
                   'lsun_church_256'}:
        return PixelNormal(logits, fixed_log_scales)
    else:
        return PixelNormal(logits, fixed_log_scales)
        # raise NotImplementedError


def get_mixed_prediction(mixed_prediction, param, mixing_logit, mixing_component=None):
    if mixed_prediction:
        assert mixing_component is not None, 'Provide mixing component when mixed_prediction is enabled.'
        coeff = torch.sigmoid(mixing_logit)
        param = (1 - coeff) * mixing_component + coeff * param

    return param


def set_vesde_sigma_max(args, vae, train_queue, logging, is_distributed):
    logging.info('')
    logging.info(
        'Calculating max. pairwise distance in latent space to set sigma2_max for VESDE...')

    eps_list = []
    vae.eval()
    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()
        x = symmetrize_image_data(x)

        # run vae
        with autocast(enabled=args.autocast_train):
            with torch.set_grad_enabled(False):
                logits, all_log_q, all_eps = vae(x)
                eps = torch.cat(all_eps, dim=1)

        eps_list.append(eps.detach())

        # if step > 5: ### DEBUG
        #     break ### DEBUG

    # concat eps tensor on each GPU and then gather all on all GPUs
    eps_this_rank = torch.cat(eps_list, dim=0)
    if is_distributed:
        eps_all_gathered = [torch.zeros_like(
            eps_this_rank)] * dist.get_world_size()
        dist.all_gather(eps_all_gathered, eps_this_rank)
        eps_full = torch.cat(eps_all_gathered, dim=0)
    else:
        eps_full = eps_this_rank

    # max pairwise distance squared between all latent encodings, is computed on CPU
    eps_full = eps_full.cpu().float()
    eps_full = eps_full.flatten(start_dim=1).unsqueeze(0)
    max_pairwise_dist_sqr = torch.cdist(eps_full, eps_full).square().max()
    max_pairwise_dist_sqr = max_pairwise_dist_sqr.cuda()

    # to be safe, we broadcast to all GPUs if we are in distributed environment. Shouldn't be necessary in principle.
    if is_distributed:
        dist.broadcast(max_pairwise_dist_sqr, src=0)

    args.sigma2_max = max_pairwise_dist_sqr.item()

    logging.info('Done! Set args.sigma2_max set to {}'.format(args.sigma2_max))
    logging.info('')
    return args


def mask_inactive_variables(x, is_active):
    x = x * is_active
    return x


def common_x_operations(x, num_x_bits):
    x = x[0] if len(x) > 1 else x
    x = x.cuda()

    # change bit length
    x = change_bit_length(x, num_x_bits)
    x = symmetrize_image_data(x)

    return x


def vae_regularization(args, vae_sn_calculator, loss_weight=None):
    """
        when using hvae_trainer, we pass args=None, and loss_weight value 
    """
    regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = 0., 0., 0., args.weight_decay_norm_vae if loss_weight is None else loss_weight
    if loss_weight is not None or args.train_vae:
        vae_norm_loss = vae_sn_calculator.spectral_norm_parallel()
        vae_bn_loss = vae_sn_calculator.batchnorm_loss()
        regularization_q = (vae_norm_loss + vae_bn_loss) * vae_wdn_coeff

    return regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff


def dae_regularization(args, dae_sn_calculator, diffusion, dae, step, t, pred_params_p, eps_t_p, var_t_p, m_t_p, g2_t_p):
    dae_wdn_coeff = args.weight_decay_norm_dae
    dae_norm_loss = dae_sn_calculator.spectral_norm_parallel()
    dae_bn_loss = dae_sn_calculator.batchnorm_loss()
    regularization_p = (dae_norm_loss + dae_bn_loss) * dae_wdn_coeff

    # Jacobian regularization
    jac_reg_loss = 0.
    if args.jac_reg_coeff > 0.0 and step % args.jac_reg_freq == 0:
        f_t = diffusion.f(t).view(-1, 1, 1, 1)
        var_N_t = diffusion.var_N(
            t).view(-1, 1, 1, 1) if args.sde_type == 'vesde' else None
        """
        # Arash: Please remove the following if it looks correct to you, Karsten.
        # jac_reg_loss = utils.calc_jacobian_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, args)
        if args.iw_sample_q in ['ll_uniform', 'll_iw']:
            pred_params_jac_reg = torch.chunk(pred_params, chunks=2, dim=0)[0]
            var_t_jac_reg, m_t_jac_reg, f_t_jac_reg = torch.chunk(var_t, chunks=2, dim=0)[0], \
                                                      torch.chunk(m_t, chunks=2, dim=0)[0], \
                                                      torch.chunk(f_t, chunks=2, dim=0)[0]
            g2_t_jac_reg = torch.chunk(g2_t, chunks=2, dim=0)[0]
            var_N_t_jac_reg = torch.chunk(var_N_t, chunks=2, dim=0)[0] if args.sde_type == 'vesde' else None
        else:
            pred_params_jac_reg = pred_params
            var_t_jac_reg, m_t_jac_reg, f_t_jac_reg, g2_t_jac_reg, var_N_t_jac_reg = var_t, m_t, f_t, g2_t, var_N_t
        jac_reg_loss = utils.calc_jacobian_regularization(pred_params_jac_reg, eps_t_p, dae, var_t_jac_reg, m_t_jac_reg,
                                                          f_t_jac_reg, g2_t_jac_reg, var_N_t_jac_reg, args)
        """
        jac_reg_loss = calc_jacobian_regularization(pred_params_p, eps_t_p, dae, var_t_p, m_t_p,
                                                    f_t, g2_t_p, var_N_t, args)
        regularization_p += args.jac_reg_coeff * jac_reg_loss

    # Kinetic regularization
    kin_reg_loss = 0.
    if args.kin_reg_coeff > 0.0:
        f_t = diffusion.f(t).view(-1, 1, 1, 1)
        var_N_t = diffusion.var_N(
            t).view(-1, 1, 1, 1) if args.sde_type == 'vesde' else None
        """
        # Arash: Please remove the following if it looks correct to you, Karsten.
        # kin_reg_loss = utils.calc_kinetic_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, args)
        if args.iw_sample_q in ['ll_uniform', 'll_iw']:
            pred_params_kin_reg = torch.chunk(pred_params, chunks=2, dim=0)[0]
            var_t_kin_reg, m_t_kin_reg, f_t_kin_reg = torch.chunk(var_t, chunks=2, dim=0)[0], \
                                                      torch.chunk(m_t, chunks=2, dim=0)[0], \
                                                      torch.chunk(f_t, chunks=2, dim=0)[0]
            g2_t_kin_reg = torch.chunk(g2_t, chunks=2, dim=0)[0]
            var_N_t_kin_reg = torch.chunk(var_N_t, chunks=2, dim=0)[0] if args.sde_type == 'vesde' else None
        else:
            pred_params_kin_reg = pred_params
            var_t_kin_reg, m_t_kin_reg, f_t_kin_reg, g2_t_kin_reg, var_N_t_kin_reg = var_t, m_t, f_t, g2_t, var_N_t
        kin_reg_loss = utils.calc_kinetic_regularization(pred_params_kin_reg, eps_t_p, dae, var_t_kin_reg, m_t_kin_reg,
                                                         f_t_kin_reg, g2_t_kin_reg, var_N_t_kin_reg, args)
        """
        kin_reg_loss = calc_kinetic_regularization(pred_params_p, eps_t_p, dae, var_t_p, m_t_p,
                                                   f_t, g2_t_p, var_N_t, args)
        regularization_p += args.kin_reg_coeff * kin_reg_loss

    return regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff, jac_reg_loss, kin_reg_loss


def update_vae_lr(args, global_step, warmup_iters, vae_optimizer):
    if global_step < warmup_iters:
        lr = args.trainer.opt.lr * float(global_step) / warmup_iters
        for param_group in vae_optimizer.param_groups:
            param_group['lr'] = lr
        # use same lr if lr for local-dae is not specified


def update_lr(args, global_step, warmup_iters, dae_optimizer, vae_optimizer, dae_local_optimizer=None):
    if global_step < warmup_iters:
        lr = args.learning_rate_dae * float(global_step) / warmup_iters
        if args.learning_rate_mlogit > 0 and len(dae_optimizer.param_groups) > 1:
            lr_mlogit = args.learning_rate_mlogit * \
                float(global_step) / warmup_iters
            for i, param_group in enumerate(dae_optimizer.param_groups):
                if i == 0:
                    param_group['lr'] = lr_mlogit
                else:
                    param_group['lr'] = lr
        else:
            for param_group in dae_optimizer.param_groups:
                param_group['lr'] = lr
        # use same lr if lr for local-dae is not specified
        lr = lr if args.learning_rate_dae_local <= 0 else args.learning_rate_dae_local * \
            float(global_step) / warmup_iters
        if dae_local_optimizer is not None:
            for param_group in dae_local_optimizer.param_groups:
                param_group['lr'] = lr

        if args.train_vae:
            lr = args.learning_rate_vae * float(global_step) / warmup_iters
            for param_group in vae_optimizer.param_groups:
                param_group['lr'] = lr


def start_meters():
    tr_loss_meter = AvgrageMeter()
    vae_recon_meter = AvgrageMeter()
    vae_kl_meter = AvgrageMeter()
    vae_nelbo_meter = AvgrageMeter()
    kl_per_group_ema = AvgrageMeter()
    return tr_loss_meter, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, kl_per_group_ema


def epoch_logging(args, writer, step, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, tr_loss_meter, kl_per_group_ema):
    average_tensor(vae_recon_meter.avg, args.distributed)
    average_tensor(vae_kl_meter.avg, args.distributed)
    average_tensor(vae_nelbo_meter.avg, args.distributed)
    average_tensor(tr_loss_meter.avg, args.distributed)
    average_tensor(kl_per_group_ema.avg, args.distributed)

    writer.add_scalar('epoch/vae_recon', vae_recon_meter.avg, step)
    writer.add_scalar('epoch/vae_kl', vae_kl_meter.avg, step)
    writer.add_scalar('epoch/vae_nelbo', vae_nelbo_meter.avg, step)
    writer.add_scalar('epoch/total_loss', tr_loss_meter.avg, step)
    # add kl value per group to tensorboard
    for i in range(len(kl_per_group_ema.avg)):
        writer.add_scalar('kl_value/group_%d' %
                          i, kl_per_group_ema.avg[i], step)


def infer_active_variables(train_queue, vae, args, device, distributed, max_iter=None):
    kl_meter = AvgrageMeter()
    vae.eval()
    for step, x in enumerate(train_queue):
        if max_iter is not None and step > max_iter:
            break
        tr_pts = x['tr_points']
        with autocast(enabled=args.autocast_train):
            # apply vae:
            with torch.set_grad_enabled(False):
                # output = model.recont(val_x) ## torch.cat([val_x, tr_x]))
                dist = vae.encode(tr_pts.to(device))
                eps = dist.sample()[0]
                all_log_q = [dist.log_p(eps)]
                ## _, all_log_q, all_eps = vae(x)
                ## all_eps = vae.concat_eps_per_scale(all_eps)
                ## all_log_q = vae.concat_eps_per_scale(all_log_q)
                all_eps = [eps]

                def make_4d(xlist): return [
                    x.unsqueeze(-1).unsqueeze(-1) if len(x.shape) == 2 else x.unsqueeze(-1) for x in xlist]

                log_q, log_p, kl_all, kl_diag = vae_terms(
                    make_4d(all_log_q), make_4d(all_eps))
                kl_meter.update(kl_diag[0], 1)  # only the top scale
    average_tensor(kl_meter.avg, distributed)
    return kl_meter.avg > 0.1
