# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" to train hierarchical VAE model 
this trainer only train the vae without prior 
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
import torch.distributed as dist
from trainers.base_trainer import BaseTrainer
from utils.eval_helper import compute_NLL_metric
from utils import model_helper, exp_helper, data_helper
from utils.checker import *
from utils import utils
from trainers.common_fun import validate_inspect_noprior
from torch.cuda.amp import autocast, GradScaler
import third_party.pvcnn.functional as pvcnn_fn
from calmsize import size as calmsize


class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        """
        Args:
            cfg: training config 
            args: used for distributed training 
        """
        super().__init__(cfg, args)
        self.train_iter_kwargs = {}
        self.sample_num_points = cfg.data.tr_max_sample_points
        device = torch.device('cuda:%d' % args.local_rank)
        self.device_str = 'cuda:%d' % args.local_rank
        if not cfg.trainer.use_grad_scalar:
            self.grad_scalar = utils.DummyGradScalar()
        else:
            logger.info('Init GradScaler!')
            self.grad_scalar = GradScaler(2**10, enabled=True)

        self.model = self.build_model().to(device)
        if len(self.cfg.sde.vae_checkpoint):
            logger.info('Load vae_checkpoint: {}', self.cfg.sde.vae_checkpoint)
            self.model.load_state_dict(
                    torch.load(self.cfg.sde.vae_checkpoint)['model'])

        logger.info('broadcast_params: device={}', device)
        utils.broadcast_params(self.model.parameters(),
                               args.distributed)
        self.build_other_module()
        if args.distributed:
            logger.info('waitting for barrier, device={}', device)
            dist.barrier()
            logger.info('pass barrier, device={}', device)

        self.train_loader, self.test_loader = self.build_data()

        # The optimizer
        self.optimizer, self.scheduler = utils.get_opt(
            self.model.parameters(),
            self.cfg.trainer.opt,
            cfg.ddpm.ema, self.cfg)
        # Build Spectral Norm Regularization if needed
        if self.cfg.trainer.sn_reg_vae:
            raise NotImplementedError

        # Prepare variable for summy
        self.num_points = self.cfg.data.tr_max_sample_points
        logger.info('done init trainer @{}', device)

        # Prepare for evaluation
        # init the latent for validate
        self.prepare_vis_data()
    # ------------------------------------------- #
    #   training fun                              #
    # ------------------------------------------- #

    def epoch_start(self, epoch):
        pass

    def epoch_end(self, epoch, writer=None, **kwargs):
        return super().epoch_end(epoch, writer=writer)

    def train_iter(self, data, *args, **kwargs):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        see get_loss in models/shapelatent_diffusion.py 
        """
        self.model.train()
        step = kwargs.get('step', None)
        assert(step is not None), 'require step as input'
        warmup_iters = len(self.train_loader) * \
            self.cfg.trainer.opt.vae_lr_warmup_epochs
        utils.update_vae_lr(self.cfg, step, warmup_iters, self.optimizer)
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.model.train()
            self.optimizer.zero_grad()
        device = torch.device(self.device_str)
        tr_pts = data['tr_points'].to(device)  # (B, Npoints, 3)
        batch_size = tr_pts.size(0)
        model_kwargs = {}
        with autocast(enabled=self.cfg.sde.autocast_train):
            res = self.model.get_loss(tr_pts, writer=self.writer,
                                      it=step, **model_kwargs)
            loss = res['loss'].mean()
            lossv = loss.detach().cpu().item()

        if not no_update:

            self.grad_scalar.scale(loss).backward()
            utils.average_gradients(self.model.parameters(),
                                    self.args.distributed)
            if self.cfg.trainer.opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.cfg.trainer.opt.grad_clip)
            self.grad_scalar.step(self.optimizer)
            self.grad_scalar.update()

        output = {}
        if self.writer is not None:
            for k, v in res.items():
                if 'print/' in k and step is not None:
                    v0 = v.mean().item() if torch.is_tensor(v) else v
                    self.writer.avg_meter(k.split('print/')[-1], v0,
                                          step=step)
                if 'hist/' in k:
                    output[k] = v

        output.update({
            'loss': lossv,
            'x_0_pred': res['x_0_pred'].detach().cpu(),  # perturbed data
            'x_0': res['x_0'].detach().cpu(),
            'x_t': res['final_pred'].detach().view(batch_size, -1, res['x_0'].shape[-1]),
            't': res.get('t', None)
        })
        for k, v in res.items():
            if 'vis/' in k or 'msg/' in k:
                output[k] = v
        # if 'x_ref_pred' in res:
        #     output['x_ref_pred'] = res['x_ref_pred'].detach().cpu()
        # if 'x_ref_pred_input' in res:
        #     output['x_ref_pred_input'] = res['x_ref_pred_input'].detach().cpu()
        return output
    # --------------------------------------------- #
    #   visulization function and sampling function #
    # --------------------------------------------- #

    @torch.no_grad()
    def vis_diffusion(self, data, writer):
        pass

    def diffusion_sample(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def vis_sample(self, writer, num_vis=None, step=0, include_pred_x0=True,
                   save_file=None):
        bound = 1.5 if 'chair' in self.cfg.data.cates else 1.0
        assert(not self.cfg.data.cond_on_cat)
        val_class_label = tr_class_label = None
        validate_inspect_noprior(self.model,
                                 step, self.writer, self.sample_num_points,
                                 need_sample=0, need_val=1, need_train=0,
                                 num_samples=self.num_val_samples,
                                 test_loader=self.test_loader,
                                 w_prior=self.w_prior,
                                 val_x=self.val_x, tr_x=self.tr_x,
                                 val_class_label=val_class_label,
                                 tr_class_label=tr_class_label,
                                 has_shapelatent=True,
                                 bound=bound, cfg=self.cfg
                                 )

    @torch.no_grad()
    def sample(self, num_shapes=2, num_points=2048, device_str='cuda',
               for_vis=True, use_ddim=False, save_file=None, ddim_step=500):
        """ return the final samples in shape [B,3,N] """
        # switch to EMA parameters
        if self.cfg.ddpm.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        self.model.eval()

        # ---- forward sampling ---- #
        gen_x = self.model.sample(
            num_samples=num_shapes, device_str=self.device_str)
        # gen_x: BNC
        CHECKEQ(gen_x.shape[2], self.cfg.ddpm.input_dim)
        traj = gen_x.permute(0, 2, 1).contiguous()  # BN3->B3N

        # switch back to original parameters
        if self.cfg.ddpm.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        return traj
