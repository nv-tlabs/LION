# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" to train hierarchical VAE model with 2 prior 
one for style latent, one for latent pts, 
based on trainers/train_prior.py 
"""
import os
import time
from PIL import Image
import gc
import functools
import psutil
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from loguru import logger
import torch.distributed as dist
from torch import optim
from utils.ema import EMA
from utils.model_helper import import_model, loss_fn
from utils.vis_helper import visualize_point_clouds_3d
from utils.eval_helper import compute_NLL_metric 
from utils import model_helper, exp_helper, data_helper
from utils.data_helper import normalize_point_clouds
## from utils.diffusion_discretized import DiffusionDiscretized
from utils.diffusion_pvd import DiffusionDiscretized
from utils.diffusion_continuous import make_diffusion, DiffusionBase
from utils.checker import *
from utils import utils
from matplotlib import pyplot as plt
import third_party.pvcnn.functional as pvcnn_fn
from timeit import default_timer as timer
from torch.optim import Adam as FusedAdam
from torch.cuda.amp import autocast, GradScaler
from trainers.train_prior import Trainer as PriorTrainer
from trainers.train_prior import validate_inspect  # import Trainer as PriorTrainer

quiet = int(os.environ.get('quiet', 0))
VIS_LATENT_PTS = 0


@torch.no_grad()
def generate_samples_vada_2prior(shape, dae, diffusion, vae, num_samples, enable_autocast,
                                 ode_eps=0.00001, ode_solver_tol=1e-5,  # None,
                                 ode_sample=False, prior_var=1.0, temp=1.0, vae_temp=1.0, noise=None, need_denoise=False,
                                 ddim_step=0, clip_feat=None, cls_emb=None, ddim_skip_type='uniform', ddim_kappa=1.0):
    output = {}
    #kwargs = {}
    # if cls_emb is not None:
    #    kwargs['cls_emb'] = cls_emb
    if ode_sample == 1:
        assert isinstance(
            diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
        assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
        assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
        start = timer()
        condition_input = None
        eps_list = []
        for i in range(len(dae)):
            assert(cls_emb is None), f' not support yet'
            eps, nfe, time_ode_solve = diffusion.sample_model_ode(
                dae[i], num_samples, shape[i], ode_eps, ode_solver_tol, enable_autocast, temp, noise,
                condition_input=condition_input, clip_feat=clip_feat,
            )
            condition_input = eps
            eps_list.append(eps)
            output['sampled_eps'] = eps
        eps = vae.compose_eps(eps_list)  # torch.cat(eps, dim=1)

    elif ode_sample == 0:
        assert isinstance(
            diffusion, DiffusionDiscretized), 'Regular sampling requires disc. diffusion!'
        assert noise is None, 'Noise is not used in ancestral sampling.'
        nfe = diffusion._diffusion_steps
        time_ode_solve = 999.999  # Yeah I know...
        start = timer()
        condition_input = None if cls_emb is None else cls_emb
        all_eps = []
        for i in range(len(dae)):
            if ddim_step > 0:
                assert(cls_emb is None), f'not support yet'
                eps, eps_list = diffusion.run_ddim(dae[i],
                                                   num_samples, shape[i], temp, enable_autocast,
                                                   is_image=False, prior_var=prior_var, ddim_step=ddim_step,
                                                   condition_input=condition_input, clip_feat=clip_feat,
                                                   skip_type=ddim_skip_type, kappa=ddim_kappa)
            else:
                eps, eps_list = diffusion.run_denoising_diffusion(dae[i],
                                                                  num_samples, shape[i], temp, enable_autocast,
                                                                  is_image=False, prior_var=prior_var,
                                                                  condition_input=condition_input, clip_feat=clip_feat,
                                                                  )
            condition_input = eps

            if cls_emb is not None:
                condition_input = torch.cat([condition_input,
                                             cls_emb.unsqueeze(-1).unsqueeze(-1)], dim=1)
            if i == 0:
                condition_input = vae.global2style(condition_input)
            # exit()
            all_eps.append(eps)

            output['sampled_eps'] = eps
        eps = vae.compose_eps(all_eps)
        output['eps_list'] = eps_list
    output['print/sample_mean_global'] = eps.view(
        num_samples, -1).mean(-1).mean()
    output['print/sample_var_global'] = eps.view(
        num_samples, -1).var(-1).mean()
    decomposed_eps = vae.decompose_eps(eps)
    image = vae.sample(num_samples=num_samples,
                       decomposed_eps=decomposed_eps, cls_emb=cls_emb)

    end = timer()
    sampling_time = end - start
    # average over GPUs
    nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
    sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
    time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
    return image, nfe_torch, time_ode_solve_torch, sampling_time_torch, output


class Trainer(PriorTrainer):
    is_diffusion = 0

    def __init__(self, cfg, args):
        """
        Args:
            cfg: training config 
            args: used for distributed training 
        """
        super().__init__(cfg, args)
        self.fun_generate_samples_vada = functools.partial(
            generate_samples_vada_2prior, ode_eps=cfg.sde.ode_eps,
            ddim_skip_type=cfg.sde.ddim_skip_type,
            ddim_kappa=cfg.sde.ddim_kappa)

    def compute_loss_vae(self, tr_pts, global_step, **kwargs):
        """ compute forward for VAE model, used in global-only prior training 
        Input: 
            tr_pts: points 
            global_step: int 
        Returns: 
            output dict including entry: 
            'eps': z ~ posterior 
            'q_loss': 0 if not train vae else the KL+rec 
            'x_0_pred': global points if not train vae 
            'x_0_target': target points 

        """
        vae = self.model
        dae = self.dae
        args = self.cfg.sde
        distributed = args.distributed
        vae_sn_calculator = self.vae_sn_calculator
        num_total_iter = self.num_total_iter
        ## diffusion = self.diffusion_cont if self.cfg.sde.ode_sample else self.diffusion_disc
        if self.cfg.sde.ode_sample == 1:
            diffusion = self.diffusion_cont
        elif self.cfg.sde.ode_sample == 0:
            diffusion = self.diffusion_disc
        elif self.cfg.sde.ode_sample == 2:
            raise NotImplementedError
            # diffusion = [self.diffusion_cont, self.diffusion_disc]

        B = tr_pts.size(0)
        with torch.set_grad_enabled(args.train_vae):
            with autocast(enabled=args.autocast_train):
                # posterior and likelihood
                if not args.train_vae:
                    output = {}
                    all_eps, all_log_q, latent_list = vae.encode(tr_pts)
                    x_0_pred = x_0_target = tr_pts
                    vae_recon_loss = 0
                    def make_4d(x): return x.unsqueeze(-1).unsqueeze(-1) if \
                        len(x.shape) == 2 else x.unsqueeze(-1)
                    eps = make_4d(all_eps)
                    output.update({'eps': eps, 'q_loss': torch.zeros(1),
                                   'x_0_pred': tr_pts, 'x_0_target': tr_pts,
                                   'x_0': tr_pts, 'final_pred': tr_pts})
                else:
                    raise NotImplementedError
        return output
    # ------------------------------------------- #
    #   training fun                              #
    # ------------------------------------------- #

    def train_iter(self, data, *args, **kwargs):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        see get_loss in models/shapelatent_diffusion.py 
        """
        # some variables

        input_dim = self.cfg.ddpm.input_dim
        loss_type = self.cfg.ddpm.loss_type
        vae = self.model
        dae = self.dae
        dae.train()
        diffusion = self.diffusion_cont if self.cfg.sde.ode_sample else self.diffusion_disc
        if self.cfg.sde.ode_sample == 1:
            diffusion = self.diffusion_cont
        elif self.cfg.sde.ode_sample == 0:
            diffusion = self.diffusion_disc
        elif self.cfg.sde.ode_sample == 2:
            raise NotImplementedError  # not support training with different solver
            ## diffusion = [self.diffusion_cont, self.diffusion_disc]

        dae_optimizer = self.dae_optimizer
        vae_optimizer = self.vae_optimizer
        args = self.cfg.sde
        device = torch.device(self.device_str)
        num_total_iter = self.num_total_iter
        distributed = self.args.distributed
        dae_sn_calculator = self.dae_sn_calculator
        vae_sn_calculator = self.vae_sn_calculator
        grad_scalar = self.grad_scalar

        global_step = step = kwargs.get('step', None)
        no_update = kwargs.get('no_update', False)

        # update_lr
        warmup_iters = len(self.train_loader) * args.warmup_epochs
        utils.update_lr(args, global_step, warmup_iters,
                        dae_optimizer, vae_optimizer)

        # input
        tr_pts = data['tr_points'].to(device)  # (B, Npoints, 3)
        inputs = data['input_pts'].to(
            device) if 'input_pts' in data else None  # the noisy points
        tr_img = data['tr_img'].to(device) if 'tr_img' in data else None
        model_kwargs = {}
        if self.cfg.data.cond_on_cat:
            class_label_int = data['cate_idx'].view(-1)  # .to(device)
            nclass = self.cfg.data.nclass
            class_label = torch.nn.functional.one_hot(class_label_int, nclass)
            model_kwargs['class_label'] = class_label.float().to(device)

        B = batch_size = tr_pts.size(0)
        if tr_img is not None:
            # tr_img: B,nimg,3,H,W
            # logger.info('image: {}', tr_img.shape)
            nimg = tr_img.shape[1]
            tr_img = tr_img.view(B*nimg, *tr_img.shape[2:])
            clip_feat = self.clip_model.encode_image(
                tr_img).view(B, nimg, -1).mean(1).float()
        else:
            clip_feat = None
        if self.cfg.clipforge.enable:
            assert(clip_feat is not None)

        # optimize vae params
        vae_optimizer.zero_grad()
        output = self.compute_loss_vae(
            tr_pts, global_step, inputs=inputs, **model_kwargs)

        # the interface between VAE and DAE is eps.
        eps = output['eps'].detach()  # 4d: B,D,-1,1
        CHECK4D(eps)
        dae_kwarg = {}
        if self.cfg.data.cond_on_cat:
            dae_kwarg['condition_input'] = output['cls_emb']
        # train prior
        if args.train_dae:
            dae_optimizer.zero_grad()
            with autocast(enabled=args.autocast_train):
                # get diffusion quantities for p sampling scheme and reweighting for q
                t_p, var_t_p, m_t_p, obj_weight_t_p, _, g2_t_p = \
                    diffusion.iw_quantities(B, args.time_eps,
                                            args.iw_sample_p, args.iw_subvp_like_vp_sde)
                # logger.info('t_p: {}, var: {}, m_t: {}', t_p[0], var_t_p[0], m_t_p[0])

                decomposed_eps = self.vae.decompose_eps(eps)
                output['vis/eps'] = decomposed_eps[1].view(
                    -1, self.dae.num_points, self.dae.num_classes)[:, :, :3]
                p_loss_list = []
                for latent_id, eps in enumerate(decomposed_eps):

                    noise_p = torch.randn(size=eps.size(), device=device)
                    eps_t_p = diffusion.sample_q(eps, noise_p, var_t_p, m_t_p)
                    # run the score model
                    eps_t_p.requires_grad_(True)
                    mixing_component = diffusion.mixing_component(
                        eps_t_p, var_t_p, t_p, enabled=args.mixed_prediction)
                    if latent_id == 0:
                        pred_params_p = dae[latent_id](
                            eps_t_p, t_p, x0=eps, clip_feat=clip_feat, **dae_kwarg)
                    else:
                        condition_input = decomposed_eps[0] if not self.cfg.data.cond_on_cat else \
                            torch.cat(
                                [decomposed_eps[0], output['cls_emb'].unsqueeze(-1).unsqueeze(-1)], dim=1)
                        condition_input = self.model.global2style(
                            condition_input)
                        pred_params_p = dae[latent_id](eps_t_p, t_p, x0=eps,
                                                       condition_input=condition_input, clip_feat=clip_feat)

                    pred_eps_t0 = (eps_t_p - torch.sqrt(var_t_p)
                                   * pred_params_p) / m_t_p

                    params = utils.get_mixed_prediction(args.mixed_prediction,
                                                        pred_params_p, dae[latent_id].mixing_logit, mixing_component)
                    if self.cfg.latent_pts.pvd_mse_loss:
                        p_loss = F.mse_loss(
                            params.contiguous().view(B, -1), noise_p.view(B, -1),
                            reduction='mean')
                    else:
                        l2_term_p = torch.square(params - noise_p)
                        p_objective = torch.sum(
                            obj_weight_t_p * l2_term_p, dim=[1, 2, 3])
                        regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff, \
                            jac_reg_loss, kin_reg_loss = utils.dae_regularization(
                                args, dae_sn_calculator, diffusion, dae, step, t_p,
                                pred_params_p, eps_t_p, var_t_p, m_t_p, g2_t_p)
                        reg_mlogit = ((torch.sum(torch.sigmoid(dae.mixing_logit)) -
                                       args.regularize_mlogit_margin)**2) * args.regularize_mlogit \
                            if args.regularize_mlogit else 0
                        p_loss = torch.mean(p_objective) + \
                            regularization_p + reg_mlogit
                    if self.writer is not None:
                        self.writer.avg_meter(
                            'train/p_loss_%d' % latent_id, p_loss.detach().item())
                    p_loss_list.append(p_loss)
            p_loss = sum(p_loss_list)  # torch.cat(p_loss_list, dim=0).sum()
            loss = p_loss
            # update dae parameters
            grad_scalar.scale(p_loss).backward()
            utils.average_gradients(dae.parameters(), distributed)
            if args.grad_clip_max_norm > 0.:         # apply gradient clipping
                grad_scalar.unscale_(dae_optimizer)
                torch.nn.utils.clip_grad_norm_(dae.parameters(),
                                               max_norm=args.grad_clip_max_norm)
            grad_scalar.step(dae_optimizer)

            # update grade scalar
            grad_scalar.update()

            if args.bound_mlogit:
                dae.mixing_logit.data.clamp_(max=args.bound_mlogit_value)
            # Bookkeeping!
            writer = self.writer
            if writer is not None:
                writer.avg_meter('train/lr_dae', dae_optimizer.state_dict()[
                    'param_groups'][0]['lr'], global_step)
                writer.avg_meter('train/lr_vae', vae_optimizer.state_dict()[
                    'param_groups'][0]['lr'], global_step)
                if self.cfg.latent_pts.pvd_mse_loss:
                    writer.avg_meter(
                        'train/p_loss', p_loss.item(), global_step)
                    if args.mixed_prediction and global_step % 500 == 0:
                        for i in range(len(dae)):
                            m = torch.sigmoid(dae[i].mixing_logit)
                            if not torch.isnan(m).any():
                                writer.add_histogram(
                                    'mixing_prob_%d' % i, m.detach().cpu().numpy(), global_step)

                    # no other loss
                else:
                    writer.avg_meter(
                        'train/p_loss', (p_loss - regularization_p).item(), global_step)
                    if torch.is_tensor(regularization_p):
                        writer.avg_meter(
                            'train/reg_p', regularization_p.item(), global_step)
                    if args.regularize_mlogit:
                        writer.avg_meter(
                            'train/m_logit', reg_mlogit / args.regularize_mlogit, global_step)
                    if args.mixed_prediction:
                        writer.avg_meter(
                            'train/m_logit_sum', torch.sum(torch.sigmoid(dae.mixing_logit)).detach().cpu(), global_step)
                    if (global_step) % 500 == 0:
                        writer.add_scalar(
                            'train/norm_loss_dae', dae_norm_loss, global_step)
                        writer.add_scalar('train/bn_loss_dae',
                                          dae_bn_loss, global_step)
                        writer.add_scalar(
                            'train/norm_coeff_dae', dae_wdn_coeff, global_step)
                        if args.mixed_prediction:
                            m = torch.sigmoid(dae.mixing_logit)
                            if not torch.isnan(m).any():
                                writer.add_histogram(
                                    'mixing_prob', m.detach().cpu().numpy(), global_step)

        # write stats
        if self.writer is not None:
            for k, v in output.items():
                if 'print/' in k and step is not None:
                    self.writer.avg_meter(k.split('print/')[-1],
                                          v.mean().item() if torch.is_tensor(v) else v,
                                          step=step)
        res = output
        output_dict = {
            'loss': loss.detach().cpu().item(),
            'x_0_pred': res['x_0_pred'].detach().cpu(),  # perturbed data
            'x_0': res['x_0'].detach().cpu(),
            # B.B,3
            'x_t': res['final_pred'].detach().view(batch_size, -1, res['x_0'].shape[-1]),
            't': res.get('t', None)
        }

        for k, v in output.items():
            if 'vis/' in k:
                output_dict[k] = v
        return output_dict
    # --------------------------------------------- #
    #   visulization function and sampling function #
    # --------------------------------------------- #

    def build_prior(self):
        args = self.cfg.sde
        device = torch.device(self.device_str)
        arch_instance_dae = utils.get_arch_cells_denoising(
            'res_ho_attn', True, False)
        num_input_channels = self.cfg.shapelatent.latent_dim

        DAE = nn.ModuleList(
            [
                import_model(self.cfg.latent_pts.style_prior)(args,
                                                              self.cfg.latent_pts.style_dim, self.cfg),  # style prior
                import_model(self.cfg.sde.prior_model)(args,
                                                       num_input_channels, self.cfg),  # global prior, conditional model
            ])

        self.dae = DAE.to(device)

        # Bad solution! it is used in validate_inspect function
        self.dae.num_points = self.dae[1].num_points
        self.dae.num_classes = self.dae[1].num_classes

        if len(self.cfg.sde.dae_checkpoint):
            logger.info('Load dae checkpoint: {}',
                        self.cfg.sde.dae_checkpoint)
            checkpoint = torch.load(
                self.cfg.sde.dae_checkpoint, map_location='cpu')
            self.dae.load_state_dict(checkpoint['dae_state_dict'])

        self.diffusion_cont = make_diffusion(args)
        self.diffusion_disc = DiffusionDiscretized(
            args, self.diffusion_cont.var, self.cfg)
        if not quiet:
            logger.info('DAE: {}', self.dae)
        logger.info('DAE: param size = %fM ' %
                    utils.count_parameters_in_M(self.dae))
        # sync all parameters between all gpus by sending param from rank 0 to all gpus.
        utils.broadcast_params(self.dae.parameters(), self.args.distributed)
