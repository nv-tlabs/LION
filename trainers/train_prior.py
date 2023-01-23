# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" to train hierarchical VAE model with single prior """
import os
import time
from PIL import Image
import gc
import psutil
import functools
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from loguru import logger
import torch.distributed as dist
from torch import optim
from trainers.base_trainer import BaseTrainer
from utils.ema import EMA
from utils.model_helper import import_model, loss_fn
from utils.vis_helper import visualize_point_clouds_3d
from utils.eval_helper import compute_NLL_metric 
from utils import model_helper, exp_helper, data_helper
from utils.data_helper import normalize_point_clouds
from utils.diffusion_pvd import DiffusionDiscretized
from utils.diffusion_continuous import make_diffusion, DiffusionBase
from utils.checker import *
from utils import utils
from matplotlib import pyplot as plt
import third_party.pvcnn.functional as pvcnn_fn
from timeit import default_timer as timer
from torch.optim import Adam as FusedAdam
from torch.cuda.amp import autocast, GradScaler
from trainers import common_fun_prior_train


@torch.no_grad()
def generate_samples_vada(shape, dae, diffusion, vae, num_samples,
                          enable_autocast, ode_eps=0.00001, ode_solver_tol=1e-5,  # None,
                          ode_sample=False, prior_var=1.0, temp=1.0, vae_temp=1.0,
                          noise=None, need_denoise=False, ddim_step=0, clip_feat=None):
    output = {}
    if ode_sample == 1:
        assert isinstance(
            diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
        assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
        assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
        start = timer()
        eps, eps_list, nfe, time_ode_solve = diffusion.sample_model_ode(
            dae, num_samples, shape, ode_eps,
            ode_solver_tol, enable_autocast, temp, noise, return_all_sample=True)
        output['sampled_eps'] = eps
        output['eps_list'] = eps_list
        logger.info('ode_eps={}', ode_eps)
    elif ode_sample == 0:
        assert isinstance(
            diffusion, DiffusionDiscretized), 'Regular sampling requires disc. diffusion!'
        assert noise is None, 'Noise is not used in ancestral sampling.'
        nfe = diffusion._diffusion_steps
        time_ode_solve = 999.999  # Yeah I know...
        start = timer()
        if ddim_step > 0:
            eps, eps_list = diffusion.run_ddim(dae,
                                               num_samples, shape, temp, enable_autocast,
                                               is_image=False, prior_var=prior_var, ddim_step=ddim_step)
        else:
            eps, eps_list = diffusion.run_denoising_diffusion(dae,
                                                              num_samples, shape, temp, enable_autocast,
                                                              is_image=False, prior_var=prior_var)
        output['sampled_eps'] = eps  # latent pts
        output['eps_list'] = eps_list
    else:
        raise NotImplementedError
    output['print/sample_mean_global'] = eps.view(
        num_samples, -1).mean(-1).mean()
    output['print/sample_var_global'] = eps.view(
        num_samples, -1).var(-1).mean()
    decomposed_eps = vae.decompose_eps(eps)
    image = vae.sample(num_samples=num_samples, decomposed_eps=decomposed_eps)

    end = timer()
    sampling_time = end - start
    # average over GPUs
    nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
    sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
    time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
    return image, nfe_torch, time_ode_solve_torch, sampling_time_torch, output


@torch.no_grad()
def validate_inspect(latent_shape,
                     model, dae, diffusion, ode_sample,
                     it, writer,
                     sample_num_points, num_samples,
                     autocast_train=False,
                     need_sample=1, need_val=1, need_train=1,
                     w_prior=None, val_x=None, tr_x=None,
                     val_input=None,
                     m_pcs=None, s_pcs=None,
                     test_loader=None,  # can be None
                     has_shapelatent=False, vis_latent_point=False,
                     ddim_step=0, epoch=0, fun_generate_samples_vada=None, clip_feat=None,
                     cls_emb=None, cfg={}):
    """ visualize the samples, and recont if needed 
    Args:
       has_shapelatent (bool): True when the model has shape latent  
       it (int): step index 
       num_samples: 
       need_* : draw samples for * or not 
    """
    assert(has_shapelatent)
    assert(w_prior is not None and val_x is not None and tr_x is not None)
    z_list = []
    num_samples = w_prior.shape[0] if need_sample else 0
    num_recon = val_x.shape[0]
    num_recon_val = num_recon if need_val else 0
    num_recon_train = num_recon if need_train else 0
    kwargs = {}
    assert(need_sample >= 0 and need_val > 0 and need_train == 0)
    if need_sample:
        # gen_x: B,N,3
        gen_x, nstep, ode_time, sample_time, output_dict = \
            fun_generate_samples_vada(latent_shape, dae, diffusion,
                                      model, w_prior.shape[0], enable_autocast=autocast_train,
                                      ode_sample=ode_sample, ddim_step=ddim_step, clip_feat=clip_feat,
                                      **kwargs)
        logger.info('cast={}, sample step={}, ode_time={}, sample_time={}',
                    autocast_train,
                    nstep if ddim_step == 0 else ddim_step,
                    ode_time, sample_time)
        gen_pcs = gen_x
    else:
        output_dict = {}
    vis_order = cfg.viz.viz_order
    vis_args = {'vis_order': vis_order,
                }
    # vis the samples
    if not vis_latent_point and num_samples > 0:
        img_list = []
        for i in range(num_samples):
            points = gen_x[i]  # N,3
            points = normalize_point_clouds([points])[0]
            img = visualize_point_clouds_3d([points], **vis_args)
            img_list.append(img)
        img = np.concatenate(img_list, axis=2)
        writer.add_image('sample', torch.as_tensor(img), it)

    # vis the latent points 
    if vis_latent_point and num_samples > 0:
        img_list = []
        eps_list = []
        eps = output_dict['sampled_eps'].view(
            num_samples, dae.num_points, dae.num_classes)[:, :, :3]
        for i in range(num_samples):
            points = gen_x[i]
            points = normalize_point_clouds([points])[0]
            img = visualize_point_clouds_3d([points], **vis_args)
            img_list.append(img)

            points = eps[i]
            points = normalize_point_clouds([points])[0]
            img = visualize_point_clouds_3d([points], **vis_args)
            eps_list.append(img)
        img = np.concatenate(img_list, axis=2)
        img_eps = np.concatenate(eps_list, axis=2)
        img = np.concatenate([img, img_eps], axis=1)
        writer.add_image('sample', torch.as_tensor(img), it)
    logger.info('call recont')
    inputs = val_input if val_input is not None else val_x
    output = model.recont(inputs) if cls_emb is None else model.recont(
        inputs, cls_emb=cls_emb)
    gen_x = output['final_pred']

    # vis the recont
    if num_recon_val > 0:
        img_list = []
        for i in range(num_recon_val):
            points = gen_x[i]
            points = normalize_point_clouds([points])
            img = visualize_point_clouds_3d(points, ['rec#%d' % i], **vis_args)
            img_list.append(img)
        gt_list = []
        for i in range(num_recon_val):
            points = normalize_point_clouds([val_x[i]])
            img = visualize_point_clouds_3d(points, ['gt#%d' % i], **vis_args)
            gt_list.append(img)
        img = np.concatenate(img_list, axis=2)
        gt = np.concatenate(gt_list, axis=2)
        img = np.concatenate([gt, img], axis=1)

        if 'vis/latent_pts' in output:
            # also vis the input, used when we take voxel points as input
            input_list = []
            for i in range(num_recon_val):
                points = output['vis/latent_pts'][i, :, :3]
                points = normalize_point_clouds([points])
                input_img = visualize_point_clouds_3d(
                    points, ['input#%d' % i], **vis_args)
                input_list.append(input_img)
            input_list = np.concatenate(input_list, axis=2)
            img = np.concatenate([img, input_list], axis=1)
        writer.add_image('valrecont', torch.as_tensor(img), it)

    if num_recon_train > 0:
        img_list = []
        for i in range(num_recon_train):
            points = gen_x[num_recon_val + i]
            points = normalize_point_clouds([tr_x[i], points])
            img = visualize_point_clouds_3d(points, ['ori', 'rec'], **vis_args)
            img_list.append(img)
        img = np.concatenate(img_list, axis=2)
        writer.add_image('train/recont', torch.as_tensor(img), it)

    logger.info('writer: {}', writer.url)
    return output_dict


class Trainer(BaseTrainer):
    is_diffusion = 0

    def __init__(self, cfg, args):
        """
        Args:
            cfg: training config 
            args: used for distributed training 
        """
        super().__init__(cfg, args)
        self.draw_sample_when_vis = 1
        self.fun_generate_samples_vada = functools.partial(
            generate_samples_vada, ode_eps=cfg.sde.ode_eps)
        self.train_iter_kwargs = {}
        self.cfg.sde.distributed = args.distributed
        self.sample_num_points = cfg.data.tr_max_sample_points
        self.model_var_type = cfg.ddpm.model_var_type
        self.clip_denoised = cfg.ddpm.clip_denoised
        self.num_steps = cfg.ddpm.num_steps
        self.model_mean_type = cfg.ddpm.model_mean_type
        self.loss_type = cfg.ddpm.loss_type
        device = torch.device(self.device_str)

        self.model = self.build_model().to(device)
        if len(self.cfg.sde.vae_checkpoint) and not args.pretrained and self.cfg.sde.vae_checkpoint != 'none':
            # if has pretrained ckpt, we dont need to load the vae ckpt anymore
            logger.info('Load vae_checkpoint: {}', self.cfg.sde.vae_checkpoint)
            vae_ckpt = torch.load(self.cfg.sde.vae_checkpoint)
            vae_weight = vae_ckpt['model']
            self.model.load_state_dict(vae_weight)

        if self.cfg.shapelatent.model == 'models.hvae_ddpm':
            self.model.build_other_module(device)
        logger.info('broadcast_params: device={}', device)
        utils.broadcast_params(self.model.parameters(),
                               args.distributed)
        self.build_other_module()
        self.build_prior()

        if args.distributed:
            logger.info('waitting for barrier, device={}', device)
            dist.barrier()
            logger.info('pass barrier, device={}', device)

        self.train_loader, self.test_loader = self.build_data()
        # The optimizer
        self.init_optimizer()
        # Prepare variable for summy
        self.num_points = self.cfg.data.tr_max_sample_points
        logger.info('done init trainer @{}', device)

        # Prepare for evaluation
        # init the latent for validate
        self.prepare_vis_data()
        self.alpha_i = utils.kl_balancer_coeff(
            num_scales=2,
            groups_per_scale=[1, 1], fun='square')

    @property
    def vae(self):
        return self.model

    def init_optimizer(self):
        out_dict = common_fun_prior_train.init_optimizer_train_2prior(
            self.cfg, self.vae, self.dae)
        self.dae_sn_calculator, self.vae_sn_calculator = out_dict[
            'dae_sn_calculator'], out_dict['vae_sn_calculator']
        self.vae_scheduler, self.vae_optimizer = out_dict['vae_scheduler'], out_dict['vae_optimizer']
        self.dae_scheduler, self.dae_optimizer = out_dict['dae_scheduler'], out_dict['dae_optimizer']
        self.grad_scalar = out_dict['grad_scalar']

    def resume(self, path, strict=True, **kwargs):
        dae, vae = self.dae, self.vae
        vae_optimizer, vae_scheduler, dae_optimizer, dae_scheduler = \
            self.vae_optimizer, self.vae_scheduler, self.dae_optimizer, self.dae_scheduler
        grad_scalar = self.grad_scalar

        checkpoint = torch.load(path, map_location='cpu')
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        dae.load_state_dict(checkpoint['dae_state_dict'])
        # load dae
        dae = dae.cuda()
        dae_optimizer.load_state_dict(checkpoint['dae_optimizer'])
        dae_scheduler.load_state_dict(checkpoint['dae_scheduler'])
        # load vae
        if self.cfg.eval.load_other_vae_ckpt:
            raise NotImplementedError
        else:
            vae.load_state_dict(checkpoint['vae_state_dict'])
            vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        vae = vae.cuda()

        # need to commend if load regular vae from voxel2input_ada trainer
        vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        global_step = checkpoint['global_step']
        ## logger.info('loaded the model at epoch %d.'%init_epoch)

        start_epoch = epoch
        self.epoch = start_epoch
        self.step = global_step
        logger.info('resumedd from : {}, epo={}', path, start_epoch)
        return start_epoch

    def save(self, save_name=None, epoch=None, step=None, appendix=None, save_dir=None, **kwargs):
        dae, vae = self.dae, self.vae
        vae_optimizer, vae_scheduler, dae_optimizer, dae_scheduler = \
            self.vae_optimizer, self.vae_scheduler, self.dae_optimizer, self.dae_scheduler
        grad_scalar = self.grad_scalar
        content = {'epoch': epoch + 1, 'global_step': step, 
                   # 'args': self.cfg.sde, 'cfg': self.cfg,
                   'grad_scalar': grad_scalar.state_dict(),
                   'dae_state_dict': dae.state_dict(), 'dae_optimizer': dae_optimizer.state_dict(),
                   'dae_scheduler': dae_scheduler.state_dict(), 'vae_state_dict': vae.state_dict(),
                   'vae_optimizer': vae_optimizer.state_dict(), 'vae_scheduler': vae_scheduler.state_dict()}
        if appendix is not None:
            content.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (
            epoch, step) if save_name is None else save_name
        if save_dir is None:
            save_dir = self.cfg.save_dir
        path = os.path.join(save_dir, "checkpoints", save_name)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        logger.info('save model as : {}', path)
        torch.save(content, path)
        return path

    def epoch_start(self, epoch):
        if epoch > self.cfg.sde.warmup_epochs:
            self.dae_scheduler.step()
            self.vae_scheduler.step()

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
        if self.cfg.sde.ode_sample == 1:
            diffusion = self.diffusion_cont
        elif self.cfg.sde.ode_sample == 0:
            diffusion = self.diffusion_disc
        elif self.cfg.sde.ode_sample == 2:
            raise NotImplementedError
            ## diffusion = [self.diffusion_cont, self.diffusion_disc]

        ## diffusion = self.diffusion_cont if self.cfg.sde.ode_sample else self.diffusion_disc
        B = tr_pts.size(0)
        with torch.set_grad_enabled(args.train_vae):
            with autocast(enabled=args.autocast_train):
                # posterior and likelihood
                if not args.train_vae:
                    dist = vae.encode(tr_pts)
                    eps = dist.sample()[0]  # B,D or B,N,D or BN,D
                    all_log_q = [dist.log_p(eps)]
                    x_0_pred = x_0_target = tr_pts
                    vae_recon_loss = 0

                    def make_4d(
                        x): return x.unsqueeze(-1).unsqueeze(-1) if len(x.shape) == 2 else x.unsqueeze(-1)
                    eps = make_4d(eps)
                    output = {'eps': eps, 'q_loss': torch.zeros(1),
                              'x_0_pred': tr_pts, 'x_0_target': tr_pts,
                              'x_0': tr_pts, 'final_pred': tr_pts}
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
        # the noisy points, used in trainers/voxel2pts.py and trainers/voxel2pts_ada.py
        inputs = data['input_pts'].to(device) if 'input_pts' in data else None
        B = batch_size = tr_pts.size(0)

        # optimize vae params
        vae_optimizer.zero_grad()
        output = self.compute_loss_vae(tr_pts, global_step, inputs=inputs)

        # backpropagate q_loss for vae and update vae params, if trained
        if args.train_vae:
            q_loss = output['q_loss']
            loss = q_loss
            grad_scalar.scale(q_loss).backward()
            utils.average_gradients(vae.parameters(), distributed)
            if args.grad_clip_max_norm > 0.:  # apply gradient clipping
                grad_scalar.unscale_(vae_optimizer)
                torch.nn.utils.clip_grad_norm_(vae.parameters(),
                                               max_norm=args.grad_clip_max_norm)
            grad_scalar.step(vae_optimizer)

        # train prior
        if args.train_dae:
            # the interface between VAE and DAE is eps.
            eps = output['eps'].detach()  # 4d: B,D,-1,1
            CHECK4D(eps)
            dae_optimizer.zero_grad()
            with autocast(enabled=args.autocast_train):
                noise_p = torch.randn(size=eps.size(), device=device)
                # get diffusion quantities for p sampling scheme and reweighting for q
                t_p, var_t_p, m_t_p, obj_weight_t_p, _, g2_t_p = \
                    diffusion.iw_quantities(B, args.time_eps,
                                            args.iw_sample_p, args.iw_subvp_like_vp_sde)
                # logger.info('t_p: {}, var: {}, m_t: {}', t_p[0], var_t_p[0], m_t_p[0])
                eps_t_p = diffusion.sample_q(eps, noise_p, var_t_p, m_t_p)
                # run the score model
                eps_t_p.requires_grad_(True)
                mixing_component = diffusion.mixing_component(
                    eps_t_p, var_t_p, t_p, enabled=args.mixed_prediction)
                pred_params_p = dae(eps_t_p, t_p, x0=eps)

                # pred_eps_t0 = (eps_t_p - torch.sqrt(var_t_p) * noise_p) / m_t_p  # this will recover the true eps
                pred_eps_t0 = (eps_t_p - torch.sqrt(var_t_p)
                               * pred_params_p) / m_t_p
                params = utils.get_mixed_prediction(args.mixed_prediction,
                                                    pred_params_p, dae.mixing_logit, mixing_component)
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
                    if args.regularize_mlogit:
                        reg_mlogit = ((torch.sum(torch.sigmoid(dae.mixing_logit)) -
                                       args.regularize_mlogit_margin)**2) * args.regularize_mlogit
                    else:
                        reg_mlogit = 0
                    p_loss = torch.mean(p_objective) + \
                        regularization_p + reg_mlogit

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
                        m = torch.sigmoid(dae.mixing_logit)
                        if not torch.isnan(m).any():
                            writer.add_histogram(
                                'mixing_prob', m.detach().cpu().numpy(), global_step)

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
                if 'hist/' in k:
                    output[k] = v
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
            if 'vis/' in k or 'msg/' in k:
                output_dict[k] = v
        return output_dict
    # --------------------------------------------- #
    #   visulization function and sampling function #
    # --------------------------------------------- #

    @torch.no_grad()
    def vis_sample(self, writer, num_vis=None, step=0, include_pred_x0=True,
                   save_file=None):
        if self.cfg.ddpm.ema:
            self.swap_vae_param_if_need()
            self.dae_optimizer.swap_parameters_with_ema(
                store_params_in_ema=True)
        shape = self.model.latent_shape()
        logger.info('Latent shape for prior: {}; num_val_samples: {}',
                    shape, self.num_val_samples)
        # [self.vae.latent_dim, .num_input_channels, dae.input_size, dae.input_size]
        ode_sample = self.cfg.sde.ode_sample
        ## diffusion = self.diffusion_cont if ode_sample else self.diffusion_disc
        if self.cfg.sde.ode_sample == 1:
            diffusion = self.diffusion_cont
        elif self.cfg.sde.ode_sample == 0:
            diffusion = self.diffusion_disc
        if self.cfg.clipforge.enable:
            assert(self.clip_feat_test is not None)
        kwargs = {}
        output = validate_inspect(shape, self.model, self.dae,
                                  diffusion, ode_sample,
                                  step, self.writer, self.sample_num_points,
                                  epoch=self.cur_epoch,
                                  autocast_train=self.cfg.sde.autocast_train,
                                  need_sample=self.draw_sample_when_vis,
                                  need_val=1, need_train=0,
                                  num_samples=self.num_val_samples,
                                  test_loader=self.test_loader,
                                  w_prior=self.w_prior,
                                  val_x=self.val_x, tr_x=self.tr_x,
                                  val_input=self.val_input,
                                  m_pcs=self.m_pcs, s_pcs=self.s_pcs,
                                  has_shapelatent=True,
                                  vis_latent_point=self.cfg.vis_latent_point,
                                  ddim_step=self.cfg.viz.vis_sample_ddim_step,
                                  clip_feat=self.clip_feat_test,
                                  cfg=self.cfg,
                                  fun_generate_samples_vada=self.fun_generate_samples_vada,
                                  **kwargs
                                  )
        if writer is not None:
            for n, v in output.items():
                if 'print/' not in n:
                    continue
                self.writer.add_scalar('%s' % (n.split('print/')[-1]), v, step)

        if self.cfg.ddpm.ema:
            self.swap_vae_param_if_need()
            self.dae_optimizer.swap_parameters_with_ema(
                store_params_in_ema=True)

    @torch.no_grad()
    def sample(self, num_shapes=2, num_points=2048, device_str='cuda',
               for_vis=True, use_ddim=False, save_file=None, ddim_step=0, clip_feat=None):
        """ return the final samples in shape [B,3,N] """
        # switch to EMA parameters
        assert(
            not self.cfg.clipforge.enable), f'not suuport yet, not sure what the clip feat will be'
        cfg = self.cfg
        if cfg.ddpm.ema:
            self.swap_vae_param_if_need()
            self.dae_optimizer.swap_parameters_with_ema(
                store_params_in_ema=True)
        self.model.eval()  # Draw sample under train mode
        S = self.num_steps
        logger.info('num_shapes={}, num_points={}, use_ddim={}, Nstep={}',
                    num_shapes, num_points, use_ddim, S)
        latent_shape = self.model.latent_shape()

        ode_sample = self.cfg.sde.ode_sample
        ## diffusion = self.diffusion_cont if ode_sample else self.diffusion_disc
        if self.cfg.sde.ode_sample == 1:
            diffusion = self.diffusion_cont
        elif self.cfg.sde.ode_sample == 0:
            diffusion = self.diffusion_disc
        elif self.cfg.sde.ode_sample == 2:
            diffusion = [self.diffusion_cont, self.diffusion_disc]

        # ---- forward sampling ---- #
        gen_x, nstep, ode_time, sample_time, output_fsample = \
            self.fun_generate_samples_vada(latent_shape, self.dae,
                                           diffusion, self.model, num_shapes,
                                           enable_autocast=self.cfg.sde.autocast_train,
                                           ode_sample=ode_sample,
                                           need_denoise=self.cfg.eval.need_denoise,
                                           ddim_step=ddim_step,
                                           clip_feat=clip_feat)
        # gen_x: BNC
        CHECKEQ(gen_x.shape[2], self.cfg.ddpm.input_dim)
        if gen_x.shape[1] > self.sample_num_points:
            gen_x = pvcnn_fn.furthest_point_sample(gen_x.permute(0, 2, 1).contiguous(),
                                                   self.sample_num_points).permute(0, 2, 1).contiguous()  # [B,C,npoint]

        traj = gen_x.permute(0, 2, 1).contiguous()  # BN3->B3N

        # ---- debug perpuse ---- #
        if save_file:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            torch.save(traj.permute(0, 2, 1), save_file)
            exit()

        # switch back to original parameters
        if cfg.ddpm.ema:
            self.dae_optimizer.swap_parameters_with_ema(
                store_params_in_ema=True)
            self.swap_vae_param_if_need()
        return traj

    def build_prior(self):
        args = self.cfg.sde
        device = torch.device(self.device_str)
        arch_instance_dae = utils.get_arch_cells_denoising(
            'res_ho_attn', True, False)
        num_input_channels = self.cfg.shapelatent.latent_dim

        if self.cfg.sde.hier_prior:
            if self.cfg.sde.prior_model == 'sim':
                DAE = NCSNppPointHie
            else:
                DAE = import_model(self.cfg.sde.prior_model)
        elif self.cfg.sde.prior_model == 'sim':
            DAE = NCSNppPoint
        else:
            DAE = import_model(self.cfg.sde.prior_model)

        self.dae = DAE(args, num_input_channels, self.cfg).to(device)
        if len(self.cfg.sde.dae_checkpoint):
            logger.info('Load dae checkpoint: {}',
                        self.cfg.sde.dae_checkpoint)
            checkpoint = torch.load(
                self.cfg.sde.dae_checkpoint, map_location='cpu')
            self.dae.load_state_dict(checkpoint['dae_state_dict'])

        self.diffusion_cont = make_diffusion(args)
        self.diffusion_disc = DiffusionDiscretized(
            args, self.diffusion_cont.var, self.cfg)
        logger.info('DAE: {}', self.dae)
        logger.info('DAE: param size = %fM ' %
                    utils.count_parameters_in_M(self.dae))

        ## self.check_consistence(self.diffusion_cont, self.diffusion_disc)
        # sync all parameters between all gpus by sending param from rank 0 to all gpus.
        utils.broadcast_params(self.dae.parameters(), self.args.distributed)

    def swap_vae_param_if_need(self):
        if self.cfg.eval.load_other_vae_ckpt:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
