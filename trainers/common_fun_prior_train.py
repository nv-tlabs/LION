# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import numpy as np
import time
from loguru import logger
from utils.ema import EMA
from torch import optim
from torch.optim import Adam as FusedAdam
from torch.cuda.amp import autocast, GradScaler
from utils.sr_utils import SpectralNormCalculator
from utils import utils
from utils.vis_helper import visualize_point_clouds_3d
from utils.diffusion_pvd import DiffusionDiscretized
from utils.eval_helper import compute_NLL_metric 
from utils import model_helper, exp_helper, data_helper
from timeit import default_timer as timer
from utils.data_helper import normalize_point_clouds


def init_optimizer_train_2prior(cfg, vae, dae, cond_enc=None):
    args = cfg.sde
    param_dict_dae = dae.parameters()
    # optimizer for prior
    if args.learning_rate_mlogit > 0:
        raise NotImplementedError
    if args.use_adamax:
        from utils.adamax import Adamax
        dae_optimizer = Adamax(param_dict_dae, args.learning_rate_dae,
                               weight_decay=args.weight_decay, eps=1e-4)
    elif args.use_adam:
        cfgopt = cfg.trainer.opt
        dae_optimizer = optim.Adam(param_dict_dae,
                                   lr=args.learning_rate_dae,
                                   betas=(cfgopt.beta1, cfgopt.beta2),
                                   weight_decay=cfgopt.weight_decay)

    else:
        dae_optimizer = FusedAdam(param_dict_dae, args.learning_rate_dae,
                                  weight_decay=args.weight_decay, eps=1e-4)
    # add EMA functionality to the optimizer
    dae_optimizer = EMA(dae_optimizer, ema_decay=args.ema_decay)
    dae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dae_optimizer, float(args.epochs - args.warmup_epochs - 1),
        eta_min=args.learning_rate_min_dae)

    # optimizer for VAE
    if args.use_adamax:
        from utils.adamax import Adamax
        vae_optimizer = Adamax(vae.parameters(), args.learning_rate_vae,
                               weight_decay=args.weight_decay, eps=1e-3)
    elif args.use_adam:
        cfgopt = cfg.trainer.opt
        vae_optimizer = optim.Adam(vae.parameters(),
                                   lr=args.learning_rate_min_vae,
                                   betas=(cfgopt.beta1, cfgopt.beta2),
                                   weight_decay=cfgopt.weight_decay)

    else:
        vae_optimizer = FusedAdam(vae.parameters(), args.learning_rate_vae,
                                  weight_decay=args.weight_decay, eps=1e-3)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_vae)
    logger.info('[grad_scalar] enabled={}', args.autocast_train)
    if not args.autocast_train:
        grad_scalar = utils.DummyGradScalar()
    else:
        grad_scalar = GradScaler(2**10, enabled=True)

    # create SN calculator
    vae_sn_calculator = SpectralNormCalculator()
    dae_sn_calculator = SpectralNormCalculator()
    if args.train_vae:
        # TODO: require using layer in layers/neural_operations
        vae_sn_calculator.add_bn_layers(vae)
    dae_sn_calculator.add_bn_layers(dae)
    return {
        'vae_scheduler': vae_scheduler,
        'vae_optimizer': vae_optimizer,
        'vae_sn_calculator': vae_sn_calculator,
        'dae_scheduler': dae_scheduler,
        'dae_optimizer': dae_optimizer,
        'dae_sn_calculator': dae_sn_calculator,
        'grad_scalar': grad_scalar
    }


@torch.no_grad()
def validate_inspect(latent_shape,
                     model, dae, diffusion, ode_sample,
                     it, writer,
                     sample_num_points, num_samples,
                     autocast_train=False,
                     need_sample=1, need_val=1, need_train=1,
                     w_prior=None, val_x=None, tr_x=None,
                     val_input=None,
                     prior_cond=None,
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
    if cls_emb is not None:
        kwargs['cls_emb'] = cls_emb
    assert(need_sample >= 0 and need_val > 0 and need_train == 0)
    # draw samples
    if need_sample:
        # gen_x: B,N,3
        gen_x, nstep, ode_time, sample_time, output_dict = \
            fun_generate_samples_vada(latent_shape, dae, diffusion,
                                      model, w_prior.shape[0], enable_autocast=autocast_train,
                                      prior_cond=prior_cond,
                                      ode_sample=ode_sample, ddim_step=ddim_step, clip_feat=clip_feat,
                                      **kwargs)
        logger.info('cast={}, sample step={}, ode_time={}, sample_time={}',
                    autocast_train,
                    nstep if ddim_step == 0 else ddim_step,
                    ode_time, sample_time)
        gen_pcs = gen_x
    else:
        output_dict = {}

    rgb_as_normal = not cfg.data.has_color  # if has color, rgb not as normal
    vis_order = cfg.viz.viz_order
    vis_args = {'rgb_as_normal': rgb_as_normal, 'vis_order': vis_order,
                'is_omap': 'omap' in cfg.data.type}
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

    if vis_latent_point and num_samples > 0:
        img_list = []
        eps_list = []
        prior_cond_list = []
        eps = output_dict['sampled_eps'].view(
            num_samples, dae.num_points, dae.num_classes)[:, :, :cfg.ddpm.input_dim]
        for i in range(num_samples):
            points = gen_x[i]
            points = normalize_point_clouds([points])[0]
            img = visualize_point_clouds_3d([points], ['samples'], **vis_args)
            img_list.append(img)

            points = eps[i]
            points = normalize_point_clouds([points])[0]
            img = visualize_point_clouds_3d([points], ['eps'], **vis_args)
            eps_list.append(img)
            if prior_cond is not None:
                points = prior_cond[i]
                if len(points.shape) > 2:  # points shape is (1,X,Y,Z)
                    output_voxel_XYZ = points[0].cpu().numpy()  # XYZ
                    coordsid = np.where(output_voxel_XYZ == 1)
                    coordsid = np.stack(coordsid, axis=1)  # N,3
                    points = torch.from_numpy(coordsid)
                    voxel_size = 1.0
                    X, Y, Z = output_voxel_XYZ.shape
                    c = torch.tensor([X, Y, Z]).view(1, 3) * 0.5
                    points = points - c  # center at 1
                    vis_points = points
                    bound = max(X, Y, Z)*0.5
                    # logger.info('voxel_size: {}, output_voxel_XYZ: {}, bound: {}',
                    #    voxel_size, output_voxel_XYZ.shape, bound)

                elif vis_args['is_omap']:
                    vis_points = points * s_pcs[i]  # range before norm
                    bound = s_pcs[0].max().item()
                    voxel_size = cfg.data.voxel_size
                else:
                    vis_points = points
                    voxel_size = cfg.data.voxel_size
                    bound = 1.5  # 2.0

                img = visualize_point_clouds_3d([vis_points], ['cond'],
                                                is_voxel=1,
                                                voxel_size=voxel_size,
                                                bound=bound,
                                                **vis_args)

                points = normalize_point_clouds([points])[0]
                ## print('points', points.shape, points.numpy().min(0), points.numpy().max(0), points[:3])
                img2 = visualize_point_clouds_3d([points], ['cond_center'],
                                                 **vis_args)
                img = np.concatenate([img, img2], axis=1)
                prior_cond_list.append(img)

        img = np.concatenate(img_list, axis=2)
        img_eps = np.concatenate(eps_list, axis=2)
        prior_cond_list = np.concatenate(prior_cond_list, axis=2) if len(
            prior_cond_list) else prior_cond_list
        img = np.concatenate([img, img_eps], axis=1)
        img = np.concatenate([img, prior_cond_list], axis=1) if len(
            prior_cond_list) else img
        writer.add_image('sample', torch.as_tensor(img), it)

    inputs = val_input if val_input is not None else val_x
    output = model.recont(inputs) if cls_emb is None else model.recont(
        inputs, cls_emb=cls_emb)
    gen_x = output['final_pred']

    # vis the recont  on val set
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
        if val_input is not None:  # also vis the input, used when we take voxel points as input
            input_list = []
            for i in range(num_recon_val):
                points = val_input[i]
                points = normalize_point_clouds([points])
                input_img = visualize_point_clouds_3d(
                    points, ['input#%d' % i], **vis_args)
                input_list.append(input_img)
            input_list = np.concatenate(input_list, axis=2)
            img = np.concatenate([img, input_list], axis=1)
        writer.add_image('valrecont', torch.as_tensor(img), it)

    # vis recont on train set
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


@torch.no_grad()
def generate_samples_vada_2prior(shape, dae, diffusion, vae, num_samples, enable_autocast,
                                 ode_eps=0.00001, ode_solver_tol=1e-5, ode_sample=False,
                                 prior_var=1.0, temp=1.0, vae_temp=1.0, noise=None,
                                 need_denoise=False, prior_cond=None, device=None, cfg=None,
                                 ddim_step=0, clip_feat=None, cls_emb=None):
    """ this function is copied from trainers/train_2prior.py 
    used by trainers/cond_prior.py 
    should also support trainers/train_2prior.py but not test yet 
    """
    output = {}
    if ode_sample == 1:
        assert isinstance(
            diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
        assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
        assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
        start = timer()
        condition_input = None
        eps_list = []
        for i in range(2):
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

        dae_kwarg = {'is_image': False, 'prior_var': prior_var}
        dae_kwarg['clip_feat'] = clip_feat
        if cfg.data.cond_on_voxel:
            output['prior_cond'] = prior_cond
            voxel_grid_enc_out = dae[2](prior_cond.to(
                device))  # embed the condition_input
            condition_input = voxel_grid_enc_out['global_emb']
        else:
            condition_input = None if cls_emb is None else cls_emb

        all_eps = []
        for i in range(2):
            if i == 1 and cfg.data.cond_on_voxel:
                dae_kwarg['grid_emb'] = voxel_grid_enc_out['grid_emb']
            if ddim_step > 0:
                assert(cls_emb is None), f'not support yet'
                eps, eps_list = diffusion.run_ddim(dae[i],
                                                   num_samples, shape[i], temp, enable_autocast,
                                                   ddim_step=ddim_step,
                                                   condition_input=condition_input,
                                                   skip_type=cfg.sde.ddim_skip_type,
                                                   kappa=cfg.sde.ddim_kappa,
                                                   dae_index=i,
                                                   **dae_kwarg)
            else:
                eps, eps_list = diffusion.run_denoising_diffusion(dae[i],
                                                                  num_samples, shape[i], temp, enable_autocast,
                                                                  condition_input=condition_input,
                                                                  **dae_kwarg
                                                                  )
            condition_input = eps

            if cls_emb is not None:
                condition_input = torch.cat([condition_input,
                                             cls_emb.unsqueeze(-1).unsqueeze(-1)], dim=1)
            if i == 0:
                condition_input = vae.global2style(condition_input)
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
