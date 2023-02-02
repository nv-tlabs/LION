""" to train hierarchical VAE model with 2 prior 
one for style latent, one for latent pts, 
based on trainers/train_prior.py 
"""
import os
import time
import torchvision 
from PIL import Image 
import functools
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from utils.data_helper import normalize_point_clouds 
from utils.vis_helper import visualize_point_clouds_3d
from utils import model_helper, exp_helper, data_helper 
from utils.diffusion_pvd import DiffusionDiscretized
from utils.diffusion_continuous import make_diffusion, DiffusionBase
from utils.checker import *
from utils import utils
from matplotlib import pyplot as plt 
from timeit import default_timer as timer
from trainers.train_2prior import Trainer as PriorTrainer
def linear_interpolate_noise(noise):
    noise_a = noise[0].contiguous() # 1,D,1,1   
    noise_b = noise[-1].contiguous() # 1,D,1,1   
    num_inter = noise.shape[0] - 2 
    for k in range(1, noise.shape[0]-1):
        p = float(k) / len(noise) # 1/8 to 7/8 
        ## logger.info('p={}; eps: {}', p, noise.shape)
        noise[k] = p * noise_b + (1-p) * noise_a  
    return noise 

def interpolate_noise(noise):
    noise_a = noise[0].contiguous() # 1,D,1,1   
    noise_b = noise[-1].contiguous() # 1,D,1,1   
    num_inter = noise.shape[0] - 2 
    for k in range(1, noise.shape[0]-1):
        p = float(k) / len(noise) # 1/8 to 7/8 
        ## logger.info('p={}; eps: {}', p, noise.shape)
        noise[k] = np.sqrt(p) * noise_b + np.sqrt(1-p) * noise_a  
    return noise 

def subtract_noise(noise):
    noise_a = noise[12].contiguous() # 1,D,1,1
    noise_b = noise[15].contiguous() # 1,D,1,1 
    diff = noise_a - noise_b 
    num_inter = noise.shape[0] - 2  
    add_target_1 = noise[9] 
    add_target_2 = noise[10] 
    noise_list = []
    noise_list.append(noise_a)
    noise_list.append(noise_b)
    noise_list.append(add_target_1)
    noise_list.append(add_target_2)
    noise_list.append(add_target_1 + diff)
    noise_list.append(add_target_2 + diff)
    noise[:6] = torch.stack(noise_list)
    return noise 


VIS_LATENT_PTS = 0 
@torch.no_grad()
def validate_inspect(vis_file, latent_shape,
        model, dae, diffusion, ode_sample,
        it, writer, 
        sample_num_points, num_samples, 
        autocast_train=False,
        need_sample=1, need_val=1, need_train=1,
        w_prior=None, val_x=None, tr_x=None, 
        val_input=None,
        m_pcs=None, s_pcs=None,
        test_loader=None, # can be None 
        has_shapelatent=False, vis_latent_point=False,
        ddim_step=0, epoch=0, fun_generate_samples_vada=None):
    """ visualize the samples, and recont if needed 
    Args:
       has_shapelatent (bool): True when the model has shape latent  
       it (int): step index 
       num_samples: 
       need_* : draw samples for * or not 
    """
    assert(has_shapelatent)
    z_list = []
    num_samples = w_prior.shape[0] if need_sample else 0
    num_recon  = val_x.shape[0] 
    num_recon_val = num_recon if need_val else 0 
    num_recon_train = num_recon if need_train else 0 

    if need_sample:
        # gen_x: B,N,3
        gen_x, nstep, ode_time, sample_time, output_dict = \
                fun_generate_samples_vada(latent_shape, dae, diffusion, model, w_prior.shape[0], 
                enable_autocast=autocast_train,
                ode_sample=ode_sample, ddim_step=ddim_step)
        logger.info('cast={}, sample step={}, ode_time={}, sample_time={}', 
                autocast_train,
                nstep if ddim_step == 0 else ddim_step, 
                ode_time, sample_time)
        gen_pcs = gen_x 
    else:
        output_dict = {}
    # vis the samples 
    if num_samples > 0:
        img_list = []
        for i in range(num_samples): 
            points = gen_x[i] # N,3  
            points = normalize_point_clouds([points])[0] 
            img = visualize_point_clouds_3d([points])
            img_list.append(img)
        img = np.concatenate(img_list, axis=2)
        writer.add_image('sample', torch.as_tensor(img), it)
        img_list = [torch.as_tensor(img) / 255.0 for img in img_list] 
        torchvision.utils.save_image(img_list, vis_file)
        logger.info('save img as: {}', vis_file)

    return output_dict 

@torch.no_grad()
def generate_samples(shape, dae, diffusion, vae, num_samples, enable_autocast, ode_eps=0.00001, ode_solver_tol=1e-5, ## None,
                          ode_sample=False, prior_var=1.0, temp=1.0, vae_temp=1.0, noise=None, need_denoise=False,
                          ddim_step=0, writer=None, generate_mode_global='interpolate', generate_mode_local='freeze'):
    output = {}
    if ode_sample:
        assert isinstance(diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
        assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
        assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
        start = timer()
        condition_input = None 
        eps_list = []
        for i in range(len(dae)):
            noise = torch.randn(size=[num_samples] + shape[i], device='cuda')
            if i == 0: # interpolation 
                generate_mode = generate_mode_global 
            else:
                generate_mode = generate_mode_local 
            logger.info('level: {}, generate_mode: {}', i, generate_mode)
            if generate_mode == 'subtract':
                logger.info('interpolate latent between left most and right most')
                noise = subtract_noise(noise) 
            elif generate_mode == 'interpolate':
                logger.info('interpolate latent between left most and right most')
                noise = interpolate_noise(noise) 
            elif generate_mode == 'linear_interpolate':
                logger.info('linear interpolate latent between left most and right most')
                noise = linear_interpolate_noise(noise) 
            elif generate_mode == 'freeze':
                for k in range(1, noise.shape[0]):
                    noise[k] = noise[0] # for local latent, use the same one for all samples  

            eps, nfe, time_ode_solve = diffusion.sample_model_ode(
                dae[i], num_samples, shape[i], ode_eps, ode_solver_tol, enable_autocast, temp, noise,
                condition_input=condition_input
                )
            condition_input = eps 
            eps_list.append(eps)
            output['sampled_eps'] = eps 
        eps = vae.compose_eps(eps_list) 
    else:
        raise NotImplementedError 
    output['print/sample_mean_global'] = eps.view(num_samples, -1).mean(-1).mean() 
    output['print/sample_var_global'] = eps.view(num_samples, -1).var(-1).mean() 
    decomposed_eps = vae.decompose_eps(eps)
    image = vae.sample(num_samples=num_samples, decomposed_eps=decomposed_eps)
    output['gen_x'] = image 

    end = timer()
    sampling_time = end - start
    nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
    sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
    time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
    return image, nfe_torch, time_ode_solve_torch, sampling_time_torch, output

class Trainer(PriorTrainer):
    is_diffusion = 0 
    generate_mode_global = 'interpolate'
    generate_mode_local = 'interpolate'
    def __init__(self, cfg, args):
        """
        Args:
            cfg: training config 
            args: used for distributed training 
        """
        cfg.num_val_samples = 20
        super().__init__(cfg, args)
    @torch.no_grad() 
    def vis_sample(self, writer, num_vis=None, step=0, include_pred_x0=True,
            save_file=None):
        if self.cfg.ddpm.ema:
            self.swap_vae_param_if_need()
            self.dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        shape = self.model.latent_shape() 
        logger.info('[url]: {}', writer.url)
        logger.info('Latent shape for prior: {}; num_val_samples: {}', shape, self.num_val_samples) 
        ## [self.vae.latent_dim, .num_input_channels, dae.input_size, dae.input_size]
        ode_sample = self.cfg.sde.ode_sample 
        diffusion = self.diffusion_cont if ode_sample else self.diffusion_disc 
        rank = 0 
        seed = 0 
        torch.manual_seed(rank + seed)
        np.random.seed(rank + seed)
        torch.cuda.manual_seed(rank + seed)
        torch.cuda.manual_seed_all(rank + seed)
        for idx in range(40):
            output_dir = os.path.join(self.cfg.save_dir, 'interp', 
                'mode_%s_%s_%d'%(self.generate_mode_global, 
                self.generate_mode_local, self.sample_num_points), 
                '%04d'%idx) 
            vis_dir = os.path.join(self.cfg.save_dir, 'interp', 
                'mode_%s_%s_%d_img'%(self.generate_mode_global, 
                self.generate_mode_local,
                self.sample_num_points))

            logger.info('will save to {}', output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            vis_file = os.path.join(vis_dir, '%04d.png'%idx) 
            output = validate_inspect(vis_file, shape, self.model, self.dae, 
                diffusion, ode_sample,
                step + idx, self.writer, self.sample_num_points, 
                epoch=self.cur_epoch,
                autocast_train=self.cfg.sde.autocast_train,
                need_sample=self.draw_sample_when_vis, 
                need_val=0, need_train=0, 
                num_samples=self.num_val_samples, 
                test_loader=self.test_loader, 
                w_prior=self.w_prior, 
                val_x=self.val_x, tr_x=self.tr_x, 
                val_input=self.val_input,
                m_pcs=self.m_pcs, s_pcs=self.s_pcs,
                has_shapelatent=True,
                vis_latent_point=self.cfg.vis_latent_point,
                ddim_step=self.cfg.viz.vis_sample_ddim_step,
                fun_generate_samples_vada=self.fun_generate_samples_vada
                )
            gen_x = output['gen_x'] 
            logger.info('gen_x shape: {}', gen_x.shape)
            for idxx in range(len(gen_x)):
                torch.save(gen_x[idxx], output_dir + '/%04d.pt'%idxx) 
            logger.info('save to {}', output_dir)

        if writer is not None:
            for n, v in output.items():
                if 'print/' not in n: continue 
                self.writer.add_scalar('%s'%(n.split('print/')[-1]), v, step) 

        if self.cfg.ddpm.ema:
            self.swap_vae_param_if_need()
            self.dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)


    def set_writer(self, writer):
        self.writer = writer
        self.fun_generate_samples_vada = functools.partial(
                generate_samples, ode_eps=self.cfg.sde.ode_eps, 
                writer=self.writer, 
                generate_mode_global=self.generate_mode_global, 
                generate_mode_local=self.generate_mode_local
                )
    def eval_sample(self, step=0):
        logger.info('skip eval-sample')
        return 0
