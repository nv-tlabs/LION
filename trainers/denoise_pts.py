""" to train VAE-encoder with two prior """
import os
import time 
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from loguru import logger
from torch import optim
from utils.ema import EMA
from utils import eval_helper, exp_helper 
from utils.sr_utils import SpectralNormCalculator
from utils.checker import *
from utils.utils import AvgrageMeter 
from utils import utils
from torch.optim import Adam as FusedAdam
from torch.cuda.amp import autocast, GradScaler
from trainers.train_2prior import Trainer as BaseTrainer 
from trainers.base_trainer import init_lns_input 
from utils.data_helper import normalize_point_clouds 
from utils.vis_helper import visualize_point_clouds_3d
from utils.eval_helper import compute_NLL_metric 
CHECKPOINT = int(os.environ.get('CHECKPOINT', 0))
EVAL_LNS = int(os.environ.get('EVAL_LNS', 0))

class Trainer(BaseTrainer):
    is_diffusion = 0 
    def __init__(self, cfg, args):
        """
        Args:
            cfg: training config 
            args: used for distributed training 
        """
        super().__init__(cfg, args)
        self.draw_sample_when_vis = 0 

    @torch.no_grad()
    def eval_sample(self, step=0):
        pass # do nothing 

    # -- shared method for all model with vae component -- # 
    @torch.no_grad() 
    def eval_nll(self, step, ntest=None, save_file=False):
        loss_dict = {} 
        cfg = self.cfg 
        if cfg.ddpm.ema:
            self.swap_vae_param_if_need()
        self.dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        args = self.args 
        device = torch.device('cuda:%d'%args.local_rank)
        tag = exp_helper.get_evalname(self.cfg) + 'D%d'%self.cfg.voxel2pts.diffusion_steps[0] 
        eval_trainnll = self.cfg.eval_trainnll  
        if eval_trainnll:
            data_loader = self.train_loader 
            tag += '-train'
        else:
            data_loader = self.test_loader 
        gen_pcs, ref_pcs = [], []
        shape_id_start = 0
        batch_metric_all = {} 
        gen_x_lns_list = []
        lns_input = init_lns_input(self.is_pointflow_data)
        tag_cur = self.cfg.voxel2pts.diffusion_steps[0]
        if 'chair'  in self.cfg.data.cates:
            input_pts = '../exp/0404/nschair/60eeeah_train_l2e-4E4k_vae_adainB5l1E3W8/eval/samples_1415999s1Ha1104diet.pt'
            output_dir_template = '../exp/0404/nschair/60eeeah_train_l2e-4E4k_vae_adainB5l1E3W8/eval/samplesmm_unorm_D%d/%04d/'  
            index_select = [137] ##, 421, 690] 
        elif 'car' in self.cfg.data.cates:
            input_pts = '../exp/0417/nscar/94d09ch_train_l2e-4_vae_adainB20l1E3W8/eval/samples_511999s1Hd2edfdiet.pt'
            output_dir_template = '../exp/0417/nscar/94d09ch_train_l2e-4_vae_adainB20l1E3W8/eval/samplesmm_unorm_D%d/%04d/' 
            index_select = [534, 579, 147]
        elif 'airplane' in self.cfg.data.cates:
            input_pts = '../exp/0428/nsairplane/8c3930h_train_l2e-4_vae_adainB20l1E3W8/eval/samples_305999s1Hd2edfdiet.pt' 
            output_dir_template = '../exp/0428/nsairplane/8c3930h_train_l2e-4_vae_adainB20l1E3W8/eval/sampesmm_unorm_D%d/%04d/'
            index_select = [72, 59, 47]
        else:
            raise ValueError 

        loaded_pts = torch.load(input_pts).cuda() 
        logger.info('loaded_pts: {}', loaded_pts.shape)        
        for vid, val_batch in enumerate(data_loader):

            output_dir = output_dir_template%(tag_cur, vid) 
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) 
            if vid % 30 == 1:
                logger.info('eval: {}/{}, BS={}', vid, len(data_loader), B)
            if vid == 3:
                break 
            m, s = val_batch['mean'], val_batch['std'] 
            B = m.shape[0]
            file_name = ['%04d'%i for i in range(B)] 
            vid_select = index_select[vid]
            val_x = loaded_pts[vid_select:vid_select+1].expand(B,-1,-1) 
            logger.info('val_x: {}', val_x.shape)

            B,N,C = val_x.shape
            inputs = val_x 
            log_info = {'x_0': val_x, 'vis/input_noise': inputs.view(B,N,-1), 'x_t': val_x}

            # -- global latent -- # 
            dae_index = 0
            dist = self.model.encode_global(inputs)

            B = inputs.shape[0]
            shape = self.model.latent_shape()[dae_index]  
            noise_shape = [B] + shape  
            eps = dist.sample()[0] 
            eps_shape = eps.shape 
            eps = eps.view(noise_shape)  
            eps_ori = eps 
            for time_start in self.cfg.voxel2pts.diffusion_steps:  
                diffusion = self.diffusion_disc # not support ode now 
                diffusion_in_latent = time_start > 0
                if diffusion_in_latent and time_start > 0:
                    noise_q = torch.randn(size=eps.size(), device='cuda')
                    t, var_t_q, m_t_q, _,_,_ = diffusion.iw_quantities_t(B, (torch.zeros(B) + time_start).cuda().long())
                    eps_t_q = diffusion.sample_q(eps, noise_q, var_t_q, m_t_q) 
                    eps, _ = diffusion.run_denoising_diffusion_from_t(self.dae[dae_index], B, shape, 
                            time_start, x_noisy=eps_t_q, is_image=False,
                            enable_autocast=False, prior_var=1.0)
            eps[0:1] = eps_ori[0:1] 
            eps_global = eps.contiguous() 

            condition_input=eps_global

            # style 
            style = self.model.global2style(eps_global.view(eps_shape))
            dist_local = self.model.encode_local(inputs, style) 

            # -- local latent -- # 
            dae_index = 1
            shape = self.model.latent_shape()[dae_index]  
            noise_shape = [B] + shape  
            eps = dist_local.sample()[0] 
            eps_shape = eps.shape 
            log_info['vis/eps_ori'] = eps.cpu().view(B,N,-1)[:,:,:3]
            logger.info('eps: {} or {} ', eps.shape, eps.view(B,N,-1).shape)
            eps = eps.view(noise_shape)  
            eps_ori = eps 
            for time_start in self.cfg.voxel2pts.diffusion_steps:  
                diffusion = self.diffusion_disc # not support ode now 
                diffusion_in_latent = time_start > 0
                if diffusion_in_latent and time_start > 0:
                    noise_q = torch.randn(size=eps.size(), device='cuda')
                    t, var_t_q, m_t_q, _,_,_ = diffusion.iw_quantities_t(B, (torch.zeros(B) + time_start).cuda().long())
                    eps_t_q = diffusion.sample_q(eps, noise_q, var_t_q, m_t_q) 
                    log_info['vis/eps_t%d'%time_start] = eps_t_q.view(B,N,-1).cpu()[:,:,:3]
                    eps, _ = diffusion.run_denoising_diffusion_from_t(self.dae[dae_index], B, shape, 
                            time_start, x_noisy=eps_t_q, is_image=False,
                            enable_autocast=False, prior_var=1.0, 
                            condition_input=condition_input)
                    log_info['vis/eps_new'] = eps.view(B,N,-1).cpu()[:,:,:3]
            eps[0:1] = eps_ori[0:1].contiguous()
            eps_local = eps.view(eps_shape) 
            gen_x = self.model.decoder(None, beta=None, context=eps_local, style=style) # (B,ncenter,3) 
            ## gen_x = val_x 

            log_info['x_0_pred'] = gen_x.detach().cpu()
            if vid == 0:
                self.vis_recont(log_info, self.writer, step, normalize_pts=True)
            # forward to get output 
            val_x = val_x.contiguous() 
            inputs = inputs.contiguous()
            output = self.model.get_loss(val_x, it=step, is_eval_nll=1, noisy_input=inputs)
            gen_x = gen_x.cpu() * s + m 
            for i, file_name_i in  enumerate(file_name):
                #logger.info('diff before and after: {}, before: {}, {}, after: {}, {}',
                #        ((val_x[i] - gen_x[i])**2).sum(), 
                #        val_x[i].min(), val_x[i].max(),
                #        gen_x[i].min(), gen_x[i].max()) 
                torch.save(gen_x[i], os.path.join(output_dir, file_name_i)) 
            logger.info('save output at : {}', output_dir)
        return 0  
