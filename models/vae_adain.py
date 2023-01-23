# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import numpy as np 
from loguru import logger
import importlib
import torch.nn as nn  
from .distributions import Normal
from utils.model_helper import import_model 
from utils.model_helper import loss_fn
from utils import utils as helper 

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_total_iter = 0
        self.args = args
        self.input_dim = args.ddpm.input_dim 
        latent_dim = args.shapelatent.latent_dim
        self.latent_dim = latent_dim
        self.kl_weight = args.shapelatent.kl_weight
        
        self.num_points = args.data.tr_max_sample_points
        # ---- global ---- #
        # build encoder  
        self.style_encoder = import_model(args.latent_pts.style_encoder)(
            zdim=args.latent_pts.style_dim, 
            input_dim=self.input_dim, 
            args=args)
        if len(args.latent_pts.style_mlp):
            self.style_mlp = import_model(args.latent_pts.style_mlp)(args) 
        else:
            self.style_mlp = None

        self.encoder = import_model(args.shapelatent.encoder_type)(
            zdim=latent_dim, 
            input_dim=self.input_dim, 
            args=args)
        
        # build decoder  
        self.decoder = import_model(args.shapelatent.decoder_type)(
            context_dim=latent_dim, 
            point_dim=args.ddpm.input_dim, 
            args=args)
        logger.info('[Build Model] style_encoder: {}, encoder: {}, decoder: {}',
            args.latent_pts.style_encoder, 
            args.shapelatent.encoder_type,
            args.shapelatent.decoder_type)

    @torch.no_grad()
    def encode(self, x, class_label=None):
        batch_size, _, point_dim = x.size()
        assert(x.shape[2] == self.input_dim), f'expect input in ' \
            f'[B,Npoint,PointDim={self.input_dim}], get: {x.shape}'
        x_0_target = x 
        latent_list = []
        all_eps = [] 
        all_log_q = []
        if self.args.data.cond_on_cat:
            assert(class_label is not None), f'require class label input for cond on cat'
            cls_emb = self.class_embedding(class_label) 
            enc_input = x, cls_emb 
        else:
            enc_input = x 

        # ---- global style encoder ---- #
        z = self.style_encoder(enc_input) 
        z_mu, z_sigma = z['mu_1d'], z['sigma_1d'] # log_sigma
        dist = Normal(mu=z_mu, log_sigma=z_sigma)  # (B, F)
        z_global = dist.sample()[0] 
        all_eps.append(z_global) 
        all_log_q.append(dist.log_p(z_global)) 
        latent_list.append( [z_global, z_mu, z_sigma] )

        # ---- original encoder ---- #
        style = z_global  # torch.cat([z_global, cls_emb], dim=1) if self.args.data.cond_on_cat else z_global 
        style = self.style_mlp(style) if self.style_mlp is not None else style  
        z = self.encoder([x, style])
        z_mu, z_sigma = z['mu_1d'], z['sigma_1d']
        z_sigma = z_sigma - self.args.shapelatent.log_sigma_offset 
        dist = Normal(mu=z_mu, log_sigma=z_sigma)  # (B, F)
        z_local = dist.sample()[0] 
        all_eps.append(z_local) 
        all_log_q.append(dist.log_p(z_local)) 
        latent_list.append( [z_local, z_mu, z_sigma] )
        all_eps = self.compose_eps(all_eps) 
        if self.args.data.cond_on_cat:
            return all_eps, all_log_q, latent_list, cls_emb 
        else:
            return all_eps, all_log_q, latent_list

    def compose_eps(self, all_eps):
        return torch.cat(all_eps, dim=1) #  style: [B,D1], latent pts: [B,ND2]

    def decompose_eps(self, all_eps):
        eps_style = all_eps[:,:self.args.latent_pts.style_dim] 
        eps_local = all_eps[:,self.args.latent_pts.style_dim:]
        return [eps_style, eps_local] 

    def encode_global(self, x, class_label=None):
        
        batch_size, N, point_dim = x.size()
        if self.args.data.cond_on_cat:
            assert(class_label is not None), f'require class label input for cond on cat'
            cls_emb = self.class_embedding(class_label) 
            enc_input = x, cls_emb 
        else:
            enc_input = x 

        z = self.style_encoder(enc_input) 
        z_mu, z_sigma = z['mu_1d'], z['sigma_1d'] # log_sigma
        dist = Normal(mu=z_mu, log_sigma=z_sigma)  # (B, F)
        return dist 

    def global2style(self, style): ##, cls_emb=None):
        Ndim = len(style.shape) 
        if Ndim == 4:
            style = style.squeeze(-1).squeeze(-1) 
        style = self.style_mlp(style) if self.style_mlp is not None else style  
        if Ndim == 4:
            style = style.unsqueeze(-1).unsqueeze(-1) 
        return style 

    def encode_local(self, x, style):
        # ---- original encoder ---- #
        z = self.encoder([x, style])
        z_mu, z_sigma = z['mu_1d'], z['sigma_1d'] # log_sigma
        z_sigma = z_sigma - self.args.shapelatent.log_sigma_offset 
        dist = Normal(mu=z_mu, log_sigma=z_sigma)  # (B, F)
        return dist 

    def recont(self, x, target=None, class_label=None, cls_emb=None):
        batch_size, N, point_dim = x.size()
        assert(x.shape[2] == self.input_dim), f'expect input in ' \
            f'[B,Npoint,PointDim={self.input_dim}], get: {x.shape}'
        x_0_target = x if target is None else target  
        latent_list = []
        all_eps = [] 
        all_log_q = []

        # ---- global style encoder ---- #
        if self.args.data.cond_on_cat: 
            if class_label is not None:
                assert(class_label is not None)
                cls_emb = self.class_embedding(class_label) 
            else:
                assert(cls_emb is not None)

            enc_input = x, cls_emb 
        else:
            enc_input = x 
        z = self.style_encoder(enc_input) 
        z_mu, z_sigma = z['mu_1d'], z['sigma_1d'] # log_sigma
        dist = Normal(mu=z_mu, log_sigma=z_sigma)  # (B, F)
        
        z_global = dist.sample()[0] 
        all_eps.append(z_global) 
        all_log_q.append(dist.log_p(z_global)) 
        latent_list.append( [z_global, z_mu, z_sigma] )

        # ---- original encoder ---- #
        style = torch.cat([z_global, cls_emb], dim=1) if self.args.data.cond_on_cat else z_global 
        style = self.style_mlp(style) if self.style_mlp is not None else style  
        z = self.encoder([x, style])
        z_mu, z_sigma = z['mu_1d'], z['sigma_1d'] # log_sigma
        z_sigma = z_sigma - self.args.shapelatent.log_sigma_offset 
        dist = Normal(mu=z_mu, log_sigma=z_sigma)  # (B, F)
        z_local = dist.sample()[0] 
        all_eps.append(z_local) 
        all_log_q.append(dist.log_p(z_local)) 
        latent_list.append( [z_local, z_mu, z_sigma] )

        # ---- decoder ---- #
        x_0_pred = self.decoder(None, beta=None, context=z_local, style=style) # (B,ncenter,3) 

        make_4d = lambda x: x.unsqueeze(-1).unsqueeze(-1) if len(x.shape) == 2 else x.unsqueeze(-1) 
        all_eps = [make_4d(e) for e in all_eps]
        all_log_q = [make_4d(e) for e in all_log_q]

        output = {  
                'all_eps': all_eps,
                'all_log_q': all_log_q,
                'latent_list': latent_list,
                'x_0_pred':x_0_pred,  
                'x_0_target': x_0_target, 
                'x_t': torch.zeros_like(x_0_target), 
                't': torch.zeros(batch_size), 
                'x_0': x_0_target
                }
        output['hist/global_var'] = latent_list[0][2].exp() 

        if 'LatentPoint' in self.args.shapelatent.decoder_type: 
            latent_shape = [batch_size, -1, self.latent_dim + self.input_dim] 
            if 'Hir' in self.args.shapelatent.decoder_type:
                latent_pts = z_local[:,:-self.args.latent_pts.latent_dim_ext[0]].view(*latent_shape)[:,:,:3].contiguous().clone()
            else:
                latent_pts = z_local.view(*latent_shape)[:,:,:self.input_dim].contiguous().clone()

            output['vis/latent_pts'] = latent_pts.detach().cpu().view(batch_size,
                    -1, self.input_dim) # B,N,3
        output['final_pred'] = output['x_0_pred'] 
        return output 

    def get_loss(self, x, writer=None, it=None, ## weight_loss_1=1, 
            noisy_input=None, class_label=None, **kwargs):
        """
        shapelatent z ~ q(z|x_0) 
        and x_t ~ q(x_t|x_0, t), t ~ Uniform(T)
        forward and get x_{t-1} ~ p(x_{t-1} | x_t, z)
        Args:
            x:  Input point clouds, (B, N, d).
        """
        ## kl_weight = self.kl_weight
        if self.args.trainer.anneal_kl and self.num_total_iter > 0: 
            global_step = it 
            kl_weight = helper.kl_coeff(step=global_step,
                 total_step=self.args.sde.kl_anneal_portion_vada * self.num_total_iter,
                 constant_step=self.args.sde.kl_const_portion_vada * self.num_total_iter,
                 min_kl_coeff=self.args.sde.kl_const_coeff_vada,
                 max_kl_coeff=self.args.sde.kl_max_coeff_vada)
        else:
            kl_weight = self.kl_weight

        batch_size = x.shape[0]
        # CHECKDIM(x, 2, self.input_dim)
        assert(x.shape[2] == self.input_dim)
        
        inputs = noisy_input if noisy_input is not None else x  
        output = self.recont(inputs, target=x, class_label=class_label)
        
        x_0_pred, x_0_target = output['x_0_pred'], output['x_0_target']
        loss_0 = loss_fn(x_0_pred, x_0_target, self.args.ddpm.loss_type, 
                self.input_dim, batch_size).mean()
        rec_loss = loss_0 
        output['print/loss_0'] = loss_0
        output['rec_loss'] = rec_loss 

        # Loss
        ## z_global, z_sigma, z_mu = output['z_global'], output['z_sigma'], output['z_mu']
        kl_term_list = []
        weighted_kl_terms = []
        for pairs_id, pairs in enumerate(output['latent_list']):
            cz, cmu, csigma = pairs 
            log_sigma = csigma
            kl_term_close = (0.5*log_sigma.exp()**2 + 
                    0.5*cmu**2 - log_sigma - 0.5).view(
                    batch_size, -1) 
            if 'LatentPoint' in self.args.shapelatent.decoder_type and 'Hir' not in self.args.shapelatent.decoder_type:
                if pairs_id == 1:
                    latent_shape = [batch_size, -1, self.latent_dim + self.input_dim] 
                    kl_pt = kl_term_close.view(*latent_shape)[:,:,:self.input_dim] 
                    kl_feat = kl_term_close.view(*latent_shape)[:,:,self.input_dim:] 
                    weighted_kl_terms.append(kl_pt.sum(2).sum(1) * self.args.latent_pts.weight_kl_pt) 
                    weighted_kl_terms.append(kl_feat.sum(2).sum(1) * self.args.latent_pts.weight_kl_feat)  

                    output['print/kl_pt%d'%pairs_id] = kl_pt.sum(2).sum(1)
                    output['print/kl_feat%d'%pairs_id] = kl_feat.sum(2).sum(1) 

                    output['print/z_var_pt%d'%pairs_id]  = (log_sigma.view(*latent_shape)[:,:,:self.input_dim]
                            ).exp()**2 
                    output['print/z_var_feat%d'%pairs_id]  = (log_sigma.view(*latent_shape)[:,:,self.input_dim:]
                            ).exp()**2 
                    output['print/z_mean_feat%d'%pairs_id] = cmu.view(*latent_shape)[:,:,self.input_dim:].mean() 
                elif pairs_id == 0:
                    kl_style = kl_term_close  
                    weighted_kl_terms.append(kl_style.sum(-1) * self.args.latent_pts.weight_kl_glb)

                    output['print/kl_glb%d'%pairs_id] = kl_style.sum(-1) 
                    output['print/z_var_glb%d'%pairs_id]  = (log_sigma).exp()**2 

            kl_term_close = kl_term_close.sum(-1)
            kl_term_list.append(kl_term_close) 
            output['print/kl_%d'%pairs_id] = kl_term_close
            output['print/z_mean_%d'%pairs_id] = cmu.mean() 
            output['print/z_mag_%d'%pairs_id]  = cmu.abs().max() 
            # logger.info('log_sigma: {}, mean: {}', log_sigma.shape, (log_sigma.exp()**2).mean())
            output['print/z_var_%d'%pairs_id]  = (log_sigma).exp()**2 
            output['print/z_logsigma_%d'%pairs_id] = log_sigma
            output['print/kl_weight'] = kl_weight 

            
        loss_recons = rec_loss  
        if len(weighted_kl_terms) > 0:
            kl = kl_weight * sum(weighted_kl_terms) 
        else:
            kl = kl_weight * sum(kl_term_list) 
        loss = kl + loss_recons * self.args.weight_recont 
        output['msg/kl'] = kl 
        output['msg/rec'] = loss_recons
        output['loss'] = loss 
        return output 

    def pz(self, w): 
       return w 

    def sample(self, num_samples=10, temp=None, decomposed_eps=[], 
            enable_autocast=False, device_str='cuda', cls_emb=None): 
        """ currently not support the samples of local level 
        Return: 
            model_output: [B,N,D]
        """ 
        batch_size = num_samples 
        center_emd = None 
        if 'LatentPoint' in self.args.shapelatent.decoder_type:
            # Latent Point Model: latent shape; B; ND 
            latent_shape = (num_samples, self.num_points*(self.latent_dim+self.input_dim))
            style_latent_shape = (num_samples, self.args.latent_pts.style_dim) 
        else:
            raise NotImplementedError 

        if len(decomposed_eps) == 0:
            z_local = torch.zeros(*latent_shape).to(
                torch.device(device_str)).normal_()
            z_global = torch.zeros(*style_latent_shape).to(
                torch.device(device_str)).normal_()
        else:
            z_global = decomposed_eps[0] 
            z_local = decomposed_eps[1]

            z_local = z_local.view(*latent_shape) 
            z_global = z_global.view(style_latent_shape)

        style = z_global
        style = self.style_mlp(style) if self.style_mlp is not None else style  
        x_0_pred = self.decoder(None, beta=None, 
                context=z_local, style=z_global) # (B,ncenter,3) 
        ## CHECKSIZE(x_0_pred, (batch_size,self.num_points,[3,6])) 
        return x_0_pred 

    def latent_shape(self):
        return [ 
            [self.args.latent_pts.style_dim, 1, 1],
            [self.num_points*(self.latent_dim+self.input_dim),1,1]
            ]
