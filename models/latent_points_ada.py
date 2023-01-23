# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch 
from loguru import logger 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from .pvcnn2_ada import \
        create_pointnet2_sa_components, create_pointnet2_fp_modules, LinearAttention, create_mlp_components, SharedMLP 

# the building block of encode and decoder for VAE 

class PVCNN2Unet(nn.Module):
    """
        copied and modified from https://github.com/alexzhou907/PVD/blob/9747265a5f141e5546fd4f862bfa66aa59f1bd33/model/pvcnn_generation.py#L172 
    """
    def __init__(self, 
                 num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, 
                 input_dim=3,
                 width_multiplier=1, 
                 voxel_resolution_multiplier=1,
                 time_emb_scales=1.0,
                 verbose=True, 
                 condition_input=False, 
                 point_as_feat=1, cfg={}, 
                 sa_blocks={}, fp_blocks={}, 
                 clip_forge_enable=0,
                 clip_forge_dim=512
                 ):
        super().__init__()
        logger.info('[Build Unet] extra_feature_channels={}, input_dim={}',
                extra_feature_channels, input_dim)
        self.input_dim = input_dim 

        self.clip_forge_enable = clip_forge_enable 
        self.sa_blocks = sa_blocks 
        self.fp_blocks = fp_blocks
        self.point_as_feat = point_as_feat
        self.condition_input = condition_input
        assert extra_feature_channels >= 0
        self.time_emb_scales = time_emb_scales
        self.embed_dim = embed_dim
        ## assert(self.embed_dim == 0)
        if self.embed_dim > 0: # has time embedding 
            # for prior model, we have time embedding, for VAE model, no time embedding 
            self.embedf = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        if self.clip_forge_enable:
            self.clip_forge_mapping = nn.Linear(clip_forge_dim, embed_dim) 
            style_dim = cfg.latent_pts.style_dim
            self.style_clip = nn.Linear(style_dim + embed_dim, style_dim) 

        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = \
            create_pointnet2_sa_components(
            input_dim=input_dim,
            sa_blocks=self.sa_blocks, 
            extra_feature_channels=extra_feature_channels, 
            with_se=True, 
            embed_dim=embed_dim, # time embedding dim 
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, 
            voxel_resolution_multiplier=voxel_resolution_multiplier, 
            verbose=verbose, cfg=cfg
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else LinearAttention(channels_sa_features, 8, verbose=verbose)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels + input_dim - 3
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, 
            sa_in_channels=sa_in_channels, 
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            verbose=verbose, cfg=cfg 
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(
                in_channels=channels_fp_features, 
                out_channels=[128, dropout, num_classes], # was 0.5
                classifier=True, dim=2, width_multiplier=width_multiplier,
                cfg=cfg)
        self.classifier = nn.ModuleList(layers)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:,0]
        assert(len(timesteps.shape) == 1), f'get shape: {timesteps.shape}'  
        timesteps = timesteps * self.time_emb_scales 

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs, **kwargs):
        # Input: coords: B3N 
        B = inputs.shape[0]
        coords = inputs[:, :self.input_dim, :].contiguous() 
        features = inputs 
        temb = kwargs.get('t', None) 
        if temb is not None:
            t = temb 
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            temb =  self.embedf(self.get_timestep_embedding(t, inputs.device 
                ))[:,:,None].expand(-1,-1,inputs.shape[-1])
            temb_ori = temb  # B,embed_dim,Npoint 
        
        style = kwargs['style'] 
        if self.clip_forge_enable:
            clip_feat = kwargs['clip_feat'] 
            assert(clip_feat is not None), f'require clip_feat as input'
            clip_feat = self.clip_forge_mapping(clip_feat) 
            style = torch.cat([style, clip_feat], dim=1).contiguous()
            style = self.style_clip(style)

        coords_list, in_features_list = [], []
        for i, sa_blocks  in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i > 0 and temb is not None:
                #TODO: implement a sa_blocks forward function; check if is PVConv layer and kwargs get grid_emb, take as additional input 
                features = torch.cat([features,temb],dim=1)
                features, coords, temb, _ = \
                    sa_blocks ((features, 
                    coords, temb, style)) 
            else: # i == 0 or temb is None 
                features, coords, temb, _ = \
                    sa_blocks ((features, coords, temb, style)) 

        in_features_list[0] = inputs[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks  in enumerate(self.fp_layers):
            if temb is not None:
                features, coords, temb, _ = fp_blocks((
                    coords_list[-1-fp_idx], coords, 
                    torch.cat([features,temb],dim=1), 
                    in_features_list[-1-fp_idx], temb, style))
            else:
                features, coords, temb, _ = fp_blocks((
                    coords_list[-1-fp_idx], coords, 
                    features, 
                    in_features_list[-1-fp_idx], temb, style))

        for l in self.classifier:
            if isinstance(l, SharedMLP):
                features = l(features, style)
            else:
                features = l(features)
        return features 

class PointTransPVC(nn.Module):
    # encoder : B,N,3 -> B,N,2*D 
    sa_blocks = [ # conv_configs, sa_configs
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (128, 128, 128))), 
    ]
    fp_blocks = [
        ((128, 128), (128, 3, 8)), # fp_configs, conv_configs
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, zdim, input_dim, args={}):
        super().__init__()
        self.zdim = zdim 
        self.layers = PVCNN2Unet(2*zdim+input_dim*2, 
                embed_dim=0, use_att=1, extra_feature_channels=0,
                input_dim=args.ddpm.input_dim, cfg=args,
                sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks,
                dropout=args.ddpm.dropout)
        self.skip_weight = args.latent_pts.skip_weight
        self.pts_sigma_offset = args.latent_pts.pts_sigma_offset 
        self.input_dim = input_dim

    def forward(self, inputs):
        x, style = inputs 
        B,N,D = x.shape 
        output = self.layers(x.permute(0,2,1).contiguous(), style=style).permute(0,2,1).contiguous() # BND  

        pt_mu_1d = output[:,:,:self.input_dim].contiguous()
        pt_sigma_1d = output[:,:,self.input_dim:2*self.input_dim].contiguous() - self.pts_sigma_offset 
        
        pt_mu_1d = self.skip_weight * pt_mu_1d + x 
        if self.zdim > 0:
            ft_mu_1d = output[:,:,2*self.input_dim:-self.zdim].contiguous()
            ft_sigma_1d = output[:,:,-self.zdim:].contiguous()

            mu_1d = torch.cat([pt_mu_1d, ft_mu_1d], dim=2).view(B,-1).contiguous()
            sigma_1d = torch.cat([pt_sigma_1d, ft_sigma_1d], dim=2).view(B,-1).contiguous() 
        else:
            mu_1d = pt_mu_1d.view(B,-1).contiguous()
            sigma_1d = pt_sigma_1d.view(B,-1).contiguous() 
        return {'mu_1d': mu_1d, 'sigma_1d': sigma_1d}

class LatentPointDecPVC(nn.Module):
    """ input x: [B,Npoint,D] with [B,Npoint,3] 
    """
    sa_blocks = [ # conv_configs, sa_configs
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (128, 128, 128))), 
    ]
    fp_blocks = [
        ((128, 128), (128, 3, 8)), # fp_configs, conv_configs
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, point_dim, context_dim, num_points=None, args={}, **kwargs):
        super().__init__()
        self.point_dim = point_dim  
        logger.info('[Build Dec] point_dim={}, context_dim={}', point_dim, context_dim)
        self.context_dim  = context_dim + self.point_dim 
        # self.num_points = num_points
        if num_points is None:
            self.num_points = args.data.tr_max_sample_points
        else:
            self.num_points = num_points
        self.layers = PVCNN2Unet(point_dim, embed_dim=0, use_att=1, 
                extra_feature_channels=context_dim,
                input_dim=args.ddpm.input_dim, cfg=args, 
                sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks, 
                dropout=args.ddpm.dropout)
        self.skip_weight = args.latent_pts.skip_weight

    def forward(self, x, beta, context, style):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d). [not used] 
            beta:     Time. (B, ). [not used] 
            context:  Latent points, (B,N_pts*D_latent_pts), D_latent_pts = D_input + D_extra
            style: Shape latents. (B,d).
        Returns: 
            points: (B,N,3)
        """ 

        # CHECKDIM(context, 1, self.num_points*self.context_dim)
        assert(context.shape[1] == self.num_points*self.context_dim)
        context = context.view(-1,self.num_points,self.context_dim) # BND 
        x = context[:,:,:self.point_dim]
        output = self.layers(context.permute(0,2,1).contiguous(), style=style).permute(0,2,1).contiguous() # BN3 
        output = output * self.skip_weight + x 
        return output  

