# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" implement the gloabl prior for LION
"""
import torch.nn as nn
from loguru import logger 
import functools
import torch
from ..utils import init_temb_fun, mask_inactive_variables

class SE(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.fc(inputs)

class ResBlockSEClip(nn.Module):
    """
    fixed the conv0 not used error in ResBlockSE
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True) 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim*2, output_dim, 1, 1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 1, 1)
        in_ch = self.output_dim 
        self.SE = SE(in_ch)
    def forward(self, x, t): 
        ## logger.info('x: {}, t: {}, input_dim={}', x.shape, t.shape, self.input_dim)
        clip_feat = t[:, self.input_dim:].contiguous() 
        t = t[:,:self.input_dim].contiguous()
        output = x + t 
        output = torch.cat([output, clip_feat], dim=1).contiguous() 
        output = self.conv1(output)
        output = self.non_linearity(output)
        output = self.conv2(output)
        output = self.non_linearity(output)
        output = self.SE(output) 
        shortcut = x
        return shortcut + output
    def __repr__(self):
        return "ResBlockSEClip(%d, %d)"%(self.input_dim, self.output_dim)



class ResBlockSEDrop(nn.Module):
    """
    fixed the conv0 not used error in ResBlockSE
    """

    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, output_dim, 1, 1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 1, 1)
        in_ch = self.output_dim
        self.SE = SE(in_ch)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ratio = dropout

    def forward(self, x, t):
        output = x + t
        output = self.conv1(output)
        output = self.non_linearity(output)
        output = self.dropout(output)
        output = self.conv2(output)
        output = self.non_linearity(output)
        output = self.SE(output)
        shortcut = x
        return shortcut + output

    def __repr__(self):
        return "ResBlockSE_withdropout(%d, %d, drop=%f)" % (
            self.input_dim, self.output_dim, self.dropout_ratio)


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        # resample=None, act=nn.ELU(),
        #           normalization=nn.InstanceNorm2d, adjust_padding=False, dilation=1):
        super().__init__()
        self.non_linearity = nn.ELU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, output_dim, 1, 1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 1, 1)
        in_ch = self.output_dim
        self.normalize1 = nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                       num_channels=in_ch, eps=1e-6)
        self.normalize2 = nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                       num_channels=in_ch, eps=1e-6)

    def forward(self, x, t):
        x = x + t
        output = self.conv1(x)
        output = self.normalize1(output)
        output = self.non_linearity(output)
        output = self.conv2(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        shortcut = x
        return shortcut + output

    def __repr__(self):
        return "ResBlock(%d, %d)" % (self.input_dim, self.output_dim)


class Prior(nn.Module):
    building_block = ResBlock

    def __init__(self, args, num_input_channels, *oargs, **kwargs):
        super().__init__()
        # args: cfg.sde
        # oargs: other argument: the global argument
        self.condition_input = kwargs.get('condition_input', False)
        self.cfg = oargs[0]
        self.clip_forge_enable = self.cfg.clipforge.enable  # kwargs.get('clipforge.enable', 0)

        logger.info('[Build Resnet Prior] Has condition input: {}; clipforge {}; '
                    'learn_mixing_logit={}, ', self.condition_input,
                    self.clip_forge_enable, args.learn_mixing_logit)

        self.act = act = nn.SiLU()
        self.num_scales = args.num_scales_dae
        self.num_input_channels = num_input_channels

        self.nf = nf = args.num_channels_dae
        num_cell_per_scale_dae = args.num_cell_per_scale_dae if 'num_cell_per_scale_dae' not in kwargs else kwargs[
            'num_cell_per_scale_dae']

        # take clip feature as input
        if self.clip_forge_enable:
            self.clip_feat_mapping = nn.Conv1d(self.cfg.clipforge.feat_dim, self.nf, 1)

        # mixed_prediction #
        self.mixed_prediction = args.mixed_prediction  # This enables mixed prediction
        if self.mixed_prediction:
            logger.info('init-mixing_logit = {}, after sigmoid = {}',
                        args.mixing_logit_init, torch.sigmoid(torch.tensor(args.mixing_logit_init)))
            assert(args.mixing_logit_init), f'require learning'
            # if not args.learn_mixing_logit and args.hypara_mixing_logit:
            #    # not learn, treat it as hyparameters
            #    init = args.mixing_logit_init * torch.ones(size=[1, num_input_channels, 1, 1])
            #    self.mixing_logit = torch.nn.Parameter(init, requires_grad=False) # not update
            #    self.is_active = None
            # elif not args.learn_mixing_logit: # not learn, loaded from c04cd1h exp
            #    init = torch.load('../exp/1110/chair/c04cd1h_hvae3s_390f8dhInitSepesTrainvae0_hvaeB72l1E4W1/mlogit.pt')
            #    self.mixing_logit = torch.nn.Parameter(init, requires_grad=False)
            #    self.is_active = None
            # else:
            if True:
                init = args.mixing_logit_init * torch.ones(size=[1, num_input_channels, 1, 1])
                self.mixing_logit = torch.nn.Parameter(init, requires_grad=True)
                self.is_active = None
        else:  # no mixing_logit
            self.mixing_logit = None
            self.is_active = None

        self.embedding_dim = args.embedding_dim
        self.embedding_dim_mult = 4
        self.temb_fun = init_temb_fun(args.embedding_type, args.embedding_scale, args.embedding_dim)
        logger.info('[temb_fun] embedding_type={}, embedding_scale={}, embedding_dim={}',
                    args.embedding_type, args.embedding_scale, args.embedding_dim)
        # exit()
        modules = []
        modules.append(nn.Conv2d(self.embedding_dim, self.embedding_dim * 4, 1, 1))
        modules.append(nn.Conv2d(self.embedding_dim * 4, nf, 1, 1))
        self.temb_layer = nn.Sequential(*modules)

        modules = []
        input_channels = num_input_channels
        self.input_layer = nn.Conv2d(input_channels, nf, 1, 1)
        in_ch = nf
        for i_block in range(args.num_cell_per_scale_dae):
            modules.append(self.building_block(nf, nf))
        self.output_layer = nn.Conv2d(nf, input_channels, 1, 1)
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, t, **kwargs):
        # timestep/noise_level embedding; only for continuous training
        # time embedding
        if t.dim() == 0:
            t = t.expand(1)
        temb = self.temb_fun(t)[:, :, None, None]  # make it 4d
        temb = self.temb_layer(temb)

        if self.clip_forge_enable:
            clip_feat = kwargs['clip_feat']
            clip_feat = self.clip_feat_mapping(clip_feat[:, :, None])[:, :, :, None]  # B,D -> BD1->B,D,1,1
            if temb.shape[0] == 1 and temb.shape[0] < clip_feat.shape[0]:
                temb = temb.expand(clip_feat.shape[0], -1, -1, -1)
            temb = torch.cat([temb, clip_feat], dim=1)  # add to temb feature
        # mask out inactive variables
        if self.mixed_prediction and self.is_active is not None:
            x = mask_inactive_variables(x, self.is_active)
        x = self.input_layer(x)
        for layer in self.all_modules:
            enc_input = x
            x = layer(enc_input, temb)

        h = self.output_layer(x)
        return h


class PriorSEDrop(Prior):
    def __init__(self, *args, **kwargs):
        self.building_block = functools.partial(ResBlockSEDrop, dropout=args[0].dropout)
        super().__init__(*args, **kwargs)

class PriorSEClip(Prior):
  building_block = ResBlockSEClip 
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

