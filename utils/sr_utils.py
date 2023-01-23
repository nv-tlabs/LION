# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" copied and modified from https://github.com/NVlabs/LSGM/blob/5eae2f385c014f2250c3130152b6be711f6a3a5a/util/sr_utils.py """
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@torch.jit.script
def fused_abs_max_add(weight: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    loss += torch.max(torch.abs(weight))
    return loss


class SpectralNormCalculator:
    def __init__(self, num_power_iter=4, custom_conv=False):
        self.num_power_iter = num_power_iter
        # increase the number of iterations for the first time
        self.num_power_iter_init = 10 * num_power_iter
        self.all_conv_layers = []
        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.all_bn_layers = []
        self.custom_conv = custom_conv

    def add_conv_layers(self, model):
        for n, layer in model.named_modules():
            if self.custom_conv:
                # add our customized conv layers
                if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                    self.all_conv_layers.append(layer)
            else:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv3d) or \
                        isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):   # add pytorch conv layers
                    self.all_conv_layers.append(layer)

    def add_bn_layers(self, model):
        for n, layer in model.named_modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or \
                    isinstance(layer, nn.GroupNorm):
                self.all_bn_layers.append(layer)

    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized if self.custom_conv else l.weight
            if not isinstance(l, nn.Linear):
                weight_mat = weight.view(weight.size(0), -1)
            else:
                weight_mat = weight
            ## logger.info('mat weight: {} | weight: {}', weight_mat.shape, weight.shape)

            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(
                        num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(
                        num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    num_iter = self.num_power_iter_init

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(
                1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

    def batchnorm_loss(self):
        loss = torch.zeros(size=()).cuda()
        for l in self.all_bn_layers:
            if l.affine:
                loss = fused_abs_max_add(l.weight, loss)

        return loss

    def state_dict(self):
        return {
            'sr_v': self.sr_v,
            'sr_u': self.sr_u
        }

    def load_state_dict(self, state_dict, device):
        # map the tensor to the device id of self.sr_v
        for s in state_dict['sr_v']:
            self.sr_v[s] = state_dict['sr_v'][s].to(device)

        for s in state_dict['sr_u']:
            self.sr_u[s] = state_dict['sr_u'][s].to(device)
