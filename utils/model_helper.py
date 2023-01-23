# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import torch.nn.functional as F
from loguru import logger
import torch
from torch.autograd import grad
import importlib
from utils.evaluation_metrics_fast import distChamferCUDA, emd_approx, distChamferCUDA_l1


def loss_fn(predv, targetv, loss_type, point_dim, batch_size, loss_weight_emd=0.02,
            loss_weight_cdnorm=1,
            return_dict=False):
    B = batch_size
    output = {}

    if loss_type == 'dcd':
        from evaluation.dist_aware_cd import calc_dcd
        res = calc_dcd(predv, targetv)
        loss = res[0]
        output['print/rec_dcd'] = loss

    elif loss_type == 'cd1_sum_emd':  # use l1 loss in chamfer distance, take the sum
        dl, dr = distChamferCUDA_l1(predv, targetv, point_dim)
        loss = dl + dr  # .view(B,-1).sum(-1) + dr.view(B,-1).sum(-1)
        output['print/rec_cd1_sum'] = loss
        emd = emd_approx(predv, targetv)
        emd = emd.view(B, -1)*predv.view(B, -1).shape[1]
        output['print/rec_emd'] = emd
        loss = loss + emd

    elif loss_type == 'cd1_sum':  # use l1 loss in chamfer distance, take the sum
        dl, dr = distChamferCUDA_l1(predv, targetv, point_dim)
        loss = dl + dr  # .view(B,-1).sum(-1) + dr.view(B,-1).sum(-1)
        output['print/rec_cd1_sum'] = loss

    # use l2 loss in chamfer distance, take the sum over N points, but its mean over point dim (3)
    elif loss_type == 'cd_sum':
        dl, dr = distChamferCUDA(predv, targetv)
        loss = dl.view(B, -1).sum(-1) + dr.view(B, -1).sum(-1)
        output['print/rec_cd1_sum'] = loss

    elif loss_type == 'chamfer':
        dl, dr = distChamferCUDA(predv, targetv)
        loss = dl.view(B, -1).mean(-1) + dr.view(B, -1).mean(-1)
        output['print/rec_cd'] = loss

    elif loss_type == 'mse_sum':
        loss = F.mse_loss(
            predv.contiguous().view(-1, point_dim), targetv.view(-1, point_dim),
            reduction='sum')
        output['print/rec_mse'] = loss

    elif loss_type == 'l1_sum':
        loss = F.l1_loss(
            predv.contiguous().view(-1, point_dim), targetv.view(-1, point_dim),
            reduction='sum')
        output['print/rec_l1'] = loss

    elif loss_type == 'l1_cd':
        loss = F.l1_loss(
            predv.contiguous().view(-1, point_dim), targetv.view(-1, point_dim),
            reduction='sum')
        output['print/rec_l1'] = loss
        dl, dr = distChamferCUDA(predv, targetv)
        cd_loss = dl.view(B, -1).sum(-1) + dr.view(B, -1).sum(-1)
        output['print/rec_cd'] = cd_loss
        loss = loss + cd_loss

    elif loss_type == 'mse':
        loss = F.mse_loss(
            predv.contiguous().view(-1, point_dim), targetv.view(-1, point_dim),
            reduction='mean')
        output['print/rec_mse'] = loss

    elif loss_type == 'emd':
        emd = emd_approx(predv, targetv)
        # dl.view(B,-1).mean(-1) + dr.view(B,-1).mean(-1)
        loss = emd.view(B, -1)
        output['print/rec_emd'] = loss

    elif loss_type == 'chamfer_emd':
        dl, dr = distChamferCUDA(predv, targetv)
        cd = dl.view(B, -1).mean(-1) + dr.view(B, -1).mean(-1)
        cd = cd.view(B, -1)
        emd = emd_approx(predv, targetv).view(B, -1)
        loss = cd + emd * loss_weight_emd  # balance the scale of two loss
        output['print/rec_emd'] = emd.mean()
        output['print/rec_weight_emd'] = loss_weight_emd
        output['print/rec_cd'] = cd.mean()

    else:
        raise ValueError(loss_type)
    if return_dict:
        return loss, output
    return loss


def import_model(model_str):
    logger.info('import: {}', model_str)
    p, m = model_str.rsplit('.', 1)
    mod = importlib.import_module(p)
    Model = getattr(mod, m)
    return Model
    ## self.encoder = Model(zdim=latent_dim, input_dim=args.ddpm.input_dim, args=args)


class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def average_gradients(model, rank=-1):
    size = float(dist.get_world_size())
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        param.grad.data /= size
        torch.cuda.synchronize()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())  # if p.requires_grad)


def get_device(model):
    param = next(model.parameters())
    return param.device
