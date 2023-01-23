# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from loguru import logger
from utils.vis_helper import visualize_point_clouds_3d
from utils.data_helper import normalize_point_clouds
from utils.checker import *

@torch.no_grad()
def validate_inspect_noprior(model,
                             it, writer,
                             sample_num_points, num_samples,
                             need_sample=1, need_val=1, need_train=1,
                             w_prior=None, val_x=None, tr_x=None,
                             test_loader=None,  # can be None
                             has_shapelatent=False,
                             bound=1.5, val_class_label=None, tr_class_label=None,
                             cfg={}):
    """ visualize the samples, and recont if needed 
    """
    assert(has_shapelatent)
    assert(w_prior is not None and val_x is not None and tr_x is not None)
    z_list = []
    num_samples = w_prior.shape[0] if need_sample else 0
    num_recon = val_x.shape[0]
    num_recon_val = num_recon if need_val else 0
    num_recon_train = num_recon if need_train else 0

    assert(need_sample == 0 and need_val > 0 and need_train == 0)
    if need_sample:
        z_prior = model.pz(w_prior, sample_num_points)
        z_list.append(z_prior)
    if val_class_label is not None:
        output = model.recont(val_x, class_label=val_class_label)
    else:
        output = model.recont(val_x)  # torch.cat([val_x, tr_x]))
    gen_x = output['final_pred']
    vis_order = cfg.viz.viz_order
    vis_args = {'vis_order': vis_order}

    # vis the samples
    if num_samples > 0:
        img_list = []
        for i in range(num_samples):
            points = gen_x[i]
            points = normalize_point_clouds([points])[0]
            img = visualize_point_clouds_3d([points], bound=bound, **vis_args)
            img_list.append(img)
        img = np.concatenate(img_list, axis=2)
        writer.add_image('sample', torch.as_tensor(img), it)

    # vis the recont
    if num_recon_val > 0:
        img_list = []
        for i in range(num_recon_val):
            points = gen_x[num_samples + i]
            points = normalize_point_clouds([points])  # val_x[i], points])
            img = visualize_point_clouds_3d(
                points, ['rec#%d' % i], bound=bound, **vis_args)
            img_list.append(img)
        gt_list = []
        for i in range(num_recon_val):
            points = normalize_point_clouds([val_x[i]])
            img = visualize_point_clouds_3d(
                points, ['gt#%d' % i], bound=bound, **vis_args)
            gt_list.append(img)
        img = np.concatenate(img_list, axis=2)
        gt = np.concatenate(gt_list, axis=2)
        img = np.concatenate([gt, img], axis=1)

        if 'vis/latent_pts' in output:
            latent_pts = output['vis/latent_pts']
            img_list = []
            for i in range(num_recon_val):
                points = latent_pts[num_samples + i]
                points = normalize_point_clouds([points])
                latent = visualize_point_clouds_3d(
                    points, ['latent#%d' % i], bound=bound, **vis_args)
                img_list.append(latent)
            latent_list = np.concatenate(img_list, axis=2)
            img = np.concatenate([img, latent_list], axis=1)

        writer.add_image('valrecont', torch.as_tensor(img), it)

    if num_recon_train > 0:
        img_list = []
        for i in range(num_recon_train):
            points = gen_x[num_samples + num_recon_val + i]
            points = normalize_point_clouds([tr_x[i], points])
            img = visualize_point_clouds_3d(
                points, ['ori', 'rec'], bound=bound, **vis_args)
            img_list.append(img)
        img = np.concatenate(img_list, axis=2)
        writer.add_image('train/recont', torch.as_tensor(img), it)

    logger.info('writer: {}', writer.url)

