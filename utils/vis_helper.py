# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
from datetime import datetime
import torchvision
from utils.checker import *
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image
# Visualization

def plot_points(output, output_name=None):
    from utils.data_helper import normalize_point_clouds
    output = output.cpu()
    input_list = []
    for idx in range(output.shape[0]):
        pts = output[idx]
        pts = normalize_point_clouds([pts])
        input_img = visualize_point_clouds_3d(pts, ['out#%d' % idx])
        input_list.append(input_img)
    input_list = np.concatenate(input_list, axis=2)
    img = Image.fromarray(input_list[:3].astype(np.uint8).transpose((1, 2, 0)))
    if output_name is None:
        output_dir = './results/nv_demos/lion/'
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir, datetime.now().strftime("%y%m%d_%H%M%S.png"))
    img.save(output_name)
    # print(f'INFO save output img as {output_name}')
    return output_name

def visualize_point_clouds_3d_list(pcl_lst, title_lst, vis_order, vis_2D, bound, S):
    t_list = []
    for i in range(len(pcl_lst)):
        img = visualize_point_clouds_3d([pcl_lst[i]], [title_lst[i]] if title_lst is not None else None,
                                        vis_order, vis_2D, bound, S)
        t_list.append(img)
    img = np.concatenate(t_list, axis=2)
    return img


def visualize_point_clouds_3d(pcl_lst, title_lst=None,
                              vis_order=[2, 0, 1], vis_2D=1, bound=1.5, S=3, rgba=0):
    """
    Copied and modified from https://github.com/stevenygd/PointFlow/blob/b7a9216ffcd2af49b24078156924de025c4dbfb6/utils.py#L109 

    Args: 
        pcl_lst: list of tensor, len $L$ = num of point sets, 
            each tensor in shape (N,3), range in [-1,1] 
    Returns: 
        image with $L$ column 
    """
    assert(type(pcl_lst) == list and torch.is_tensor(pcl_lst[0])
           ), f'expect list of tensor, get {type(pcl_lst)} and {type(pcl_lst[0])}'
    if len(pcl_lst) > 1:
        return visualize_point_clouds_3d_list(pcl_lst, title_lst, vis_order, vis_2D, bound, S)

    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    num_col = len(pcl_lst)
    assert(num_col == len(title_lst)
           ), f'require same len, get {num_col} and {len(title_lst)}'
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
        ax1.set_title(title)
        rgb = None
        if type(S) is list:
            psize = S[idx]
        else:
            psize = S
        ax1.scatter(pts[:, vis_order[0]], pts[:, vis_order[1]],
                    pts[:, vis_order[2]], s=psize, c=rgb)
        ax1.set_xlim(-bound, bound)
        ax1.set_ylim(-bound, bound)
        ax1.set_zlim(-bound, bound)
        ax1.grid(False)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = fig2data(fig)
    res = np.transpose(res, (2, 0, 1))  # 3,H,W

    plt.close()

    if vis_2D:
        v1 = 0.5
        v2 = 0
        fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
        num_col = len(pcl_lst)
        assert(num_col == len(title_lst)
               ), f'require same len, get {num_col} and {len(title_lst)}'
        for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
            ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
            rgb = None
            if type(S) is list:
                psize = S[idx]
            else:
                psize = S
            ax1.scatter(pts[:, vis_order[0]], pts[:, vis_order[1]],
                        pts[:, vis_order[2]], s=psize, c=rgb)
            ax1.set_xlim(-bound, bound)
            ax1.set_ylim(-bound, bound)
            ax1.set_zlim(-bound, bound)
            ax1.grid(False)
            ax1.set_title(title + '-2D')
            ax1.view_init(v1, v2)  # 0.5, 0)

        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        # res_2d = np.array(fig.canvas.renderer._renderer)
        res_2d = fig2data(fig)
        res_2d = np.transpose(res_2d, (2, 0, 1))
        plt.close()

        res = np.concatenate([res, res_2d], axis=1)
    return res


def fig2data(fig):
    """
    Adapted from https://stackoverflow.com/questions/55703105/convert-matplotlib-figure-to-numpy-array-of-same-shape 
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    ## fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
