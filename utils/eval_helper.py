# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
import json
from comet_ml import Experiment, OfflineExperiment
## import open3d as o3d
import time
import numpy as np
import torch
from loguru import logger
import torchvision
from PIL import Image
from utils.vis_helper import visualize_point_clouds_3d
from utils.data_helper import normalize_point_clouds
from utils.checker import *
import torchvision
import sys
import math
from utils.evaluation_metrics_fast import compute_all_metrics, \
    jsd_between_point_cloud_sets, print_results, write_results
from utils.evaluation_metrics_fast import EMD_CD
CD_ONLY = int(os.environ.get('CD_ONLY', 0))
VIS = 1

def pair_vis(gen_x, tr_x, titles, subtitles, writer, step=-1):
    img_list = []
    num_recon = len(gen_x)
    for i in range(num_recon):
        points = gen_x[i]
        points = normalize_point_clouds([tr_x[i], points])
        img = visualize_point_clouds_3d(points, subtitles[i])
        img_list.append(torch.as_tensor(img) / 255.0)
    grid = torchvision.utils.make_grid(img_list, nrow=num_recon//2)
    if writer is not None:
        writer.add_image(titles, grid, step)

def compute_NLL_metric(gen_pcs, ref_pcs, device, writer=None, output_name='', batch_size=200, step=-1, tag=''):
    # evaluate the reconstrution results
    metrics = EMD_CD(gen_pcs.to(device), ref_pcs.to(device),
                     batch_size=batch_size, accelerated_cd=True, reduced=False)
    titles = 'nll/first-10-%s' % tag
    k1, k2 = list(metrics.keys())
    subtitles = [['ori', 'gen-%s=%.1fx1e-2;%s=%.1fx1e-2' %
                  (k1, metrics[k1][j]*1e2, k2, metrics[k2][j]*1e2)] for j in range(10)]
    pair_vis(gen_pcs[:10], ref_pcs[:10], titles, subtitles, writer, step=step)
    results = {}

    for k in metrics.keys():
        sorted, indices = torch.sort(metrics[k])
        worse_ten, worse_score = indices[-10:].cpu(), sorted[-10:].cpu()
        titles = 'nll/worst-%s-%s' % (k, tag)
        subtitles = [['ori', 'gen-%s=%.2fx1e-2' %
                      (k, worse_score[j]*1e2)] for j in range(len(worse_score))]
        pair_vis(gen_pcs[worse_ten], ref_pcs[worse_ten],
                 titles, subtitles, writer, step=step)
        if 'score_detail' not in results:
            results['score_detail'] = metrics[k]
        metrics[k] = metrics[k].mean()

    logger.info('best 10: {}', indices[:10])
    results.update({k: v.item() for k, v in metrics.items()})
    output = ''
    for k, v in results.items():
        if 'detail' in k:
            continue
        output += '%s=%.3fx1e-2 ' % (k, v*1e2)
        logger.info('{}: {}', k, v)
        if 'CD' in k:
            score = v

    url = writer.url if writer is not None else ''
    logger.info('\n' + '-'*60 +
                f'\n{output_name} | \n{output} step={step} \n {url} \n ' + '-'*60)
    return results


def get_ref_num(cats, luo_split=False):
    #ref = './scripts/test_data/ref_%s.pt'%cats
    #assert(os.path.exists(ref)), f'file not found: {ref}'
    num_test = {
        'animal': 100,
        'airplane': 405,
        'airplane_ps': 405,
        'chair': 662,
        'chair_ps': 662,
        'car': 352,
        'car_ps': 352,
        'all': 1000,
        'mug': 22,
        'bottle': 43
    }
    if luo_split:
        num_test = {
            'airplane': 607,
            'chair': 989,
            'car': 528
        }

    assert(cats in num_test), f'not found: {cats} in {num_test}'
    return num_test[cats]


def get_cats(cats):
    # return the category name for this dataset
    all_cats = ['airplane', 'chair', 'car', 'all', 'animal', 'mug', 'bottle']
    for c in all_cats:
        if c in cats or c == cats:
            cats = c
            break
    assert(cats in all_cats), f'not foud cats for {cats} in {all_cats}'
    return cats


def get_ref_pt(cats, data_type="datasets.pointflow_datasets", luo_split=False):
    cats = get_cats(cats)
    root = './datasets/test_data/'
    if 'pointflow' in data_type:
        ref = 'ref_val_%s.pt' % cats
    elif 'neuralspline_datasets' in data_type:
        ref = 'ref_ns_val_%s.pt' % cats
    else:
        logger.info('get_ref_pt not support data_type: %s' % data_type)
        return None

    ref = os.path.join(root, ref)
    assert(os.path.exists(ref)), f'file not found: {ref}'
    return ref


#@torch.no_grad()
#def compute_score_fast(gen_pcs, ref_pcs, m_pcs, s_pcs,
#                       batch_size_test=256, device_str='cuda', cd_only=1,
#                       exp=None, verbose=False,
#                       device=None, accelerated_cd=True, writer=None, norm_box=False, **print_kwargs):
#    """ used to eval the pcs during training; all the files will not be dumpped into disk (to save time) 
#    the ref_pcs will be part of the full dataset only 
#    Args: 
#        output_name (str) path to sample obj: tensor: (Nsample.Npoint.3or6)
#        ref_name (str) path to torch obj: 
#            torch.save({'ref': ref_pcs, 'mean': m_pcs, 'std': s_pcs}, ref_name)
#        print_kwargs (dict): entries: dataset, hash, step, epoch; 
#    """
#    if gen_pcs.shape[1] > ref_pcs.shape[1]:
#        xperm = np.random.permutation(np.arange(gen_pcs.shape[1]))[
#            :ref_pcs.shape[1]]
#        gen_pcs = gen_pcs[:, xperm]
#    if ref_pcs.shape[0] > gen_pcs.shape[0]:
#        ref_pcs = ref_pcs[:gen_pcs.shape[0]]
#        m_pcs = m_pcs[:gen_pcs.shape[0]]
#        s_pcs = s_pcs[:gen_pcs.shape[0]]
#    elif ref_pcs.shape[0] < gen_pcs.shape[0]:
#        gen_pcs = gen_pcs[:ref_pcs.shape[0]]
#
#    device = torch.device(device_str) if device is None else device
#    CHECKEQ(ref_pcs.shape[0], gen_pcs.shape[0])
#    N_ref = ref_pcs.shape[0]  # subset it
#    batch_size_test = N_ref  # * 0.5
#    if gen_pcs.shape[2] == 6:  # B,N,3 or 6
#        gen_pcs = gen_pcs[:, :, :3]
#        ref_pcs = ref_pcs[:, :, :3]
#    if norm_box:
#        ref_pcs = 0.5 * torch.stack(normalize_point_clouds(ref_pcs), dim=0)
#        gen_pcs = 0.5 * torch.stack(normalize_point_clouds(gen_pcs), dim=0)
#        print_kwargs['dataset'] = print_kwargs.get('dataset',
#                                                   '')+'-normbox'
#
#        #ref_pcs = normalize_point_clouds(ref_pcs)
#        #gen_pcs = normalize_point_clouds(gen_pcs)
#        # print_kwargs['dataset'] = print_kwargs.get('dataset',
#        #    '')+'-normbox'
#        # logger.info('[data shape] ref_pcs: {}, gen_pcs: {}, mean={}, std={}; norm_box={}',
#        # ref_pcs.shape, gen_pcs.shape, m_pcs.shape, s_pcs.shape, norm_box)
#    elif m_pcs is not None and s_pcs is not None:
#        ref_pcs = ref_pcs * s_pcs + m_pcs
#        gen_pcs = gen_pcs * s_pcs + m_pcs
#    # visualize first few samples:
#    if VIS and writer is not None and writer.exp is not None or exp is not None:
#        logger.info('vis the result')
#        if exp is None:
#            exp = writer.exp
#        img_list = []
#        for i in range(min(20, ref_pcs.shape[0])):
#            NORM_VIS = 0
#            if NORM_VIS:
#                norm_ref, norm_gen = normalize_point_clouds([
#                    ref_pcs[i], gen_pcs[i]])
#            else:
#                norm_ref = ref_pcs[i]
#                norm_gen = gen_pcs[i]
#            img = visualize_point_clouds_3d([norm_ref, norm_gen],
#                                            [f'ref-{i}', f'gen-{i}'], bound=0.5)
#            img_list.append(torch.as_tensor(img) / 255.0)
#        grid = torchvision.utils.make_grid(img_list)
#        # to 3,H,W to H,W,3
#        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
#            1, 2, 0).to('cpu', torch.uint8).numpy()
#        exp.log_image(ndarr, 'samples/verse_%s' %
#                      print_kwargs.get('hash', '_'), step=print_kwargs.get('step', 0))
#        # epoch=print_kwargs.get('epoch', 0))
#
#    metric2 = 'EMD' if not cd_only else None
#    results = compute_all_metrics(gen_pcs.to(device).float(),
#                                  ref_pcs.to(device).float(), batch_size_test,
#                                  accelerated_cd=accelerated_cd, metric2=metric2,
#                                  verbose=verbose,
#                                  **print_kwargs)
#    print_results(results, **print_kwargs)
#
#    return results


@torch.no_grad()
def compute_score(output_name, ref_name, batch_size_test=256, device_str='cuda',
                  device=None, accelerated_cd=True, writer=None,
                  exp=None,
                  norm_box=False, skip_write=False, **print_kwargs):
    """
    Args: 
        output_name (str) path to sample obj: tensor: (Nsample.Npoint.3or6)
        ref_name (str) path to torch obj: 
            torch.save({'ref': ref_pcs, 'mean': m_pcs, 'std': s_pcs}, ref_name)
        print_kwargs (dict): entries: dataset, hash, step, epoch; 
    """
    logger.info('[compute sample metric] sample: {} and ref: {}',
                output_name, ref_name)
    ref = torch.load(ref_name)
    ref_pcs = ref['ref'][:, :, :3]
    m_pcs, s_pcs = ref['mean'], ref['std']
    gen_pcs = torch.load(output_name)
    if gen_pcs.shape[1] > ref_pcs.shape[1]:
        xperm = np.random.permutation(np.arange(gen_pcs.shape[1]))[
            :ref_pcs.shape[1]]
        gen_pcs = gen_pcs[:, xperm]
    if type(gen_pcs) is dict:
        logger.info('WARNING: the gen_pcs is a dict, with key '
                    'as {}| usuaglly its a tensor '
                    'you perhaps takes the train data,',
                    gen_pcs.keys())
        gen_pcs = gen_pcs['ref']
    device = torch.device(device_str) if device is None else device
    # batch_size_test = ref_pcs.shape[0]
    logger.info('[data shape] ref_pcs: {}, gen_pcs: {}, mean={}, std={}; norm_box={}',
                ref_pcs.shape, gen_pcs.shape, m_pcs.shape, s_pcs.shape, norm_box)
    N_ref = ref_pcs.shape[0]  # subset it
    m_pcs = m_pcs[:N_ref]
    s_pcs = s_pcs[:N_ref]
    ref_pcs = ref_pcs[:N_ref]
    gen_pcs = gen_pcs[:N_ref]
    if gen_pcs.shape[2] == 6:  # B,N,3 or 6
        gen_pcs = gen_pcs[:, :, :3]
        ref_pcs = ref_pcs[:, :, :3]
    if norm_box:
        #ref_pcs = ref_pcs * s_pcs + m_pcs
        #gen_pcs = gen_pcs * s_pcs + m_pcs
        ref_pcs = 0.5 * torch.stack(normalize_point_clouds(ref_pcs), dim=0)
        gen_pcs = 0.5 * torch.stack(normalize_point_clouds(gen_pcs), dim=0)
        print_kwargs['dataset'] = print_kwargs.get('dataset',
                                                   '')+'-normbox'
    else:
        ref_pcs = ref_pcs * s_pcs + m_pcs
        gen_pcs = gen_pcs * s_pcs + m_pcs
    # visualize first few samples:
    if VIS:
        if exp is not None:
            exp = exp
        elif writer is not None:
            exp = writer.exp
        elif os.path.exists('.comet_api'):
            comet_args = json.load(open('.comet_api', 'r'))
            exp = Experiment(display_summary_level=0,
                            **comet_args)
        else:
            exp = OfflineExperiment(offline_directory="/tmp")
        img_list = []
        gen_list = []
        ref_list = []
        for i in range(20):
            NORM_VIS = 0
            if NORM_VIS:
                norm_ref, norm_gen = normalize_point_clouds([
                    ref_pcs[i], gen_pcs[i]])
            else:
                norm_ref = ref_pcs[i]
                norm_gen = gen_pcs[i]
            ref_img = visualize_point_clouds_3d([norm_ref],
                                                [f'ref-{i}'], bound=1.0)  # 0.8)
            gen_img = visualize_point_clouds_3d([norm_gen],
                                                [f'gen-{i}'], bound=1.0)  # 0.8)
            ref_list.append(torch.as_tensor(ref_img) / 255.0)
            gen_list.append(torch.as_tensor(gen_img) / 255.0)
            img_list.append(ref_list[-1])
            img_list.append(gen_list[-1])

        path = output_name.replace('.pt', '_eval.png')

        grid = torchvision.utils.make_grid(gen_list)
        # to 3,H,W to H,W,3
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        exp.log_image(ndarr, 'samples')

        ref_grid = torchvision.utils.make_grid(ref_list)
        # to 3,H,W to H,W,3
        ref_ndarr = ref_grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr = np.concatenate([ndarr, ref_ndarr], axis=0)
        exp.log_image(ndarr, 'samples_vs_ref')

        torchvision.utils.save_image(img_list, path)
        logger.info(exp.url)
        logger.info('save vis at {}', path)
    metric2 = 'EMD' if not CD_ONLY else None
    logger.info('print_kwargs: {}', print_kwargs)
    results = compute_all_metrics(gen_pcs.to(device).float(),
                                  ref_pcs.to(device).float(), batch_size_test, accelerated_cd=accelerated_cd, metric2=metric2,
                                  **print_kwargs)

    jsd = jsd_between_point_cloud_sets(
        gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd
    msg = print_results(results, **print_kwargs)
    # with open('../exp/eval_out.txt', 'a') as f:
    #     run_time = time.strftime('%m%d-%H%M-%S')
    #     f.write('<< date: %s >>\n' % run_time)
    #     f.write('%s\n%s\n' % (exp.url, msg))
    results['url'] = exp.url
    if not skip_write:
        os.makedirs('results', exist_ok=True)
        msg = write_results(
            os.path.join('./results/', 'eval_out.csv'),
            results, **print_kwargs)
    if metric2 is None:
        logger.info('early exit')
        exit()
    return results

