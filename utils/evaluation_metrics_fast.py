# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
"""
copied and modified from 
    https://github.com/luost26/diffusion-point-cloud/blob/910334a8975aa611423a920869807427a6b60efc/evaluation/evaluation_metrics.py
and 
    https://github.com/stevenygd/PointFlow/tree/b7a9216ffcd2af49b24078156924de025c4dbfb6/metrics
"""
import torch
import time
from tabulate import tabulate
import numpy as np
from loguru import logger
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from utils.exp_helper import ExpTimer
from third_party.PyTorchEMD.emd_nograd import earth_mover_distance_nograd
from third_party.PyTorchEMD.emd import earth_mover_distance
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist_nograd, chamfer_3DDist
from utils.checker import *
import torch.nn.functional as F


def distChamferCUDA_l1(pred, target, points_dim=3):
    import models.pvcnn.functional as pvcnn_fun
    # expect B.2048.3 and B.2048.3
    B = pred.shape[0]
    CHECKDIM(pred, 2, points_dim)
    CHECKDIM(target, 2, points_dim)
    CHECK3D(pred)
    CHECK3D(target)
    target_nndist, pred_nndist, target_nnidx, pred_nnidx \
        = chamfer_3DDist()(target[:, :, :3], pred[:, :, :3])
    target_normal = target.contiguous().permute(0, 2, 1).contiguous()  # BN3->B3N
    pred_normal = pred.contiguous().permute(0, 2, 1).contiguous()   # BN3->B3N

    target_point_normal = pvcnn_fun.grouping(
        target_normal, pred_nnidx[:, :, None])  # B,3,Np,1
    target_point_normal = target_point_normal.squeeze(-1)  # B,3,Np
    cham_norm_y = F.l1_loss(pred_normal.view(-1, points_dim),
                            target_point_normal.view(-1, points_dim),
                            reduction='sum')

    closest_pred_point_normal = pvcnn_fun.grouping(
        pred_normal, target_nnidx[:, :, None]).squeeze(-1)  # B,3,Np,1 -> B,3,Np,
    cham_norm_y2 = F.l1_loss(closest_pred_point_normal.view(-1, points_dim),
                             target_normal.view(-1, points_dim),
                             reduction='sum')

    return cham_norm_y, cham_norm_y2
    ## target_nndist, pred_nndist, cham_norm_y, pred_with_gt_normal
    # return nn_distance(x, y)


#def distChamferCUDA_withnormal(pred, target, normal_loss='cos'):
#    # expect B.2048.3 and B.2048.3
#    import models.pvcnn.functional as pvcnn_fun
#    B = pred.shape[0]
#    CHECKDIM(pred, 2, 6)
#    CHECKDIM(target, 2, 6)
#    CHECK3D(pred)
#    CHECK3D(target)
#
#    target_nndist, pred_nndist, target_nnidx, pred_nnidx \
#        = chamfer_3DDist()(target[:, :, :3], pred[:, :, :3])
#    target_normal = target[:, :, 3:].contiguous().permute(
#        0, 2, 1).contiguous()  # BN3->B3N
#    pred_normal = pred[:, :, 3:].contiguous().permute(
#        0, 2, 1).contiguous()  # BN3->B3N
#    target_point_normal = pvcnn_fun.grouping(
#        target_normal, pred_nnidx[:, :, None])  # B,3,Np,1
#    target_point_normal = target_point_normal.squeeze(-1)  # B,3,Np
#    if normal_loss == 'cos':
#        pred_normal = pred_normal / \
#            (1e-8 + (pred_normal**2).sum(1, keepdim=True).sqrt())
#        cham_norm_y = 1 - torch.abs(
#            F.cosine_similarity(pred_normal, target_point_normal,
#                                dim=1, eps=1e-6))
#    elif normal_loss == 'l2':
#        cham_norm_y = F.mse_loss(pred_normal.view(B, -1), target_point_normal.view(B, -1),
#                                 reduction='none').view(B, 3, -1).mean(1)
#    else:
#        raise NotImplementedError(normal_loss)
#    pred_with_gt_normal = torch.cat([pred[:, :, :3], target_point_normal.permute(0, 2, 1)],
#                                    dim=2).contiguous()
#    CHECKEQ(cham_norm_y.shape[-1], pred_nndist.shape[-1])
#
#    return target_nndist, pred_nndist, cham_norm_y, pred_with_gt_normal
#    # return nn_distance(x, y)


def distChamferCUDA(x, y):
    # expect B.2048.3 and B.2048.3
    B = x.shape[0]
    CHECKDIM(x, 2, 3)
    CHECKDIM(y, 2, 3)
    CHECK3D(x)
    CHECK3D(y)
    # assert (x.shape[-1] == 3
    #        and y.shape[-1] == 3), f'get {x.shape} and {y.shape}'
    dist1, dist2, _, _ = chamfer_3DDist()(x.cuda(), y.cuda())
    return dist1, dist2


def distChamferCUDAnograd(x, y):
    # expect B.2048.3 and B.2048.3
    assert (x.shape[-1] == 3
            and y.shape[-1] == 3), f'get {x.shape} and {y.shape}'
    # return nn_distance_nograd(x, y)
    # dist1, _, dist2, _ = NNDistance(x, y)
    dist1, dist2, _, _ = chamfer_3DDist_nograd()(x.cuda(), y.cuda())
    return dist1, dist2


def emd_approx(sample, ref, require_grad=True):
    #B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
    #assert N == N_ref, f"Not sure what would EMD do in this case; get N={N};N_ref={N_ref}"
    # if not require_grad:
    #    t00 = time.time()
    #    match, _ = ApproxMatch(sample, ref)
    #    print('am: ', time.time() - t00)
    #    emd = MatchCost(sample, ref, match)
    #    del match
    #    # emd = match_cost_nograd(sample, ref)
    # else:
    #    logger.info('error, required require_grad for faster compute ')
    #    exit()
    #    emd = match_cost(sample, ref)  # (B,)
    # emd_norm = emd / float(N)  # (B,)
    # logger.info('emd_norm: {} | sample: {}, ref: {}',
    #    emd_norm.shape, sample.shape, ref.shape)
    if not require_grad:
        emd_pyt = earth_mover_distance_nograd(
            sample.cuda(), ref.cuda(), transpose=False)
    else:
        emd_pyt = earth_mover_distance(
            sample.cuda(), ref.cuda(), transpose=False)

    #logger.info('emd_pyt: {}, diff: {}', emd_pyt.shape, ((emd_pyt - emd_norm)**2).sum())
    return emd_pyt


# def emd_approx(sample, ref, require_grad=True):
#    B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
#    assert N == N_ref, f"Not sure what would EMD do in this case; get N={N};N_ref={N_ref}"
#    if not require_grad:
#        t00 = time.time()
#        match, _ = ApproxMatch(sample, ref)
#        print('am: ', time.time() - t00)
#        emd = MatchCost(sample, ref, match)
#        del match
#        # emd = match_cost_nograd(sample, ref)
#    else:
#        logger.info('error, required require_grad for faster compute ')
#        exit()
#        emd = match_cost(sample, ref)  # (B,)
#    emd_norm = emd / float(N)  # (B,)
#    logger.info('emd_norm: {} | sample: {}, ref: {}',
#        emd_norm.shape, sample.shape, ref.shape)
#    return emd_norm


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs,
           ref_pcs,
           batch_size,
           accelerated_cd=False,
           reduced=True,
           require_grad=False):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd and not require_grad:
            dl, dr = distChamferCUDAnograd(sample_batch, ref_batch)
        elif accelerated_cd:
            dl, dr = distChamferCUDA(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch,
                               require_grad=require_grad)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def formulate_results(results, dataset, hash, step, epoch):
    reported = f'S{step}E{epoch}'
    reported = '' if reported == 'SE' else reported
    msg_head, msg_oneline = '', ''
    if dataset != '-':
        msg_head += "Dataset "
        msg_oneline += f"{dataset} "
    if hash != '-':
        msg_head += "Model "
        msg_oneline += f"{hash} "
    if step != '' or epoch != '':
        msg_head += 'reported '
        msg_oneline += f"{reported} "

    msg_head += "MMD-CDx0.001\u2193 MMD-EMDx0.01\u2193 COV-CD%\u2191 COV-EMD%\u2191 1-NNA-CD%\u2193 1-NNA-EMD%\u2193 JSD\u2193"
    msg_oneline += f"{results.get('lgan_mmd-CD', 0)*1000:.4f} {results.get('lgan_mmd-EMD', 0)*100:.4f} {results.get('lgan_cov-CD', 0)*100:.2f} {results.get('lgan_cov-EMD', 0)*100:.2f} {results.get('1-NN-CD-acc', 0)*100:.2f} {results.get('1-NN-EMD-acc', 0)*100:.2f} {results.get('jsd', 0):.2f}"
    if results.get('url', None) is not None:
        msg_head += " url"
        msg_oneline += f" {results.get('url', '-')}"
    msg_oneline = msg_oneline.split(' ')
    msg_head = msg_head.split(' ')
    return msg_head, msg_oneline


def write_results(out_file, results, dataset='', hash='', step='', epoch=''):
    msg_head, msg_oneline = formulate_results(
        results, dataset, hash, step, epoch)
    content2 = tabulate([msg_oneline], msg_head, tablefmt="tsv")
    text_file = open(out_file, "a")
    text_file.write(content2+'\n')
    text_file.close()
    return content2


def print_results(results, dataset='-', hash='-', step='', epoch=''):
    msg_head, msg_oneline = formulate_results(
        results, dataset, hash, step, epoch)
    msg = '{}'.format(
        tabulate([msg_oneline], msg_head, tablefmt="plain"))
    logger.info('\n{}', msg)
    return msg


def _pairwise_EMD_CD_sub(metric, sample_batch, ref_pcs, N_ref, batch_size, accelerated_cd, verbose, require_grad):
    cd_lst = []
    emd_lst = []
    sub_iterator = range(0, N_ref, batch_size)
    total_iter = int(N_ref / float(batch_size) + 0.5)
    # if verbose:
    #    import tqdm
    #    sub_iterator = tqdm.tqdm(sub_iterator, leave=False)
    #t00 = time.time()
    iter_id = 0
    for ref_b_start in sub_iterator:
        ref_b_end = min(N_ref, ref_b_start + batch_size)
        ref_batch = ref_pcs[ref_b_start:ref_b_end]

        batch_size_ref = ref_batch.size(0)
        point_dim = ref_batch.size(2)
        sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
            batch_size_ref, -1, -1)
        sample_batch_exp = sample_batch_exp.contiguous()
        # print('before cuda {:.5f}s'.format(time.time() - t00))
        # t00 = time.time()
        if metric == 'CD':
            if accelerated_cd and not require_grad:
                dl, dr = distChamferCUDAnograd(sample_batch_exp, ref_batch)
            elif accelerated_cd:
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            else:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)
            # print('cuda: {:.5f}'.format(time.time() - t00))
            #t00 = time.time()
            cd_lst.append(((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
                          )
        elif metric == 'EMD':
            emd_batch = emd_approx(
                sample_batch_exp, ref_batch, require_grad=require_grad)
            emd_lst.append(emd_batch.view(1, -1))
        else:
            raise NotImplementedError
        # torch.cuda.empty_cache()
        # print('approx: {:.5f}'.format(time.time() - t00))
    if metric == 'CD':
        cd_lst = torch.cat(cd_lst, dim=1)
        return cd_lst, cd_lst
    else:
        emd_lst = torch.cat(emd_lst, dim=1)
        return emd_lst, emd_lst
    return cd_lst, emd_lst


def _pairwise_EMD_CD_(metric,
                      sample_pcs,
                      ref_pcs,
                      batch_size,
                      require_grad=True,
                      accelerated_cd=True,
                      verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    # N_sample = 50
    all_cd, all_emd = [], []
    iterator = range(N_sample)
    total_iter = N_sample
    exp_timer = ExpTimer(total_iter)
    iter_id = 0
    print_every = max(int(total_iter // 3), 5)
    for i, sample_b_start in enumerate(iterator):
        exp_timer.tic()
        if iter_id % print_every == 0 and iter_id > 0 and verbose:
            logger.info('done {:02.1f}%({}) eta={:.1f}m',
                        100.0*iter_id/total_iter, total_iter,
                        exp_timer.hours_left()*60)
        sample_batch = sample_pcs[sample_b_start]
        cd_lst, emd_lst = _pairwise_EMD_CD_sub(metric,
                                               sample_batch, ref_pcs, N_ref, batch_size,
                                               accelerated_cd, verbose, require_grad)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)
        exp_timer.toc()
        iter_id += 1

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd
# def _pairwise_EMD_CD_(sample_pcs,
# ref_pcs,
# batch_size,
# require_grad=True,
# accelerated_cd=True,
# verbose=True,
# bs=50):
##    N_sample = sample_pcs.shape[0]
##    N_ref = ref_pcs.shape[0]
# N_sample = 10
# all_cd = torch.zeros(N_sample, N_ref) #[]
# all_emd = torch.zeros(N_sample, N_ref)
# N_sample = 50
##    all_cd, all_emd = [], []
##    iterator = range(0, N_sample, bs)
# if verbose:
##        import tqdm
##        iterator = tqdm.tqdm(iterator)
# for i, sample_b_start in enumerate(iterator):
# if bs == 1:
##            sample_batch = sample_pcs[sample_b_start]
# cd_lst, emd_lst = _pairwise_EMD_CD_sub(
##                sample_batch, ref_pcs, N_ref, batch_size,
# accelerated_cd, verbose, require_grad)
# elif bs > 1:
##            sample_b_end = min(N_sample, sample_b_start + bs)
# cd_lst, emd_lst = _pairwise_EMD_CD_(sample_pcs[sample_b_start:sample_b_end],
##                    ref_pcs, batch_size,
# require_grad=require_grad,
# accelerated_cd=accelerated_cd,
# verbose=verbose,
# bs=1)
##
# all_cd[i:i+1] = cd_lst.cpu()
# all_emd[i:i+1] = emd_lst.cpu()
# all_cd.append(cd_lst)
# all_emd.append(emd_lst)
##
# if (len(all_cd)+1) % 36 == 0:
# torch.cuda.empty_cache()
##
# all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
# all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref
# return all_cd, all_emd
# torch.cuda.empty_cache()
# return all_cd.cpu(), all_emd.cpu()


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    #logger.info('n0={}, n1={}: ', Mxx.shape, Myy.shape)
    #logger.info('Mxx={}, Myy={}, Mxy{}', Mxx, Myy, Mxy)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    #logger.info('label: {}', label.shape)

    M = torch.cat(
        [torch.cat((Mxx, Mxy), 1),
         torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count,
                    (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()
    #logger.info('ored: {}, label: {}', pred, label)
    s = {
        'tp': (pred * label).sum(),  # pred is 1, label is 1
        'fp': (pred * (1 - label)).sum(),  # pred is 1, label is 0
        'fn': ((1 - pred) * label).sum(),  # pred is 0, label is 1
        'tn': ((1 - pred) * (1 - label)).sum(),  # pred is 0, label is 0
    }
    #logger.info( 'label: {} | shape: {} ', label.sum(), label.shape)
    #logger.info('s={}', s)

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size,
                        verbose=True, accelerated_cd=False, metric1='CD',
                        metric2='EMD', **print_kwargs):
    results = {}
    ## metric2 = 'EMD'
    ## metric1 = 'CD'
    if verbose:
        logger.info("Pairwise EMD CD")
    batch_size = ref_pcs.shape[0] // 2 if ref_pcs.shape[0] != batch_size else batch_size
    v1 = False
    v2 = True if verbose else False
    # --- eval CD results --- #
    metric = metric1  # 'CD'
    if verbose:
        logger.info('eval metric: {}; batch-size={}, device: {}, {}',
                    metric, batch_size, ref_pcs.device, sample_pcs.device)
        # batch_size = 100
        # v1 = True
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                          sample_pcs,
                                          batch_size,
                                          accelerated_cd=accelerated_cd,
                                          require_grad=False, verbose=v1)
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                          sample_pcs,
                                          batch_size,
                                          accelerated_cd=accelerated_cd,
                                          require_grad=False, verbose=v1)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({'%s-%s' % (k, metric): v.item()
                   for k, v in res_cd.items()})
    # logger.info('results: {}', results)
    if verbose:
        print_results(results, **print_kwargs)
    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                          ref_pcs,
                                          batch_size,
                                          accelerated_cd=accelerated_cd,
                                          require_grad=False, verbose=v1)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(metric, sample_pcs,
                                          sample_pcs,
                                          batch_size,
                                          accelerated_cd=accelerated_cd,
                                          require_grad=False, verbose=v1)
    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update(
        {"1-NN-%s-%s" % (metric, k): v.item()
         for k, v in one_nn_cd_res.items() if 'acc' in k})
    # logger.info('results: {}', results)
    if verbose:
        print_results(results, **print_kwargs)
    #logger.info('early exit')
    # exit()
    # --- eval EMD results --- #
    metric = metric2  # 'EMD'
    if metric is not None:
        if verbose:
            logger.info('eval metric: {}', metric)
            ## batch_size = min(batch_size, 31)
        M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                              sample_pcs,
                                              batch_size,
                                              accelerated_cd=accelerated_cd,
                                              require_grad=False, verbose=v2)
        M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                              sample_pcs,
                                              batch_size,
                                              accelerated_cd=accelerated_cd,
                                              require_grad=False, verbose=v2)

        res_cd = lgan_mmd_cov(M_rs_cd.t())
        results.update({'%s-%s' % (k, metric): v.item()
                       for k, v in res_cd.items()})
        if verbose:
            print_results(results, **print_kwargs)
        # logger.info('results: {}', results)
        M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                              ref_pcs,
                                              batch_size,
                                              accelerated_cd=accelerated_cd,
                                              require_grad=False, verbose=v2)
        M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(metric, sample_pcs,
                                              sample_pcs,
                                              batch_size,
                                              accelerated_cd=accelerated_cd,
                                              require_grad=False, verbose=v2)
        # 1-NN results
        one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
        results.update(
            {"1-NN-%s-%s" % (metric, k): v.item()
             for k, v in one_nn_cd_res.items() if 'acc' in k})
        if verbose:
            print_results(results, **print_kwargs)
    # logger.info('results: {}', results)

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution,
                                                in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution,
                                             in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds,
                              grid_resolution,
                              in_sphere=False,
                              verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds**2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution,
                                                     in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""
    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))
