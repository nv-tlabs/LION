import numpy as np
import torch
from torch.autograd import Function

# from modules.functional.backend import _backend
from third_party.pvcnn.functional.backend import _backend

__all__ = ['gather', 'furthest_point_sample', 'logits_mask']


class Gather(Function):
    @staticmethod
    def forward(ctx, features, indices):
        """
        Gather
        :param ctx:
        :param features: features of points, FloatTensor[B, C, N]
        :param indices: centers' indices in points, IntTensor[b, m]
        :return:
            centers_coords: coordinates of sampled centers, FloatTensor[B, C, M]
        """
        features = features.contiguous()
        indices = indices.int().contiguous()
        ctx.save_for_backward(indices)
        ctx.num_points = features.size(-1)
        return _backend.gather_features_forward(features, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_features = _backend.gather_features_backward(
            grad_output.contiguous(), indices, ctx.num_points)
        return grad_features, None


gather = Gather.apply


def furthest_point_sample(coords, num_samples, normals=None):
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance to the sampled point set
    :param coords: coordinates of points, FloatTensor[B, 3, N]
    :param num_samples: int, M
    :return:
       center_coords: coordinates of sampled centers, FloatTensor[B, 3, M]
    """
    assert(len(coords.shape) == 3 and coords.shape[1] == 3), f'expect input as B,3,N; get: {coords.shape}'
    coords = coords.contiguous()
    indices = _backend.furthest_point_sampling(coords, num_samples)
    centers_coords = gather(coords, indices)
    if normals is not None:
        center_normals = gather(normals, indices)
    return centers_coords if normals is None else (centers_coords, center_normals) 


def logits_mask(coords, logits, num_points_per_object):
    """
    Use logits to sample points
    :param coords: coords of points, FloatTensor[B, 3, N]
    :param logits: binary classification logits, FloatTensor[B, 2, N]
    :param num_points_per_object: M, #points per object after masking, int
    :return:
        selected_coords: FloatTensor[B, 3, M]
        masked_coords_mean: mean coords of selected points, FloatTensor[B, 3]
        mask: mask to select points, BoolTensor[B, N]
    """
    batch_size, _, num_points = coords.shape
    mask = torch.lt(logits[:, 0, :], logits[:, 1, :])  # [B, N]
    num_candidates = torch.sum(mask, dim=-1, keepdim=True)  # [B, 1]
    masked_coords = coords * mask.view(batch_size, 1, num_points)  # [B, C, N]
    masked_coords_mean = torch.sum(masked_coords, dim=-1) / torch.max(
        num_candidates, torch.ones_like(num_candidates)).float()  # [B, C]
    selected_indices = torch.zeros((batch_size, num_points_per_object),
                                   device=coords.device,
                                   dtype=torch.int32)
    for i in range(batch_size):
        current_mask = mask[i]  # [N]
        current_candidates = current_mask.nonzero().view(-1)
        current_num_candidates = current_candidates.numel()
        if current_num_candidates >= num_points_per_object:
            choices = np.random.choice(current_num_candidates,
                                       num_points_per_object,
                                       replace=False)
            selected_indices[i] = current_candidates[choices]
        elif current_num_candidates > 0:
            choices = np.concatenate([
                np.arange(current_num_candidates).repeat(
                    num_points_per_object // current_num_candidates),
                np.random.choice(current_num_candidates,
                                 num_points_per_object %
                                 current_num_candidates,
                                 replace=False)
            ])
            np.random.shuffle(choices)
            selected_indices[i] = current_candidates[choices]
    selected_coords = gather(
        masked_coords - masked_coords_mean.view(batch_size, -1, 1),
        selected_indices)
    return selected_coords, masked_coords_mean, mask
