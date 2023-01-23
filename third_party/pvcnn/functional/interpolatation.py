from torch.autograd import Function

# from modules.functional.backend import _backend
from third_party.pvcnn.functional.backend import _backend
import torch
from torch.cuda.amp import autocast, GradScaler, custom_fwd, custom_bwd 

__all__ = ['nearest_neighbor_interpolate']


class NeighborInterpolation(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) 
    def forward(ctx, points_coords, centers_coords, centers_features):
        """
        :param ctx:
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
        :param centers_features: features of centers, FloatTensor[B, C, M]
        :return:
            points_features: features of points, FloatTensor[B, C, N]
        """
        centers_coords = centers_coords[:,:3].contiguous()
        points_coords = points_coords[:,:3].contiguous()
        centers_features = centers_features.contiguous()
        points_features, indices, weights = _backend.three_nearest_neighbors_interpolate_forward(
            points_coords, centers_coords, centers_features)
        ctx.save_for_backward(indices, weights)
        ctx.num_centers = centers_coords.size(-1)
        return points_features

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        grad_centers_features = _backend.three_nearest_neighbors_interpolate_backward(
            grad_output.contiguous(), indices, weights, ctx.num_centers)
        return None, None, grad_centers_features


nearest_neighbor_interpolate = NeighborInterpolation.apply

#def nearest_neighbor_interpolate(points_coords, centers_coords, centers_features):
#    # points_coords:         (B,6,  64)
#    # centers_coords:        (B,6,  16)
#    # centers_features:      (B,128,16)
#    # interpolated_features: (B,128,64) 
#    B = points_coords.shape[0] 
#    D = centers_features.shape[1]
#    N = points_coords.shape[2] 
#    output = torch.zeros(B,D,N).to(points_coords.shape) 
#    for b in range(B):
#        for n in range(N):
#            points_coords_cur = points_coords 
