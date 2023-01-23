import torch
#import emd_cuda
# from evaluation.PyTorchEMD import emd_cuda 
from third_party.PyTorchEMD.backend import emd_cuda_dynamic as emd_cuda 


class EarthMoverDistanceFunctionNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        # ctx.save_for_backward(xyz1, xyz2, match)
        return cost


def earth_mover_distance_nograd(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    # xyz1: B,N,3
    N = xyz1.shape[1]
    assert(xyz1.shape[-1] == 3), f'require it to be B,N,3; get: {xyz1.shape}'
    #print('xyz1: ', xyz1.shape, xyz2.shape, xyz1.min(), xyz1.max(), xyz2.min(), xyz2.max())
    cost = EarthMoverDistanceFunctionNoGrad.apply(xyz1, xyz2) / float(N)
    return cost

