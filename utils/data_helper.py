# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

def normalize_point_clouds(pcs, mode='shape_bbox'):  # can be a property func
    '''
    copied from https://github.com/luost26/diffusion-point-cloud/blob/0bfd688379e78ac75fa75e6a2c5029e362496169/test_gen.py#L16
    Args:
        pcs: list of [N,3] or tensor in shape: B,N,3
    '''
    # logger.debug('Normalization mode: %s' % mode)
    assert(type(pcs) == list or len(pcs.shape) ==
           3), f'expect pcs to be list, get: {type(pcs)} or 3d tensor; '
    output_list = []
    for i in range(len(pcs)):  # , desc='Normalize'):
        pc = pcs[i]
        pc = pc.detach().clone()
        assert(mode == 'shape_bbox')
        assert(len(pc.shape) == 2 and pc.shape[-1] in [3, 4,
               6, 9]), f'expect get (N,3 or 6), get {pc.shape}'
        pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
        pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
        pc_min = pc_min[:, :3]
        pc_max = pc_max[:, :3]
        shift = ((pc_min + pc_max) / 2).view(1, 3)
        scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc[:, :3] = (pc[:, :3] - shift) / scale
        # pcs[i] = pc
        output_list.append(pc)
    return output_list

