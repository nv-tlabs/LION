import os

from torch.utils.cpp_extension import load
_src_path = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(_src_path, 'build')):
    os.makedirs(os.path.join(_src_path, 'build'))
_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                verbose=True,
                sources=[
                    os.path.join(_src_path, 'src', f) for f in [
                        'ball_query/ball_query.cpp',
                        'ball_query/ball_query.cu',
                        'grouping/grouping.cpp',
                        'grouping/grouping.cu',
                        'interpolate/neighbor_interpolate.cpp',
                        'interpolate/neighbor_interpolate.cu',
                        'interpolate/trilinear_devox.cpp',
                        'interpolate/trilinear_devox.cu',
                        'sampling/sampling.cpp',
                        'sampling/sampling.cu',
                        'voxelization/vox.cpp',
                        'voxelization/vox.cu',
                        'bindings.cpp',
                    ]
                ])

__all__ = ['_backend']
