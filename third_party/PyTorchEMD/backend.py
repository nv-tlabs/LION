import os
import time
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(_src_path, 'build_dynamic')):
    os.makedirs(os.path.join(_src_path, 'build_dynamic'))
tic = time.time() 
emd_cuda_dynamic = load(name='emd_ext',
                extra_cflags=['-O3', '-std=c++17'],
                ## build_directory=os.path.join(_src_path, 'build_dynamic'),
                verbose=True,
                sources=[
                    os.path.join(_src_path, f) for f in [
                        'cuda/emd.cpp',
                        'cuda/emd_kernel.cu',
                    ]
                ])
print('load emd_ext time: {:.3f}s'.format(time.time() - tic))
__all__ = ['emd_cuda_dynamic']
