from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_5D',
    ext_modules=[
        CUDAExtension('chamfer_5D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer5D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })