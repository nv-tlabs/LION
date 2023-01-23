from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_2D',
    ext_modules=[
        CUDAExtension('chamfer_2D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer2D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })