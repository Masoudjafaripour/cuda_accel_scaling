from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sobel_cuda',
    ext_modules=[
        CUDAExtension('sobel_cuda', [
            'sobel.cpp',
            'sobel_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
