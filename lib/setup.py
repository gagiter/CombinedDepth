#https://github.com/facebookresearch/pytorch3d/issues/10
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='libwarp',
      ext_modules=[CUDAExtension('libwarp', ['warp.cu'])],
      cmdclass={'build_ext': BuildExtension})
