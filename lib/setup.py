#https://github.com/facebookresearch/pytorch3d/issues/10
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# setup(name='libwarp',
#       ext_modules=[CUDAExtension('libwarp', ['warp.cpp', 'warp.cu'])],
#       cmdclass={'build_ext': BuildExtension})


setup(name='libwarp',
      ext_modules=[CppExtension('libwarp', ['warp.cpp'])],
      cmdclass={'build_ext': BuildExtension})