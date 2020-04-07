
from torch.utils.cpp_extension import load

libwarp = load(name="libwarp", sources=["warp.cpp"])

# libwarp = load(
#     name="libwarp",
#     sources=['warp.cpp', 'warp.cu'],
#     verbose=True
#  )

help(libwarp)
