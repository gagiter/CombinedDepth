
from torch.utils.cpp_extension import load

libwarp = load(name="libwarp", sources=["warp.cu"])

# libwarp = load(
#     name="libwarp",
#     sources=['warp.cpp', 'warp.cu'],
#     verbose=True
#  )

help(libwarp)
