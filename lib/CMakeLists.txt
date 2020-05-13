
cmake_minimum_required(VERSION 3.0)

project(libwarp)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

#cuda_add_executable(${PROJECT_NAME}
cuda_add_executable(${PROJECT_NAME}
    warp.h
    warp.cu
)
# warp.cpp

target_link_libraries(${PROJECT_NAME}
	${TORCH_LIBRARIES}
)

#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET libwarp
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:libwarp>)
#endif (MSVC)