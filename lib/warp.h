#ifndef LIB_WARP_H
#define LIB_WARP_H

#ifdef _DEBUG
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif

torch::Tensor warp_forward(torch::Tensor image, torch::Tensor grid);
torch::Tensor warp_backward(torch::Tensor image, torch::Tensor grid, torch::Tensor grad);
torch::Tensor warp_forward_cuda(torch::Tensor image, torch::Tensor grid);
torch::Tensor warp_backward_cuda(torch::Tensor image, torch::Tensor grid, torch::Tensor grad);

#ifndef _DEBUG
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &warp_forward, "warp forward");
	m.def("backward", &warp_backward, "warp backward");
}
#endif

#endif
//#include <torch/extension.h>

//torch::Tensor warp_forward(torch::Tensor image, torch::Tensor grid);
//torch::Tensor warp_backward(torch::Tensor image, torch::Tensor grid)
//
//torch::Tensor warp_cuda_forward(torch::Tensor image, torch::Tensor grid);
//torch::Tensor warp_cuda_backward(torch::Tensor image, torch::Tensor grid);

//#include <torch/torch.h>


//torch::Tensor warp_forward_cuda(torch::Tensor image, torch::Tensor grid);
