#ifndef LIB_WARP_H
#define LIB_WARP_H

#ifdef _DEBUG
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif
#include <vector>


torch::Tensor warp_forward(torch::Tensor image, torch::Tensor sample);
torch::Tensor warp_backward(torch::Tensor image, torch::Tensor sample, torch::Tensor grad);

std::vector<torch::Tensor> warp_forward_direct(torch::Tensor image, torch::Tensor sample, torch::Tensor depth);
torch::Tensor warp_backward_direct(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor grad);

#ifndef _DEBUG
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &warp_forward, "warp forward");
	m.def("backward", &warp_backward, "warp backward");
	m.def("forward_direct", &warp_forward_direct, "warp forward");
	m.def("backward_direct", &warp_backward_direct, "warp backward");
}
#endif

#endif

