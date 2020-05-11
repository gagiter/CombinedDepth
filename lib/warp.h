#ifndef LIB_WARP_H
#define LIB_WARP_H

#ifdef _DEBUG
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif
#include <vector>

std::vector<torch::Tensor> warp_forward(torch::Tensor image, torch::Tensor sample, torch::Tensor depth);
torch::Tensor warp_backward(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor grad);

#ifndef _DEBUG
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &warp_forward, "warp forward");
	m.def("backward", &warp_backward, "warp backward");
}
#endif

#endif

