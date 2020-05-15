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

torch::Tensor warp_forward_wide(torch::Tensor image, torch::Tensor sample);
torch::Tensor warp_backward_wide(torch::Tensor image, torch::Tensor sample, torch::Tensor grad);

std::vector<torch::Tensor> warp_forward_direct(torch::Tensor image, torch::Tensor sample, torch::Tensor depth);
torch::Tensor warp_backward_direct(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor grad);

std::vector<torch::Tensor> warp_forward_record(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, float sigma);
torch::Tensor warp_backward_record(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor weight, torch::Tensor grad, float sigma);


#ifndef _DEBUG
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &warp_forward, "warp forward");
	m.def("backward", &warp_backward, "warp backward");
	m.def("forward_direct", &warp_forward_direct, "warp forward direct");
	m.def("backward_direct", &warp_backward_direct, "warp backward direct");
	m.def("forward_record", &warp_forward_record, "warp forward record");
	m.def("backward_record", &warp_backward_record, "warp backward record");
	m.def("forward_wide", &warp_forward_wide, "warp forward record");
	m.def("backward_wide", &warp_backward_wide, "warp backward record");
}
#endif

#endif

