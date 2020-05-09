#ifndef LIB_WARP_H
#define LIB_WARP_H

#ifdef _DEBUG
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif

torch::Tensor warp_forward(torch::Tensor image, torch::Tensor sample);
torch::Tensor warp_backward(torch::Tensor image, torch::Tensor sample, torch::Tensor grad);
torch::Tensor warp_forward_cuda(torch::Tensor image, torch::Tensor sample);
torch::Tensor warp_backward_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor grad);

torch::Tensor warp_forward_with_occlusion(torch::Tensor image, torch::Tensor sample, torch::Tensor occlusion);
torch::Tensor warp_backward_with_occlusion(torch::Tensor image, torch::Tensor sample, torch::Tensor occlusion, torch::Tensor grad);
torch::Tensor warp_forward_with_occlusion_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor occlusion);
torch::Tensor warp_backward_with_occlusion_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor occlusion, torch::Tensor grad);

#ifndef _DEBUG
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &warp_forward, "warp forward");
	m.def("backward", &warp_backward, "warp backward");
	m.def("forward_with_occlusion", &warp_forward_with_occlusion, "warp forward with occlusion");
	m.def("backward_with_occlusion", &warp_backward_with_occlusion, "warp backward with occlusion");
}
#endif

#endif

