#ifdef _DEBUG
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif



torch::Tensor warp_forward_cuda(torch::Tensor image, torch::Tensor grid);


torch::Tensor warp_forward(torch::Tensor image, torch::Tensor grid) {

	TORCH_CHECK(image.is_contiguous());
	TORCH_CHECK(grid.is_contiguous());
	TORCH_CHECK(image.type().is_cuda());
	TORCH_CHECK(grid.type().is_cuda());
	TORCH_CHECK(image.dtype() == torch::kFloat32);
	TORCH_CHECK(grid.dtype() == torch::kFloat32)
	TORCH_CHECK(image.dim() == 4);
	TORCH_CHECK(grid.dim() == 4);
	TORCH_CHECK(image.size(0) == grid.size(0));
	TORCH_CHECK(image.size(1) == 3);
	TORCH_CHECK(grid.size(1) == 2);
	TORCH_CHECK(image.size(2) == grid.size(2));
	TORCH_CHECK(image.size(3) == grid.size(3));

	//return warp_forward_cuda(image, grid);
	return image;

}

#ifndef _DEBUG
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &warp_forward, "warp forward");
	//m.def("backward", &warp_backward, "warp backward");
}
#endif 

//
//torch::Tensor warp_backward(torch::Tensor image, torch::Tensor grid) {
//    CHECK_INPUT(image);
//    CHECK_INPUT(grid);
//    return image;
//}
//

#ifdef _DEBUG
int main() {
	torch::TensorOptions option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
	torch::Tensor image = torch::rand({ 2, 3, 128, 256 }, option);
	torch::Tensor grid = torch::rand({ 2, 2, 128, 256 }, option);
	torch::Tensor warped = warp_forward(image, grid);
    
	return 0;
}
#endif // _DEBUG



