
#ifdef _DEBUG
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_forward_cuda_kernel(float* image, float* grid, float* out, int height, int width) {

	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int b = girdIdx.x;

}

torch::Tensor warp_forward_cuda(torch::Tensor image, torch::Tensor grid) {

	torch::Tensor out = torch::zeros_like(image);
	//int batch_size = image.size(0);
	//int height = image.size(2);
	//int width = image.size(3);

	//dim3 threads(16, 16);
	//dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);
	////dim3 grids(batch_size);

	//warp_forward_cuda_kernel<<<10, 100>>>(0, 0, 0, 100, 100);
	////	image.data<float>(),
	////	gird.data<float>(),
	////	out.data<float>(),
	////	height, width);

	return out;
}

