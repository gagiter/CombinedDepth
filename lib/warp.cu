#include "warp.h"

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _DEBUG

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

#endif

__global__ void warp_forward_cuda_kernel(float* image, float* grid, float* out, int height, int width) {

	int batch_id = blockIdx.x;
	int channel_id = threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	size_t idx_u = batch_id * (blockDim.x * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (blockDim.x * height * width) + 1 * (height * width) + row * width + col;
	size_t idx_out = batch_id * (blockDim.x * height * width) + channel_id * (height * width) + row * width + col;

	float u = (width - 1) * 0.5 * (grid[idx_u] + 1.0);
	float v = (height - 1) * 0.5 * (grid[idx_v] + 1.0);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu >= 0 && iv >= 0 && iu < (width - 1) && iv < (height - 1)) {
		float uu = u - iu;
		float vv = v - iv;
		float w11 = (1.0 - uu) * (1.0 - vv);
		float w12 = uu * (1.0 - vv);
		float w21 = (1.0 - uu) * vv;
		float w22 = uu * vv;

		size_t idx_11 = batch_id * (blockDim.x * height * width) + channel_id * (height * width) + row * (width + 0) + (col + 0);
		size_t idx_12 = batch_id * (blockDim.x * height * width) + channel_id * (height * width) + row * (width + 0) + (col + 1);
		size_t idx_21 = batch_id * (blockDim.x * height * width) + channel_id * (height * width) + row * (width + 1) + (col + 0);
		size_t idx_22 = batch_id * (blockDim.x * height * width) + channel_id * (height * width) + row * (width + 1) + (col + 1);

		out[idx_out] = w11 * image[idx_11] + w12 * image[idx_12] + w21 * image[idx_21] + w22 * image[idx_22];

	}


}

torch::Tensor warp_forward_cuda(torch::Tensor image, torch::Tensor grid) {

	torch::Tensor out = torch::zeros_like(image);
	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	float* image_data = image.data<float>();
	float* grid_data = grid.data<float>();
	float* out_data = out.data<float>();

	dim3 threads(channels, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_forward_cuda_kernel << <blocks, threads >> > (image_data, grid_data, out_data, height, width);
#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
#endif


	return out;
}


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

	return warp_forward_cuda(image, grid);
//	return image;

}



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



