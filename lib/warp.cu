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

__global__ void warp_forward_cuda_kernel(float* image, float* sample, float* out, int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (row >= height || col >= width) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * 0.5 * (sample[idx_u] + 1.0);
	float v = (height - 1) * 0.5 * (sample[idx_v] + 1.0);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu >= 0 && iv >= 0 && iu < (width - 1) && iv < (height - 1)) {
		float uu = u - iu;
		float vv = v - iv;
		float w11 = (1.0 - uu) * (1.0 - vv);
		float w12 = uu * (1.0 - vv);
		float w21 = (1.0 - uu) * vv;
		float w22 = uu * vv;

		for (int c = 0; c < channels; c++) {
			size_t idx_11 = batch_id * (channels * height * width) + c * (height * width) + (iv + 0) * width + (iu + 0);
			size_t idx_12 = batch_id * (channels * height * width) + c * (height * width) + (iv + 0) * width + (iu + 1);
			size_t idx_21 = batch_id * (channels * height * width) + c * (height * width) + (iv + 1) * width + (iu + 0);
			size_t idx_22 = batch_id * (channels * height * width) + c * (height * width) + (iv + 1) * width + (iu + 1);
			size_t idx_out = batch_id * (channels * height * width) + c * (height * width) + row * width + col;

			out[idx_out] = w11 * image[idx_11] + w12 * image[idx_12] + w21 * image[idx_21] + w22 * image[idx_22];
		}
	}


}

__global__ void warp_backward_cuda_kernel(float* image, float* sample, float* grad, float* out, int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (row >= height || col >= width) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * 0.5 * (sample[idx_u] + 1.0);
	float v = (height - 1) * 0.5 * (sample[idx_v] + 1.0);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu >= 0 && iv >= 0 && iu < (width - 1) && iv < (height - 1)) {
		float uu = u - iu;
		float vv = v - iv;

		float dw11du = -(1.0 - vv) * (width - 1) * 0.5;
		float dw12du = (1.0 - vv) * (width - 1) * 0.5;
		float dw21du = -vv * (width - 1) * 0.5;
		float dw22du = vv * (width - 1) * 0.5;

		float dw11dv = -(1.0 - uu) * (height - 1) * 0.5;
		float dw12dv = -uu * (height - 1) * 0.5;
		float dw21dv = (1.0 - uu) * (height - 1) * 0.5;
		float dw22dv = uu * (height - 1) * 0.5;

		float gu = 0.0;
		float gv = 0.0;
		for (int c = 0; c < channels; c++) {
			size_t idx_11 = batch_id * (channels * height * width) + c * (height * width) + (iv + 0) * width + (iu + 0);
			size_t idx_12 = batch_id * (channels * height * width) + c * (height * width) + (iv + 0) * width + (iu + 1);
			size_t idx_21 = batch_id * (channels * height * width) + c * (height * width) + (iv + 1) * width + (iu + 0);
			size_t idx_22 = batch_id * (channels * height * width) + c * (height * width) + (iv + 1) * width + (iu + 1);
			size_t idx_gd = batch_id * (channels * height * width) + c * (height * width) + row * width + col;
			gu += grad[idx_gd] * (dw11du * image[idx_11] + dw12du * image[idx_12] + dw21du * image[idx_21] + dw22du * image[idx_22]);
			gv += grad[idx_gd] * (dw11dv * image[idx_11] + dw12dv * image[idx_12] + dw21dv * image[idx_21] + dw22dv * image[idx_22]);
		}

		out[idx_u] = gu;
		out[idx_v] = gv;
	}
}

torch::Tensor warp_forward_cuda(torch::Tensor image, torch::Tensor sample) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor out = torch::zeros_like(image);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_forward_cuda_kernel << <blocks, threads >> > (image_data, sample_data, out_data, channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
#else
	cudaDeviceSynchronize();
#endif

	return out;
}

torch::Tensor warp_backward_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor grad) {

	torch::Tensor out = torch::zeros_like(sample);
	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* grad_data = grad.data<float>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_backward_cuda_kernel << <blocks, threads >> > (image_data, sample_data, grad_data, out_data, channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
#else
	cudaDeviceSynchronize();
#endif

	return out;
}


torch::Tensor warp_forward(torch::Tensor image, torch::Tensor sample) {

	TORCH_CHECK(image.is_contiguous());
	TORCH_CHECK(sample.is_contiguous());
	TORCH_CHECK(image.type().is_cuda());
	TORCH_CHECK(sample.type().is_cuda());
	TORCH_CHECK(image.dtype() == torch::kFloat32);
	TORCH_CHECK(sample.dtype() == torch::kFloat32);
	TORCH_CHECK(image.dim() == 4);
	TORCH_CHECK(sample.dim() == 4);
	TORCH_CHECK(image.size(0) == sample.size(0));
	TORCH_CHECK(sample.size(1) == 2);
	TORCH_CHECK(image.size(2) == sample.size(2));
	TORCH_CHECK(image.size(3) == sample.size(3));

	return warp_forward_cuda(image, sample);

}

torch::Tensor warp_backward(torch::Tensor image, torch::Tensor sample, torch::Tensor grad) {

	TORCH_CHECK(image.is_contiguous());
	TORCH_CHECK(sample.is_contiguous());
	TORCH_CHECK(grad.is_contiguous());
	TORCH_CHECK(image.type().is_cuda());
	TORCH_CHECK(sample.type().is_cuda());
	TORCH_CHECK(grad.type().is_cuda());
	TORCH_CHECK(image.dtype() == torch::kFloat32);
	TORCH_CHECK(sample.dtype() == torch::kFloat32);
	TORCH_CHECK(grad.dtype() == torch::kFloat32);
	TORCH_CHECK(image.dim() == 4);
	TORCH_CHECK(sample.dim() == 4);
	TORCH_CHECK(grad.dim() == 4);
	TORCH_CHECK(image.size(0) == sample.size(0));
	TORCH_CHECK(image.size(0) == grad.size(0));
	TORCH_CHECK(sample.size(1) == 2);
	TORCH_CHECK(image.size(1) == grad.size(1));
	TORCH_CHECK(image.size(2) == sample.size(2));
	TORCH_CHECK(image.size(2) == grad.size(2));
	TORCH_CHECK(image.size(3) == sample.size(3));
	TORCH_CHECK(image.size(3) == grad.size(3));

	return warp_backward_cuda(image, sample, grad);

}

#ifdef _DEBUG
int main() {
	torch::TensorOptions option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
	torch::Tensor image = torch::rand({ 2, 3, 128, 256 }, option);
	torch::Tensor sample = torch::rand({ 2, 2, 128, 256 }, option);
	torch::Tensor warped = warp_forward(image, sample);

	torch::Tensor grad = torch::zeros({ 2, 3, 128, 256 }, option);
	torch::Tensor grad_sample = warp_backward(image, sample, grad);
    
	return 0;
}
#endif




