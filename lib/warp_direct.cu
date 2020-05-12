#include "warp.h"

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _DEBUG

#define checkCudaErrors(val) warp_direct_check_cuda((val), #val, __FILE__, __LINE__)
void warp_direct_check_cuda(cudaError_t result, char const* const func, const char* file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

#endif


__global__ void warp_backward_direct_cuda_kernel(
	float* image, float* sample, float* depth, float* record, float* grad, float* out, 
	int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (col <= 0 || col >= (width - 1) || row <= 0 || row >= (height - 1)) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * (0.5 * sample[idx_u] + 0.5);
	float v = (height - 1) * (0.5 * sample[idx_v] + 0.5);
	int iu = (int)floor(u);
	int iv = (int)floor(v);
	if (iu <= 0 || iu >= (width - 1) || iv <= 0 || iv >= (height - 1)) { return; }

	size_t idx_depth = batch_id * (1 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_record = batch_id * (1 * height * width) + 0 * (height * width) + iv * width + iu;

	bool visible = depth[idx_depth] >= record[idx_record];
	if (!visible) { return; }

	float gu = 0.0;
	float gv = 0.0;
	for (int c = 0; c < channels; c++) {
		size_t idx_center = batch_id * (channels * height * width) + c * (height * width) + row * width + col;
		size_t idx_left = idx_center - 1;
		size_t idx_right = idx_center + 1;
		size_t idx_top = idx_center - width;
		size_t idx_bottom = idx_center + width;
		size_t idx_gd = batch_id * (channels * height * width) + c * (height * width) + iv * width + iu;
		gu += grad[idx_gd] * (image[idx_left] - image[idx_right]) * 0.5 * (width - 1);
		gv += grad[idx_gd] * (image[idx_top] - image[idx_bottom]) * 0.5 * (height - 1);
	}
	out[idx_u] = gu;
	out[idx_v] = gv;
}


torch::Tensor warp_backward_direct_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor grad) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor out = torch::zeros_like(sample);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* depth_data = depth.data<float>();
	float* record_data = record.data<float>();
	float* grad_data = grad.data<float>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_backward_direct_cuda_kernel << <blocks, threads >> > (
		image_data, sample_data, depth_data, record_data, grad_data, out_data, channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
//#else
//	cudaDeviceSynchronize();
#endif

	return out;
}


torch::Tensor warp_backward_direct(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor grad) {

	TORCH_CHECK(image.is_contiguous());
	TORCH_CHECK(sample.is_contiguous());
	TORCH_CHECK(depth.is_contiguous());
	TORCH_CHECK(record.is_contiguous());
	TORCH_CHECK(grad.is_contiguous());
	TORCH_CHECK(image.type().is_cuda());
	TORCH_CHECK(sample.type().is_cuda());
	TORCH_CHECK(depth.type().is_cuda());
	TORCH_CHECK(record.type().is_cuda());
	TORCH_CHECK(grad.type().is_cuda());
	TORCH_CHECK(image.dtype() == torch::kFloat32);
	TORCH_CHECK(sample.dtype() == torch::kFloat32);
	TORCH_CHECK(depth.dtype() == torch::kFloat32);
	TORCH_CHECK(record.dtype() == torch::kFloat32);
	TORCH_CHECK(grad.dtype() == torch::kFloat32);
	TORCH_CHECK(image.dim() == 4);
	TORCH_CHECK(sample.dim() == 4);
	TORCH_CHECK(depth.dim() == 4);
	TORCH_CHECK(record.dim() == 4);
	TORCH_CHECK(grad.dim() == 4);
	int batch_num = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);
	TORCH_CHECK(sample.size(0) == batch_num);
	TORCH_CHECK(sample.size(1) == 2);
	TORCH_CHECK(sample.size(2) == height);
	TORCH_CHECK(sample.size(3) == width);
	TORCH_CHECK(sample.device() == image.device());
	TORCH_CHECK(depth.size(0) == batch_num);
	TORCH_CHECK(depth.size(1) == 1);
	TORCH_CHECK(depth.size(2) == height);
	TORCH_CHECK(depth.size(3) == width);
	TORCH_CHECK(depth.device() == image.device());
	TORCH_CHECK(record.size(0) == batch_num);
	TORCH_CHECK(record.size(1) == 1);
	TORCH_CHECK(record.size(2) == height);
	TORCH_CHECK(record.size(3) == width);
	TORCH_CHECK(record.device() == image.device());
	TORCH_CHECK(grad.size(0) == batch_num);
	TORCH_CHECK(grad.size(1) == channels);
	TORCH_CHECK(grad.size(2) == height);
	TORCH_CHECK(grad.size(3) == width);
	TORCH_CHECK(grad.device() == image.device());

	return warp_backward_direct_cuda(image, sample, depth, record, grad);
}


__global__ void warp_forward_direct_cuda_kernel(
	float* image, float* sample, float* depth, float* record, float* out, int* lock,
	int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (col == 0 || col >= (width - 1) || row == 0 || row >= (height - 1)) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * (0.5 * sample[idx_u] + 0.5);
	float v = (height - 1) * (0.5 * sample[idx_v] + 0.5);
	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu < 0 || iu >= (width - 1) || iv < 0 || iv >= (height - 1)) { return; }

	size_t idx_depth = batch_id * (1 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_record = batch_id * (1 * height * width) + 0 * (height * width) + iv * width + iu;

	bool visible = depth[idx_depth] > record[idx_record];
	if (!visible) { return; }

	while (visible) {
		if (atomicExch(&(lock[idx_record]), 1u) == 0u) {
			visible = depth[idx_depth] > record[idx_record];
			if (visible) {
				for (int c = 0; c < channels; c++) {
					size_t idx_image = batch_id * (channels * height * width) + c * (height * width) + row * width + col;
					size_t idx_out = batch_id * (channels * height * width) + c * (height * width) + iv * width + iu;
					out[idx_out] = image[idx_image];
				}
				record[idx_record] = depth[idx_depth];
				visible = false;
			}
			atomicExch(&(lock[idx_record]), 0u);
		}
	}
}


std::vector<torch::Tensor> warp_forward_direct_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor depth) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor record = torch::zeros_like(depth);
	torch::TensorOptions lock_option = torch::TensorOptions().dtype(torch::kInt32).device(depth.device());
	torch::Tensor lock = torch::zeros_like(depth, lock_option);
	TORCH_CHECK(lock.dtype() == torch::kInt32);
	torch::Tensor out = torch::zeros_like(image);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* depth_data = depth.data<float>();
	float* record_data = record.data<float>();
	int* lock_data = lock.data<int>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_forward_direct_cuda_kernel << <blocks, threads >> > (
		image_data, sample_data, depth_data, record_data, out_data, lock_data,
		channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//#else
	//	cudaDeviceSynchronize();
#endif

	return { out, record };

}


std::vector<torch::Tensor> warp_forward_direct(torch::Tensor image, torch::Tensor sample, torch::Tensor depth) {

	TORCH_CHECK(image.is_contiguous());
	TORCH_CHECK(sample.is_contiguous());
	TORCH_CHECK(depth.is_contiguous());
	TORCH_CHECK(image.type().is_cuda());
	TORCH_CHECK(sample.type().is_cuda());
	TORCH_CHECK(depth.type().is_cuda());
	TORCH_CHECK(image.dtype() == torch::kFloat32);
	TORCH_CHECK(sample.dtype() == torch::kFloat32);
	TORCH_CHECK(depth.dtype() == torch::kFloat32);
	TORCH_CHECK(image.dim() == 4);
	TORCH_CHECK(sample.dim() == 4);
	TORCH_CHECK(depth.dim() == 4);
	int batch_num = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);
	TORCH_CHECK(sample.size(0) == batch_num);
	TORCH_CHECK(sample.size(1) == 2);
	TORCH_CHECK(sample.size(2) == height);
	TORCH_CHECK(sample.size(3) == width);
	TORCH_CHECK(sample.device() == image.device());
	TORCH_CHECK(depth.size(0) == batch_num);
	TORCH_CHECK(depth.size(1) == 1);
	TORCH_CHECK(depth.size(2) == height);
	TORCH_CHECK(depth.size(3) == width);
	TORCH_CHECK(depth.device() == image.device());

	return warp_forward_direct_cuda(image, sample, depth);
}


#ifdef _DEBUG
int main() {
	torch::TensorOptions option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
	torch::Tensor image = torch::rand({ 2, 3, 128, 256 }, option);
	torch::Tensor sample = torch::rand({ 2, 2, 128, 256 }, option);
	torch::Tensor depth = torch::rand({ 2, 1, 128, 256 }, option);
	torch::Tensor grad = torch::rand({ 2, 3, 128, 256 }, option);

	torch::Tensor warp_output = warp_forward(image, sample);
	torch::Tensor grad_sample = warp_backward(image, sample, grad);

	std::vector<torch::Tensor> warp_direct_output = warp_forward_direct(image, sample, depth);
	torch::Tensor grad_direct_sample = warp_backward_direct(image, sample, depth, warp_direct_output[1], grad);

	return 0;
}
#endif




