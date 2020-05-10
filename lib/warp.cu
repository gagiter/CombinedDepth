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

__global__ void warp_backward_cuda_kernel(
	float* image, float* sample, float* depth, float* weight, float* grad, float* out, 
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

	float wd = depth[batch_id * height * width + row * width + col];
	float uu = u - iu;
	float vv = v - iv;
	int offset[4] = { iv * width + iu, iv * width + iu + 1, (iv + 1) * width + iu, (iv + 1) * width + iu + 1 };
	float ww[4] = { (1.0f - uu) * (1.0f - vv), uu * (1.0f - vv), (1.0f - uu) * vv, uu * vv };

	float gu = 0.0;
	float gv = 0.0;
	for (int i = 0; i < 4; i++) {
		size_t base = batch_id * height * width;
		float weight0 = weight[base + offset[i]];
		float weight1 = ww[i] * wd;

		if (weight0 > FLT_EPSILON) {
			float w = weight1 / weight0;
			for (int c = 0; c < channels; c++) {
				size_t channel_base = batch_id * channels * height * width + c * height * width;
				size_t idx_center = channel_base + row * width + col;
				size_t idx_left = idx_center - 1;
				size_t idx_right = idx_center + 1;
				size_t idx_top = idx_center - width;
				size_t idx_bottom = idx_center + width;

				gu += grad[channel_base + offset[i]] * (image[idx_left] - image[idx_right]) * (width - 1) * w;
				gv += grad[channel_base + offset[i]] * (image[idx_top] - image[idx_bottom]) * (height - 1) * w;
			}
		}
	}
	out[idx_u] = gu;
	out[idx_v] = gv;
}


torch::Tensor warp_backward_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor weight, torch::Tensor grad) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor out = torch::zeros_like(sample);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* depth_data = depth.data<float>();
	float* weight_data = weight.data<float>();
	float* grad_data = grad.data<float>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_backward_cuda_kernel << <blocks, threads >> > (
		image_data, sample_data, depth_data, weight_data, grad_data, out_data, channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
//#else
//	cudaDeviceSynchronize();
#endif

	return out;
}


torch::Tensor warp_backward(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor weight, torch::Tensor grad) {

	TORCH_CHECK(image.is_contiguous());
	TORCH_CHECK(sample.is_contiguous());
	TORCH_CHECK(depth.is_contiguous());
	TORCH_CHECK(weight.is_contiguous());
	TORCH_CHECK(grad.is_contiguous());
	TORCH_CHECK(image.type().is_cuda());
	TORCH_CHECK(sample.type().is_cuda());
	TORCH_CHECK(depth.type().is_cuda());
	TORCH_CHECK(weight.type().is_cuda());
	TORCH_CHECK(grad.type().is_cuda());
	TORCH_CHECK(image.dtype() == torch::kFloat32);
	TORCH_CHECK(sample.dtype() == torch::kFloat32);
	TORCH_CHECK(depth.dtype() == torch::kFloat32);
	TORCH_CHECK(weight.dtype() == torch::kFloat32);
	TORCH_CHECK(grad.dtype() == torch::kFloat32);
	TORCH_CHECK(image.dim() == 4);
	TORCH_CHECK(sample.dim() == 4);
	TORCH_CHECK(depth.dim() == 4);
	TORCH_CHECK(weight.dim() == 4);
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
	TORCH_CHECK(weight.size(0) == batch_num);
	TORCH_CHECK(weight.size(1) == 1);
	TORCH_CHECK(weight.size(2) == height);
	TORCH_CHECK(weight.size(3) == width);
	TORCH_CHECK(weight.device() == image.device());
	TORCH_CHECK(grad.size(0) == batch_num);
	TORCH_CHECK(grad.size(1) == channels);
	TORCH_CHECK(grad.size(2) == height);
	TORCH_CHECK(grad.size(3) == width);
	TORCH_CHECK(grad.device() == image.device());

	return warp_backward_cuda(image, sample, depth, weight, grad);
}


__global__ void warp_forward_cuda_kernel(
	float* image, float* sample, float* depth, float* weight, unsigned char* mask, float* out, int* lock,
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

	float wd = depth[batch_id * height * width + row * width + col];
	float uu = u - iu;
	float vv = v - iv;
	size_t offset[4] = { iv * width + iu, iv * width + iu + 1, (iv + 1) * width + iu, (iv + 1) * width + iu + 1 };
	float ww[4] = { (1.0f - uu) * (1.0f - vv), uu * (1.0f - vv), (1.0f - uu) * vv, uu * vv };
	size_t base = batch_id * height * width;

	for (int i = 0; i < 4; i++) {
		bool flag = true;
		while (flag) {
			if (atomicExch(&(lock[base + offset[i]]), 1u) == 0u) {
				float weight0 = weight[base + offset[i]];
				float weight1 = weight0 + ww[i] * wd;
				if (weight1 > FLT_EPSILON) {
					for (int c = 0; c < channels; c++) {
						size_t channel_base = batch_id * channels * height * width + c * height * width;
						float color = image[channel_base + row * width + col];
						float warp0 = out[channel_base + offset[i]];
						out[channel_base + offset[i]] = (weight0 * warp0 + ww[i] * wd * color) / weight1;
					}
					mask[base + offset[i]] = 1;
					weight[base + offset[i]] = weight1;
				}
				flag = false;
				atomicExch(&(lock[base + offset[i]]), 0u);
			}
		}
	}
}


std::vector<torch::Tensor> warp_forward_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor depth) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor weight = torch::zeros_like(depth);
	torch::TensorOptions lock_option = torch::TensorOptions().dtype(torch::kInt32).device(depth.device());
	torch::Tensor lock = torch::zeros_like(depth, lock_option);
	TORCH_CHECK(lock.dtype() == torch::kInt32);
	torch::TensorOptions mask_option = torch::TensorOptions().dtype(torch::kUInt8).device(depth.device());
	torch::Tensor mask = torch::zeros_like(depth, mask_option);
	TORCH_CHECK(mask.dtype() == torch::kUInt8);
	torch::Tensor out = torch::zeros_like(image);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* depth_data = depth.data<float>();
	float* weight_data = weight.data<float>();
	int* lock_data = lock.data<int>();
	unsigned char* mask_data = mask.data<unsigned char>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_forward_cuda_kernel << <blocks, threads >> > (
		image_data, sample_data, depth_data, weight_data, mask_data, out_data, lock_data,
		channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//#else
	//	cudaDeviceSynchronize();
#endif

	return { out, weight, mask };

}


std::vector<torch::Tensor> warp_forward(torch::Tensor image, torch::Tensor sample, torch::Tensor depth) {

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

	return warp_forward_cuda(image, sample, depth);
}


#ifdef _DEBUG
int main() {
	torch::TensorOptions option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
	torch::Tensor image = torch::rand({ 2, 3, 128, 256 }, option);
	torch::Tensor sample = torch::rand({ 2, 2, 128, 256 }, option);
	torch::Tensor depth = torch::rand({ 2, 1, 128, 256 }, option);

	std::vector<torch::Tensor> warp_output = warp_forward(image, sample, depth);
	torch::Tensor grad = torch::zeros({ 2, 3, 128, 256 }, option);
	torch::Tensor grad_sample = warp_backward(image, sample, depth, warp_output[1], grad);

	return 0;
}
#endif




