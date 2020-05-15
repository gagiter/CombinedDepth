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

// 0 warp
__global__ void warp_backward_cuda_kernel(float* image, float* sample, float* grad, float* out, int channels, int height, int width);



// 2 record
std::vector<torch::Tensor> warp_forward_record(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, float sigma);
__global__ void warp_forward_record_record_cuda_kernel(float* image, float* sample, float* depth, float* record, float* weight, int* lock, int channels, int height, int width);
__global__ void warp_forward_record_warp_cuda_kernel(float* image, float* sample, float* depth, float* record, float* weight, float* out, int channels, int height, int width, float sigma);

torch::Tensor warp_backward_record(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor weight, torch::Tensor grad, float sigma);
torch::Tensor warp_backward_record_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor weight, torch::Tensor grad, float sigma);
__global__ void warp_backward_record_cuda_kernel(float* image, float* sample, float* depth, float* record, float* weight, float* grad, float* out, int channels, int height, int width, float sigma);



// 3 wide
__global__ void warp_forward_wide_cuda_kernel(float* image, float* sample, float* out, int channels, int height, int width);
torch::Tensor warp_backward_wide_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor grad);
__global__ void warp_backward_wide_cuda_kernel(float* image, float* sample, float* grad, float* out, int channels, int height, int width);



__global__ void warp_backward_wide_cuda_kernel(float* image, float* sample, float* grad, float* out, int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (row >= height || col >= width) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * 0.5f * (sample[idx_u] + 1.0f);
	float v = (height - 1) * 0.5f * (sample[idx_v] + 1.0f);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu > 0 && iv > 0 && iu < (width - 1) && iv < (height - 1)) {

		float gu = 0.0f;
		float gv = 0.0f;
		for (int c = 0; c < channels; c++) {
			size_t base_channel = batch_id * (channels * height * width) + c * (height * width);
			//size_t idx_center = base_channel + (iv + 0) * width + (iu + 0);
			size_t idx_right = base_channel + (iv + 0) * width + (iu + 1);
			size_t idx_left = base_channel + (iv + 0) * width + (iu - 1);
			size_t idx_top = base_channel + (iv - 1) * width + (iu + 0);
			size_t idx_bottom = base_channel + (iv + 1) * width + (iu + 0);
			size_t idx_gd = base_channel + row * width + col;
			float guu = (image[idx_right] - image[idx_left]) * 0.5f;
			float gvv = (image[idx_bottom] - image[idx_top]) * 0.5f;

			gu += grad[idx_gd] * guu * (width - 1) * 0.5f;
			gv += grad[idx_gd] * gvv * (height - 1) * 0.5f;

		}
		out[idx_u] = gu;
		out[idx_v] = gv;
	}
}


torch::Tensor warp_backward_wide_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor grad) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor out = torch::zeros_like(sample);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* grad_data = grad.data<float>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_backward_wide_cuda_kernel << <blocks, threads >> > (image_data, sample_data, grad_data, out_data, channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//#else
	//  cudaDeviceSynchronize();
#endif

	return out;
}


torch::Tensor warp_backward_wide(torch::Tensor image, torch::Tensor sample, torch::Tensor grad) {

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
	int batch_num = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	TORCH_CHECK(sample.size(0) == batch_num);
	TORCH_CHECK(sample.size(1) == 2);
	TORCH_CHECK(sample.size(2) == height);
	TORCH_CHECK(sample.size(3) == width);
	TORCH_CHECK(sample.device() == image.device());
	TORCH_CHECK(grad.size(0) == batch_num);
	TORCH_CHECK(grad.size(1) == channels);
	TORCH_CHECK(grad.size(2) == height);
	TORCH_CHECK(grad.size(3) == width);
	TORCH_CHECK(grad.device() == image.device());

	return warp_backward_wide_cuda(image, sample, grad);
}



__global__ void warp_forward_wide_cuda_kernel(float* image, float* sample, float* out, int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (row >= height || col >= width) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * 0.5f * (sample[idx_u] + 1.0f);
	float v = (height - 1) * 0.5f * (sample[idx_v] + 1.0f);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu > 0 && iv > 0 && iu < (width - 1) && iv < (height - 1)) {
		float uu = u - iu;
		float vv = v - iv;

		for (int c = 0; c < channels; c++) {
			size_t base_channel = batch_id * (channels * height * width) + c * (height * width);
			size_t idx_center = base_channel + (iv + 0) * width + (iu + 0);
			size_t idx_right = base_channel + (iv + 0) * width + (iu + 1);
			size_t idx_left = base_channel + (iv + 0) * width + (iu - 1);
			size_t idx_top = base_channel + (iv - 1) * width + (iu + 0);
			size_t idx_bottom = base_channel + (iv + 1) * width + (iu + 0);
			size_t idx_out = base_channel + row * width + col;
			float guu = (image[idx_right] - image[idx_left]) * 0.5f;
			float gvv = (image[idx_bottom] - image[idx_top]) * 0.5f;
			out[idx_out] = image[idx_center] + guu * uu + gvv * vv;
		}
	}
}


torch::Tensor warp_forward_wide_cuda(torch::Tensor image, torch::Tensor sample) {

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

	warp_forward_wide_cuda_kernel << <blocks, threads >> > (image_data, sample_data, out_data, channels, height, width);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//#else
	//  cudaDeviceSynchronize();
#endif

	return out;
}


torch::Tensor warp_forward_wide(torch::Tensor image, torch::Tensor sample) {

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
	TORCH_CHECK(image.device() == sample.device());

	return warp_forward_wide_cuda(image, sample);

}



__global__ void warp_backward_record_cuda_kernel(
	float* image, float* sample, float* depth, float* record, float* weight, float* grad, float* out,
	int channels, int height, int width, float sigma) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (row >= height || col >= width) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * 0.5f * (sample[idx_u] + 1.0f);
	float v = (height - 1) * 0.5f * (sample[idx_v] + 1.0f);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu < 0 || iu >= (width - 1) || iv < 0 || iv >= (height - 1)) { return; }

	int base = batch_id * height * width;
	float wwww = weight[base + row * width + col];

	if (wwww > 0.00001f) {

		float inv_sigma2 = 1.0f / sigma / sigma;
		float uu = u - iu;
		float vv = v - iv;
		int base = batch_id * height * width;
		float d = depth[base + row * width + col];
		int offset[4] = { iv * width + iu, iv * width + iu + 1, (iv + 1) * width + iu, (iv + 1) * width + iu + 1 };
		float wd[4] = { d - record[base + offset[0]], d - record[base + offset[1]], d - record[base + offset[2]], d - record[base + offset[3]] };
		float wdd[4] = { exp(-wd[0] * wd[0] * 0.5f * inv_sigma2), exp(-wd[1] * wd[1] * 0.5f * inv_sigma2),
			exp(-wd[2] * wd[2] * 0.5f * inv_sigma2), exp(-wd[3] * wd[3] * 0.5f * inv_sigma2) };
		float dw_du[4] = { -(1.0f - vv) * wdd[0], (1.0f - vv) * wdd[1], -vv * wdd[2], vv * wdd[3] };
		float dw_dv[4] = { -(1.0f - uu) * wdd[0], -uu * wdd[1], (1.0f - uu) * wdd[2], uu * wdd[3] };

		float gu = 0.0f;
		float gv = 0.0f;
		for (int c = 0; c < channels; c++) {
			size_t channel_base = batch_id * channels * height * width + c * height * width;
			float g[4] = { image[channel_base + offset[0]], image[channel_base + offset[1]], image[channel_base + offset[2]], image[channel_base + offset[3]] };
			size_t grad_idx = channel_base + row * width + col;
			gu += grad[grad_idx] * (dw_du[0] * g[0] + dw_du[1] * g[1] + dw_du[2] * g[2] + dw_du[3] * g[3]) * (width - 1) * 0.5f / wwww;
			gv += grad[grad_idx] * (dw_dv[0] * g[0] + dw_dv[1] * g[1] + dw_dv[2] * g[2] + dw_dv[3] * g[3]) * (height - 1) * 0.5f / wwww;
		}

		out[idx_u] = gu;
		out[idx_v] = gv;
	}


}


torch::Tensor warp_backward_record_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor weight, torch::Tensor grad, float sigma) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor out = torch::zeros_like(sample);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* depth_data = depth.data<float>();
	float* record_data = record.data<float>();
	float* weight_data = weight.data<float>();
	float* grad_data = grad.data<float>();
	float* out_data = out.data<float>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_backward_record_cuda_kernel << <blocks, threads >> > (
		image_data, sample_data, depth_data, record_data, weight_data, grad_data, out_data, channels, height, width, sigma);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//#else
	//	cudaDeviceSynchronize();
#endif

	return out;
}


torch::Tensor warp_backward_record(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, torch::Tensor record, torch::Tensor weight, torch::Tensor grad, float sigma) {


	TORCH_CHECK(image.is_contiguous());
	TORCH_CHECK(sample.is_contiguous());
	TORCH_CHECK(depth.is_contiguous());
	TORCH_CHECK(record.is_contiguous());
	TORCH_CHECK(weight.is_contiguous());
	TORCH_CHECK(grad.is_contiguous());
	TORCH_CHECK(image.type().is_cuda());
	TORCH_CHECK(sample.type().is_cuda());
	TORCH_CHECK(depth.type().is_cuda());
	TORCH_CHECK(record.type().is_cuda());
	TORCH_CHECK(weight.type().is_cuda());
	TORCH_CHECK(grad.type().is_cuda());
	TORCH_CHECK(image.dtype() == torch::kFloat32);
	TORCH_CHECK(sample.dtype() == torch::kFloat32);
	TORCH_CHECK(depth.dtype() == torch::kFloat32);
	TORCH_CHECK(record.dtype() == torch::kFloat32);
	TORCH_CHECK(weight.dtype() == torch::kFloat32);
	TORCH_CHECK(grad.dtype() == torch::kFloat32);
	TORCH_CHECK(image.dim() == 4);
	TORCH_CHECK(sample.dim() == 4);
	TORCH_CHECK(depth.dim() == 4);
	TORCH_CHECK(record.dim() == 4);
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
	TORCH_CHECK(record.size(0) == batch_num);
	TORCH_CHECK(record.size(1) == 1);
	TORCH_CHECK(record.size(2) == height);
	TORCH_CHECK(record.size(3) == width);
	TORCH_CHECK(record.device() == image.device());
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

	return warp_backward_record_cuda(image, sample, depth, record, weight, grad, sigma);
}



__global__ void warp_forward_record_record_cuda_kernel(
	float* image, float* sample, float* depth, float* record, float* weight, int* lock,
	int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (col >= width || row >= height) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * (0.5 * sample[idx_u] + 0.5);
	float v = (height - 1) * (0.5 * sample[idx_v] + 0.5);
	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu < 0 || iu >= (width - 1) || iv < 0 || iv >= (height - 1)) { return; }

	int base = batch_id * height * width;
	float d = depth[base + row * width + col];
	float uu = u - iu;
	float vv = v - iv;
	int offset[4] = { iv * width + iu, iv * width + iu + 1, (iv + 1) * width + iu, (iv + 1) * width + iu + 1 };
	float ww[4] = { (1.0f - uu) * (1.0f - vv), uu * (1.0f - vv), (1.0f - uu) * vv, uu * vv };


	for (int i = 0; i < 4; i++) {
		float w = ww[i] * d;
		bool visible = w > weight[base + offset[i]];
		while (visible) {
			if (atomicExch(&(lock[base + offset[i]]), 1u) == 0u) {
				visible = w > weight[base + offset[i]];
				if (visible) {
					record[base + offset[i]] = d;
					weight[base + offset[i]] = w;
					visible = false;
				}
				atomicExch(&(lock[base + offset[i]]), 0u);
			}
		}
	}
}


__global__ void warp_forward_record_warp_cuda_kernel(
	float* image, float* sample, float* depth, float* record, float* weight, float* out,
	int channels, int height, int width, float sigma) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (col >= width || row >= height) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * (0.5 * sample[idx_u] + 0.5);
	float v = (height - 1) * (0.5 * sample[idx_v] + 0.5);
	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu < 0 || iu >= (width - 1) || iv < 0 || iv >= (height - 1)) { return; }

	float inv_sigma2 = 1.0f / sigma / sigma;
	float uu = u - iu;
	float vv = v - iv;
	int base = batch_id * height * width;
	int offset[4] = { iv * width + iu, iv * width + iu + 1, (iv + 1) * width + iu, (iv + 1) * width + iu + 1 };
	float ww[4] = { (1.0f - uu) * (1.0f - vv), uu * (1.0f - vv), (1.0f - uu) * vv, uu * vv };
	float d = depth[base + row * width + col];
	float wd[4] = { d - record[base + offset[0]], d - record[base + offset[1]], d - record[base + offset[2]], d - record[base + offset[3]] };
	float wdd[4] = { exp(-wd[0] * wd[0] * 0.5f * inv_sigma2), exp(-wd[1] * wd[1] * 0.5f * inv_sigma2), 
		exp(-wd[2] * wd[2] * 0.5f * inv_sigma2), exp(-wd[3] * wd[3] * 0.5f * inv_sigma2) };
	float w[4] = { ww[0] * wdd[0], ww[1] * wdd[1], ww[2] * wdd[2], ww[3] * wdd[3] };
	float wwww = w[0] + w[1] + w[2] + w[3];
	if (wwww > 0.00001f) {
		for (int c = 0; c < channels; c++) {
			size_t channel_base = batch_id * channels * height * width + c * height * width;
			out[channel_base + row * width + col] = (w[0] * image[channel_base + offset[0]] + w[1] * image[channel_base + offset[1]] +
				w[2] * image[channel_base + offset[2]] + w[3] * image[channel_base + offset[3]]) / wwww;
		}
		weight[base + row * width + col] = wwww;
	}
}

std::vector<torch::Tensor> warp_forward_record_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, float sigma) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor record = torch::zeros_like(depth);
	torch::Tensor weight_record = torch::zeros_like(depth);
	torch::Tensor weight_warp = torch::zeros_like(depth);
	torch::Tensor out = torch::zeros_like(image);
	torch::TensorOptions lock_option = torch::TensorOptions().dtype(torch::kInt32).device(depth.device());
	torch::Tensor lock = torch::zeros_like(depth, lock_option);
	TORCH_CHECK(lock.dtype() == torch::kInt32);
	float* image_data = image.data<float>();
	float* sample_data = sample.data<float>();
	float* depth_data = depth.data<float>();
	float* record_data = record.data<float>();
	float* weight_record_data = weight_record.data<float>();
	float* weight_warp_data = weight_warp.data<float>();
	float* out_data = out.data<float>();
	int* lock_data = lock.data<int>();

	dim3 threads(1, 16, 16);
	dim3 blocks(batch_size, height / 16 + 1, width / 16 + 1);

	warp_forward_record_record_cuda_kernel << <blocks, threads >> > (
		image_data, sample_data, depth_data, record_data, weight_record_data, lock_data,
		channels, height, width);

	warp_forward_record_warp_cuda_kernel << <blocks, threads >> > (
		image_data, sample_data, depth_data, record_data, weight_warp_data, out_data,
		channels, height, width, sigma);

#ifdef _DEBUG
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//#else
	//	cudaDeviceSynchronize();
#endif

	return { out, record, weight_record, weight_warp };

}


std::vector<torch::Tensor> warp_forward_record(torch::Tensor image, torch::Tensor sample, torch::Tensor depth, float sigma) {

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

	return warp_forward_record_cuda(image, sample, depth, sigma);

}




__global__ void warp_backward_cuda_kernel(float* image, float* sample, float* grad, float* out, int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (row >= height || col >= width) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * 0.5f * (sample[idx_u] + 1.0f);
	float v = (height - 1) * 0.5f * (sample[idx_v] + 1.0f);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu >= 0 && iv >= 0 && iu < (width - 1) && iv < (height - 1)) {
		float uu = u - iu;
		float vv = v - iv;

		float dw11du = -(1.0f - vv) * (width - 1) * 0.5f;
		float dw12du = (1.0f - vv) * (width - 1) * 0.5f;
		float dw21du = -vv * (width - 1) * 0.5;
		float dw22du = vv * (width - 1) * 0.5;

		float dw11dv = -(1.0f - uu) * (height - 1) * 0.5f;
		float dw12dv = -uu * (height - 1) * 0.5f;
		float dw21dv = (1.0f - uu) * (height - 1) * 0.5f;
		float dw22dv = uu * (height - 1) * 0.5f;

		float gu = 0.0f;
		float gv = 0.0f;
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





torch::Tensor warp_backward_cuda(torch::Tensor image, torch::Tensor sample, torch::Tensor grad) {

	int batch_size = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	torch::Tensor out = torch::zeros_like(sample);
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
	//#else
	//  cudaDeviceSynchronize();
#endif

	return out;
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
	int batch_num = image.size(0);
	int channels = image.size(1);
	int height = image.size(2);
	int width = image.size(3);

	TORCH_CHECK(sample.size(0) == batch_num);
	TORCH_CHECK(sample.size(1) == 2);
	TORCH_CHECK(sample.size(2) == height);
	TORCH_CHECK(sample.size(3) == width);
	TORCH_CHECK(sample.device() == image.device());
	TORCH_CHECK(grad.size(0) == batch_num);
	TORCH_CHECK(grad.size(1) == channels);
	TORCH_CHECK(grad.size(2) == height);
	TORCH_CHECK(grad.size(3) == width);
	TORCH_CHECK(grad.device() == image.device());

	return warp_backward_cuda(image, sample, grad);

}


__global__ void warp_forward_cuda_kernel(float* image, float* sample, float* out, int channels, int height, int width) {

	int batch_id = blockIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.z * blockDim.z + threadIdx.z;

	if (row >= height || col >= width) { return; }

	size_t idx_u = batch_id * (2 * height * width) + 0 * (height * width) + row * width + col;
	size_t idx_v = batch_id * (2 * height * width) + 1 * (height * width) + row * width + col;

	float u = (width - 1) * 0.5f * (sample[idx_u] + 1.0f);
	float v = (height - 1) * 0.5f * (sample[idx_v] + 1.0f);

	int iu = (int)floor(u);
	int iv = (int)floor(v);

	if (iu >= 0 && iv >= 0 && iu < (width - 1) && iv < (height - 1)) {
		float uu = u - iu;
		float vv = v - iv;
		float w11 = (1.0f - uu) * (1.0f - vv);
		float w12 = uu * (1.0f - vv);
		float w21 = (1.0f - uu) * vv;
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
	//#else
	//  cudaDeviceSynchronize();
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
	TORCH_CHECK(image.device() == sample.device());

	return warp_forward_cuda(image, sample);

}


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

	float gu = 0.0f;
	float gv = 0.0f;
	for (int c = 0; c < channels; c++) {
		size_t idx_center = batch_id * (channels * height * width) + c * (height * width) + row * width + col;
		size_t idx_left = idx_center - 1;
		size_t idx_right = idx_center + 1;
		size_t idx_top = idx_center - width;
		size_t idx_bottom = idx_center + width;
		size_t idx_gd = batch_id * (channels * height * width) + c * (height * width) + iv * width + iu;
		gu += grad[idx_gd] * (image[idx_left] - image[idx_right]) * 0.25 * (width - 1);
		gv += grad[idx_gd] * (image[idx_top] - image[idx_bottom]) * 0.25 * (height - 1);
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

	float sigma = 0.1f;
	std::vector<torch::Tensor> warp_record_output = warp_forward_record(image, sample, depth, sigma);
	torch::Tensor grad_record_sample = warp_backward_record(image, sample, depth, warp_record_output[1], warp_record_output[2],  grad, sigma);

	return 0;
}
#endif


