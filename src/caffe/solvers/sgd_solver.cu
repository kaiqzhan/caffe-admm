#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void sgd_update_gpu<float>(int, float*, float*, float, float);
template void sgd_update_gpu<double>(int, double*, double*, double, double);

template <typename Dtype>
__global__ void abs_min_filter_kernel(int N, Dtype* a, Dtype min) {
  CUDA_KERNEL_LOOP(i, N) {
    if (abs(a[i]) < min) a[i] = 0;
  }
}
template <typename Dtype>
void abs_min_filter_gpu(int N, Dtype* a, Dtype min) {
  abs_min_filter_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, min);
  CUDA_POST_KERNEL_CHECK;
}
template void abs_min_filter_gpu<float>(int, float*, float);
template void abs_min_filter_gpu<double>(int, double*, double);

template <typename Dtype>
__global__ void set_mask_gpu_kernel(int N, const Dtype* a, Dtype min, Dtype* mask) {
  CUDA_KERNEL_LOOP(i, N) {
    if (abs(a[i]) < min)
      mask[i] = Dtype(0);
    else
      mask[i] = Dtype(1);
  }
}
template <typename Dtype>
void set_mask_gpu(int N, const Dtype* a, Dtype min, Dtype* mask) {
  set_mask_gpu_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, min, mask);
  CUDA_POST_KERNEL_CHECK;
}
template void set_mask_gpu<float>(int, const float*, float, float*);
template void set_mask_gpu<double>(int, const double*, double, double*);

}  // namespace caffe
