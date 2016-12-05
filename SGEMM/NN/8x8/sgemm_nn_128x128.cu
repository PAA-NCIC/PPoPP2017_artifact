extern "C"
__global__ void __launch_bounds__(256) sgemm_nn_128x128
(
 const float* param_A,
 const float* param_B,
 float*       param_C,
 float        param_alpha,
 float        param_beta,
 int          param_lda,
 int          param_ldb8,  
 int          param_ldc,
 int          param_m,
 int          param_n,
 int          param_k
 ) {
  __shared__ float share[128 * 8 * 4 + 32];

  int tid = threadIdx.x;

  share[tid] = 1;

  __syncthreads();

  param_C[tid] = share[255 - tid];
}
