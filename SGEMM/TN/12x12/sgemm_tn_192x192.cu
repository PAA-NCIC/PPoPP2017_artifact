extern "C"
__global__ void __launch_bounds__(256) sgemm_tn_192x192
(
    unsigned*    param_Rand,
    const float* param_A,
    const float* param_B,
    float*       param_C,
    int          param_lda8,  
    int          param_ldb8,  
    int          param_ldc,
    int          param_m,
    int          param_n,
    int          param_k,
    float        param_alpha,
    float        param_beta,
    int          param_flags,
    int          param_ldaz,
    int          param_ldbz,
    int          param_ldcz,
    float * param_debug
)
{
    __shared__ float share[192* 8 * 4 + 32];

    int tid = threadIdx.x;

    share[tid] = 1;

    param_C[tid] = share[255-tid];
}
