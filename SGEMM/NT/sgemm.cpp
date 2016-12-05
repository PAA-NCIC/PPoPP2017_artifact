// sgemm.cpp : Defines the entry point for the console application.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

CUcontext      hContext = 0;
cublasHandle_t hCublas  = 0;
cublasHandle_t handle;

float assemblySgemm(const char* kernel, float * devRand,  float * devC, float * devA, float * devB, int M,  int N, int K, CUevent hStart, CUevent hStop);
void gflops(int M, int K, int N, float ms);

#define REPEAT_BLOCK 1

#define CUDA_CHECK( fn ) do { \
		CUresult status = (fn); \
		if ( CUDA_SUCCESS != status ) { \
			const char* errstr; \
			cuGetErrorString(status, &errstr); \
			printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
			if (hCublas)  cublasDestroy(hCublas); \
			if (hContext) cuCtxDestroy(hContext); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

#define CUBLAS_CHECK( fn ) do { \
		cublasStatus_t status = (fn); \
		if ( CUBLAS_STATUS_SUCCESS != status ) { \
			printf("Cublas Failure (line %d of file %s):\n\t%s returned %d\n", __LINE__, __FILE__, #fn, status); \
			if (hCublas)  cublasDestroy(hCublas); \
			if (hContext) cuCtxDestroy(hContext); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

int main(int argc, char* argv[])
{
    char deviceName[32];
    int count, ordinal, major, minor;
    CUdevice  hDevice;
    CUevent hStart, hStop;
    //float * devRand, devA, devB, devC, devT, otherDevA, otherDevB;
    float *devA, *devB, *devRand, *devC, *devT, *otherDevA, *otherDevB;


    // Initialize the Driver API and find a device
    CUDA_CHECK( cuInit(0) );
    CUDA_CHECK( cuDeviceGetCount(&count) );
    //====================== search for K20 GPU =================================//
    for (ordinal = 0; ordinal < count; ordinal++)
    {
        CUDA_CHECK( cuDeviceGet(&hDevice, ordinal) );
        CUDA_CHECK( cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice) );
        CUDA_CHECK( cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice) );
        CUDA_CHECK( cuDeviceGetName(deviceName, sizeof(deviceName), hDevice) );
        if (major == 3 && minor == 5)
        {
            printf("\nUsing: Id:%d %s (%d.%d)\n", ordinal, deviceName, major, minor);
            break;
        }
    }
    //============== Initialize host data : A[M][K] B[K][N] C[M][N] ============//
    int M, N, K;
    M = atoi(argv[1]);
    K = atoi(argv[2]);
    N  = atoi(argv[3]); 
    float alpha = 1, beta = 0, ms = 1;
    size_t sizeFloatA = M * K *4;
    size_t sizeFloatB = K * N *4;
    size_t sizeFloatC = M * N *4;
    size_t sizeRand = sizeof(unsigned int);

	float* A = (float*)malloc(sizeFloatA);
	unsigned int * Rand = (unsigned int*)malloc(sizeRand);
	float* B = (float*)malloc(sizeFloatB);
	float* C = (float*)malloc(sizeFloatC);
    float *cublasC = (float*)malloc(sizeFloatC);

    srand(1);
	for(int i = 0; i < M * K; i++) //
	{
        A[i] = 1.0 * rand() /(float) RAND_MAX;
        //A[i] = i + 1.0;
	}
	for(int i = 0; i < N*K; i++) //
	{
        B[i] = 1.0 * rand() / (float)RAND_MAX;
        //B[i] = i + 1.0;
	}

    for (int i = 0; i < M*N; i ++)
    {
        C[i] = 0;
        cublasC[i] = 0;
    }
    //===========================Initialize device data====================== //
	CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );

    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	CUDA_CHECK( cuEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) ); // CU_EVENT_DEFAULT
	CUDA_CHECK( cuEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC) );
    cudaMalloc((void **)&devA, sizeFloatA);
    cudaMalloc((void **)&devB, sizeFloatB);
    cudaMalloc((void **)&devC, sizeFloatC);
    cudaMemcpy(devA, A,sizeFloatA, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B,sizeFloatB, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C,sizeFloatC, cudaMemcpyHostToDevice);

	//================================== Launch our kernel	==============================//
    ms = assemblySgemm("sgemm_nt_128x128", devRand, devC, devA, devB, M, N, K, hStart, hStop);
    gflops(M, K, N, ms);
	// Get back our results from each kernel
    cudaMemcpy(C, devC, sizeFloatC, cudaMemcpyDeviceToHost);

#if 0
    FILE *fp = fopen("c.txt", "w");
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++)
            fprintf(fp, "%f ", C[i*N + j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
#endif
    //=============================== cuBLAS SGEMM code test ==================================//
    cublasCreate(&handle);
    cudaMemcpy(devC, cublasC, sizeFloatC, cudaMemcpyHostToDevice);
    int iterations = 1;
	( cuEventRecord( hStart, NULL ) );
    for(int i = 0; i < iterations; i ++) {
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, devA, K, devB, K, &beta, devC, M);
    }
    ( cuEventRecord( hStop, NULL ) );
    ( cuEventSynchronize( hStop ) );
    ( cuEventElapsedTime( &ms, hStart, hStop ) );
    ms = ms /iterations;
    printf("Matrix Size %d %d %d \t", M, K, N);
    printf("iteration %d times  \n", iterations);
    printf("cuBLAS: ");
	gflops(M, K, N, ms);
	// Get back our results from each kernel
    cudaMemcpy(cublasC, devC, sizeFloatC, cudaMemcpyDeviceToHost);

    //=========================check result compare our result with cuBLAS========================//
#if 0
    fp = fopen("cublasc.txt", "w");
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++)
            fprintf(fp, "%f ", cublasC[i*N + j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
#endif
    
    int index, indexT, error;
    error= 0;
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            index= i*N + j;
            indexT = j*M + i;
            if (C[index] != cublasC[indexT]) {
                printf("%d %d %f, %f \n", i, j, C[index], cublasC[indexT]);
                error = 1;
                break;
            }
        }
        if (error == 1)
            break;
    }
    if (error == 1)
        printf("Error:cuBLAS result is different from Our SGEMM \n\n");
    else
        printf("Passed result check.... \n\n");

	// Cleanup and shutdown of cuda
	cudaFree(devRand);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFree(devT);
	cuEventDestroy(hStart);
	cuEventDestroy(hStop);
	cuCtxDestroy(hContext);
	hContext = 0;
	return 0;
}

// Our kernel wrapper function
float assemblySgemm(const char* kernel, float * devRand, float * devC, float * devA, float * devB, int M, int N, int K, CUevent hStart, CUevent hStop)
{
	int threads, width;
    threads = 256;
    int sizeA, sizeB;
    sizeA = 128;
    sizeB = 128;
    int gridA = M / sizeA + (M %sizeA !=0);
    int gridB = N / sizeB + (N %sizeB !=0);

	// Setup out debug printf output buffer
	float * devD;
	float * D = NULL;
	int  sizeD = 0;

    sizeD = gridA * gridB * threads * sizeof(int)*4;
    D = (float *)malloc(sizeD);

    //CUDA_CHECK( cuMemAlloc(&devD, sizeD) );
    //CUDA_CHECK( cuMemsetD8(devD, 0, sizeD) );

	// Load the cubin
	CUmodule hModule;
	CUDA_CHECK( cuModuleLoad(&hModule, "sgemm_nt_128x128.cubin") );
	// Load the kernel function
	CUfunction hKernel;
	CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, kernel) );

	// Setup the params
	float alpha = 1.0f;
	float beta = 0.0f;
    int lda= M;
    int ldb= N;
    int ldc= N;
    int flags= 0;
    int ldaz, ldbz, ldcz;
    ldaz= ldbz = ldcz = 0;
    int batch_loop = 1;
    int loops = 0;
    //int k = N;
	void* params[] = {&devA, &devB,  &devC, &alpha, &beta, &lda, &ldb, &ldc, &M, &N, &K};
    unsigned int  sharedMemBytes=0;
	float totalTime = 0;
    float ms;
    int iterations = 1;
    int i, j, k;

    //printf("gridA: %d, gridB %d \n", gridA, gridB );
    //printf("threads: %d \n", threads);
    printf("Matrix Size %d %d %d \t", M, K, N);
    printf("iteration %d times  \n", iterations);
    printf("Our SGEMM: ");
    ( cuEventRecord( hStart, NULL ) );
    for(i = 0; i < iterations; i ++) {
        ( cuLaunchKernel(hKernel,1, gridA, gridB,  threads, 1, 1, sharedMemBytes, 0, params, 0) );

    }
    ( cuEventRecord( hStop, NULL ) );
    ( cuEventSynchronize( hStop ) );
    ( cuEventElapsedTime( &ms, hStart, hStop ) );
    totalTime += ms;


	CUDA_CHECK( cuModuleUnload(hModule) );
    //debug
    //CUDA_CHECK( cuMemcpyDtoH(D, devD, sizeD) );
    //CUDA_CHECK( cuMemFree(devD) );
	return totalTime/iterations;
}

void gflops(int M, int K, int N, float ms)
{
	// Standard sgemm flops formula
	printf("Time %f ms, Performance: %.2f gflops\n", ms,  ((double)M * K * N * 2.0 ) / (ms * 1000000.0));
}
