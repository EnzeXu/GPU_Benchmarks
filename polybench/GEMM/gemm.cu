/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
// #define NI 512
// #define NJ 512
// #define NK 512

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int ni, int nj, int nk)
{
	int i,j,k;
	
	for (i = 0; i < ni; i++)
	{
    	for (j = 0; j < nj; j++)
    	{
			C[i*nj + j] *= BETA;
	
			for (k = 0; k < nk; ++k)
			{
	  			C[i*nj + j] += ALPHA * A[i*nk + k] * B[k*nj + j];
			}
      	}
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int ni, int nj, int nk)
{
	int i, j;

  	for (i = 0; i < ni; i++)
	{
    	for (j = 0; j < nk; j++)
		{
      		A[i*nk + j] = ((DATA_TYPE) i*j) / ni;
		}
	}

  	for (i = 0; i < nk; i++)
	{
    	for (j = 0; j < nj; j++)
		{
      		B[i*nj + j] = ((DATA_TYPE) i*j + 1) / nj;
		}
	}

  	for (i = 0; i < ni; i++)
	{
    	for (j = 0; j < nj; j++)
		{
      		C[i*nj + j] = ((DATA_TYPE) i*j + 2) / nj;
		}
	}
}


// void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu, int ni, int nj, int nk)
// {
// 	int i, j, fail;
// 	fail = 0;
	
// 	// Compare C1 and C2
// 	for (i=0; i < ni; i++) 
// 	{
// 		for (j=0; j < nj; j++) 
// 		{
// 			if (percentDiff(C[i*nj + j], C_outputFromGpu[i*nj + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
// 			{
// 				fail++;
// 			}
// 		}
// 	}
	
// 	// Print results
// 	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
// }


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c, int ni, int nj, int nk)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < ni) && (j < nj))
	{	
		c[i * nj + j] *= BETA;
		int k;
		for(k=0; k < nk; k++)
		{
			c[i * nj + j] += ALPHA * a[i * nk + k] * b[k * nj +j];
		}
	}
}


void gemmCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu, int ni, int nj, int nk)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * ni * nk);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * nk * nj);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * ni * nj);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * ni * nk, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * nk * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)ni)/ ((float)block.x) )),(size_t)(ceil( ((float)nj)/ ((float)block.y) )));

	t_start = rtclock();

	gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu, ni, nj, nk);
	cudaThreadSynchronize();

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyDeviceToHost);    
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}
	

int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C;  
	DATA_TYPE* C_outputFromGpu;
	
	int size = 128; //2048; // [MODIFIED CODE]
	int ni = size, nj = size, nk = size; // [MODIFIED CODE]

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-size") && i + 1 < argc) {
			size = atoi(argv[++i]);
			if (size < DIM_THREAD_BLOCK_X || size < DIM_THREAD_BLOCK_Y) {
				fprintf(stderr, "Error: size must be >= %d and %d.\n", DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
				exit(1);
			}
			ni = nj = nk = size;
		}
	}

	A = (DATA_TYPE*)malloc(ni*nk*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(nk*nj*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(ni*nj*sizeof(DATA_TYPE)); 
	C_outputFromGpu = (DATA_TYPE*)malloc(ni*nj*sizeof(DATA_TYPE)); 

	init(A, B, C, ni, nj, nk);
	
	GPU_argv_init();
	
	gemmCuda(A, B, C, C_outputFromGpu, ni, nj, nk);

	// t_start = rtclock();	
	// gemm(A, B, C, ni, nj, nk);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// compareResults(C, C_outputFromGpu);

	free(A);
	free(B);  
	free(C);  
	free(C_outputFromGpu); 

    	return 0;
}

