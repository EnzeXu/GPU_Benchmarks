/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size */
#define NI 256
#define NJ 256
#define NK 256

/* Thread block dimensions */
// #define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void conv3D(DATA_TYPE* A, DATA_TYPE* B, int ni, int nj, int nk)
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < ni - 1; ++i) // 0
	{
		for (j = 1; j < nj - 1; ++j) // 1
		{
			for (k = 1; k < nk -1; ++k) // 2
			{
				//printf("i:%d\nj:%d\nk:%d\n", i, j, k);
				B[i*(nk * nj) + j*nk + k] = c11 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c13 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c21 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c23 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c31 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c33 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c12 * A[(i + 0)*(nk * nj) + (j - 1)*nk + (k + 0)]  +  c22 * A[(i + 0)*(nk * nj) + (j + 0)*nk + (k + 0)]   
					     +   c32 * A[(i + 0)*(nk * nj) + (j + 1)*nk + (k + 0)]  +  c11 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k + 1)]  
					     +   c13 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k + 1)]  +  c21 * A[(i - 1)*(nk * nj) + (j + 0)*nk + (k + 1)]  
					     +   c23 * A[(i + 1)*(nk * nj) + (j + 0)*nk + (k + 1)]  +  c31 * A[(i - 1)*(nk * nj) + (j + 1)*nk + (k + 1)]  
					     +   c33 * A[(i + 1)*(nk * nj) + (j + 1)*nk + (k + 1)];
			}
		}
	}
}


void init(DATA_TYPE* A, int ni, int nj, int nk)
{
	int i, j, k;

	for (i = 0; i < ni; ++i)
    	{
		for (j = 0; j < nj; ++j)
		{
			for (k = 0; k < nk; ++k)
			{
				A[i*(nk * nj) + j*nk + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


// void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu, int ni, int nj, int nk)
// {
// 	int i, j, k, fail;
// 	fail = 0;
	
// 	// Compare result from cpu and gpu...
// 	for (i = 1; i < ni - 1; ++i) // 0
// 	{
// 		for (j = 1; j < nj - 1; ++j) // 1
// 		{
// 			for (k = 1; k < nk - 1; ++k) // 2
// 			{
// 				if (percentDiff(B[i*(nk * nj) + j*nk + k], B_outputFromGpu[i*(nk * nj) + j*nk + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
// 				{
// 					fail++;
// 				}
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


__global__ void convolution3D_kernel(DATA_TYPE *A, DATA_TYPE *B, int i, int ni, int nj, int nk)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;


	if ((i < (ni-1)) && (j < (nj-1)) &&  (k < (nk-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(nk * nj) + j*nk + k] = c11 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c13 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c21 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c23 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c31 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k - 1)]  +  c33 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k - 1)]
					     +   c12 * A[(i + 0)*(nk * nj) + (j - 1)*nk + (k + 0)]  +  c22 * A[(i + 0)*(nk * nj) + (j + 0)*nk + (k + 0)]   
					     +   c32 * A[(i + 0)*(nk * nj) + (j + 1)*nk + (k + 0)]  +  c11 * A[(i - 1)*(nk * nj) + (j - 1)*nk + (k + 1)]  
					     +   c13 * A[(i + 1)*(nk * nj) + (j - 1)*nk + (k + 1)]  +  c21 * A[(i - 1)*(nk * nj) + (j + 0)*nk + (k + 1)]  
					     +   c23 * A[(i + 1)*(nk * nj) + (j + 0)*nk + (k + 1)]  +  c31 * A[(i - 1)*(nk * nj) + (j + 1)*nk + (k + 1)]  
					     +   c33 * A[(i + 1)*(nk * nj) + (j + 1)*nk + (k + 1)];
	}
}


void convolution3DCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* B_outputFromGpu, int ni, int nj, int nk, int dim_thread_block_x)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * ni * nj * nk);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * ni * nj * nk);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * ni * nj * nk, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * ni * nj * nk, cudaMemcpyHostToDevice);
	
	
	dim3 block(dim_thread_block_x, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)nk) / ((float)block.x) )), (size_t)(ceil( ((float)nj) / ((float)block.y) )));
	
	t_start = rtclock();
	int i;
	for (i = 1; i < ni - 1; ++i) // 0
	{
		convolution3D_kernel<<< grid, block >>>(A_gpu, B_gpu, i, ni, nj, nk);
	}

	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * ni * nj * nk, cudaMemcpyDeviceToHost);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* B_outputFromGpu;

	int dim_thread_block_x = 32;
	int size = 32; //2048; // [MODIFIED CODE]
	int ni = size, nj = size, nk = size; // [MODIFIED CODE]

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-size") && i + 1 < argc) {
			size = atoi(argv[++i]);
			ni = nj = nk = size;
		}
		if (!strcmp(argv[i], "-blockDimX") && i + 1 < argc) {
			dim_thread_block_x = atoi(argv[++i]);
		}
		if (size < dim_thread_block_x || size < DIM_THREAD_BLOCK_Y) {
			fprintf(stderr, "Error: size must be >= dim_thread_block_x=%d and dim_thread_block_y=%d.\n", dim_thread_block_x, DIM_THREAD_BLOCK_Y);
			exit(1);
		}
	}
	printf("size=%d, dim_thread_block_x=%d\n", size, dim_thread_block_x);

	A = (DATA_TYPE*)malloc(ni*nj*nk*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(ni*nj*nk*sizeof(DATA_TYPE));
	B_outputFromGpu = (DATA_TYPE*)malloc(ni*nj*nk*sizeof(DATA_TYPE));
	
	init(A, ni, nj, nk);
	
	GPU_argv_init();

	convolution3DCuda(A, B, B_outputFromGpu, ni, nj, nk, dim_thread_block_x); // [MODIFIED CODE]

	// t_start = rtclock();
	// conv3D(A, B, ni, nj, nk);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// compareResults(B, B_outputFromGpu);

	free(A);
	free(B);
	free(B_outputFromGpu);

	return 0;
}

