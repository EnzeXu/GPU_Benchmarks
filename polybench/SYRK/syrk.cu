/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "../polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
// #define N 1024
// #define M 1024

/* Thread block dimensions */
// #define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for alpha and beta (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_arrays(DATA_TYPE* A, DATA_TYPE* C, int n, int m)
{
	int i, j;
	
	for (i = 0; i < n; i++)
    	{
		for (j = 0; j < m; j++)
		{
			A[i*m + j] = ((DATA_TYPE) i*j) / n;
		}
		
		for (j = 0; j < n; j++)
		{
			C[i*m + j] = ((DATA_TYPE) i*j + 2) / n;
		}
	}
}


void syrk(DATA_TYPE* A, DATA_TYPE* C, int n, int m)
{
	int i, j, k;
	
	/*  C := alpha*A*A' + beta*C */
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			C[i*m + j] *= beta;
		}
	}
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < m; k++)
			{
				C[i*n + j] += alpha * A[i*m + k] * A[j*m + k];
			}
		}
	}
}


// void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu, int n, int m)
// {
// 	int i,j,fail;
// 	fail = 0;

// 	// Compare C with D
// 	for (i=0; i<n; i++)
// 	{
// 		for (j=0; j<m; j++)
// 		{
// 			if (percentDiff(C[i*m + j], C_outputFromGpu[i*m + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
// 			{
// 				fail++;
// 			}
// 		}
// 	}
	
// 	// print results
// 	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
// }


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
	
	return;
}


__global__ void syrk_kernel(DATA_TYPE ALPHA, DATA_TYPE BETA, DATA_TYPE *a, DATA_TYPE *c, int n, int m)
{
	/*  C := alpha*A*A' + beta*C */
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < n) && (j < n))
	{
		c[i * n + j] *= beta;
		int k;		
		for(k=0; k< m; k++)
		{
			c[i * n + j] += alpha * a[i * m + k] * a[j * m + k];
		}
	}
}


void syrkCuda(DATA_TYPE* A, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu, int n, int m, int dim_thread_block_x)
{
	double t_start, t_end;

	DATA_TYPE* A_gpu;
	DATA_TYPE* C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * n * m);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * n * n);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
	
	dim3 block(dim_thread_block_x, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil(((float)n) / ((float)dim_thread_block_x))), (size_t)ceil(((float)n) / ((float)DIM_THREAD_BLOCK_Y)));
	t_start = rtclock();
	syrk_kernel<<<grid,block>>>(alpha, beta, A_gpu,C_gpu, n, m);
	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(C_gpu);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* C;
	DATA_TYPE* C_outputFromGpu;

	int dim_thread_block_x = 32;
	int size = 32; //2048; // [MODIFIED CODE]
	int n = size, m = size; // [MODIFIED CODE]

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-size") && i + 1 < argc) {
			size = atoi(argv[++i]);
			n = m = size;
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

	A = (DATA_TYPE*)malloc(n*m*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(n*m*sizeof(DATA_TYPE));
	C_outputFromGpu = (DATA_TYPE*)malloc(n*m*sizeof(DATA_TYPE));

	init_arrays(A, C, n, m);
	
	GPU_argv_init();	
	syrkCuda(A, C, C_outputFromGpu, n, m, dim_thread_block_x);

	// t_start = rtclock();
	// syrk(A, C, n, m);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	// compareResults(C, C_outputFromGpu);

	free(A);
	free(C);
	free(C_outputFromGpu);

	return 0;
}

