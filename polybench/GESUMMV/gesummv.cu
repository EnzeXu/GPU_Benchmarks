/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
// #define n 4096

/* Thread block dimensions */
// #define DIM_THREAD_BLOCK_X 32 // 256
#define DIM_THREAD_BLOCK_Y 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp, int n)
{
	int i, j;
	
	for (i = 0; i < n; i++)
	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < n; j++)
		{
			tmp[i] = A[i*n + j] * x[j] + tmp[i];
			y[i] = B[i*n + j] * x[j] + y[i];
		}
		
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}


void init(DATA_TYPE* A, DATA_TYPE* x, int n)
{
  	int i, j;

 	for (i = 0; i < n; i++)
    {
    	x[i] = ((DATA_TYPE) i) / n;
      	
		for (j = 0; j < n; j++) 
		{
			A[i*n + j] = ((DATA_TYPE) i*j) / n;
		}
    }
}


// void compareResults(DATA_TYPE* y, DATA_TYPE* y_outputFromGpu, int n)
// {
// 	int i, fail;
// 	fail = 0;
	
// 	for (i=0; i<(n); i++) 
// 	{
// 		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
// 		{
// 			fail++;
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


__global__ void gesummv_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		int j;
		for(j = 0; j < n; j++)
		{	
			tmp[i] += a[i * n + j] * x[j];
			y[i] += b[i * n + j] * x[j];
		}
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

void gesummvCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, DATA_TYPE* y_outputFromGpu, int n, int dim_thread_block_x)
{
	double t_start, t_end;		

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * n * n);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * n * n);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * n);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * n);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * n);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);

	dim3 block(dim_thread_block_x, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)n) / ((float)block.x) ), 1);


	t_start = rtclock();
	gesummv_kernel<<< grid, block>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu, n);
	cudaThreadSynchronize();
	t_end = rtclock();
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);

	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;


	int dim_thread_block_x = 32;
	int size = 32; //2048; // [MODIFIED CODE]
	int n = size; // [MODIFIED CODE]

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-size") && i + 1 < argc) {
			size = atoi(argv[++i]);
			n = size;
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
	
	A = (DATA_TYPE*)malloc(n*n*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(n*n*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(n*sizeof(DATA_TYPE)); 
	y = (DATA_TYPE*)malloc(n*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(n*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(n*sizeof(DATA_TYPE));

	init(A, x, n);
	
	GPU_argv_init();
	gesummvCuda(A, B, x, y, tmp, y_outputFromGpu, n, dim_thread_block_x);
	
	// t_start = rtclock();
	// gesummv(A, B, x, y, tmp, n);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// compareResults(y, y_outputFromGpu);

	free(A);
	free(B);  
	free(x);  
	free(y);
	free(y_outputFromGpu);
	free(tmp);

	return 0;
}

