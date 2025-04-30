/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
// #define NX 4096
// #define NY 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32 // 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *x, DATA_TYPE *A, int nx, int ny)
{
	int i, j;

	for (i = 0; i < nx; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < ny; j++)
		{
			A[i*ny + j] = ((DATA_TYPE) i*(j)) / nx;
		}
	}
}


// void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu, int nx, int ny)
// {
// 	int i, fail;
// 	fail = 0;

// 	for (i=0; i<ny; i++)
// 	{
// 		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
// 		{
// 			fail++;
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
}


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp, int nx, int ny)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nx)
	{
		int j;
		for(j=0; j < ny; j++)
		{
			tmp[i] += A[i * ny + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp, int nx, int ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < ny)
	{
		int i;
		for(i=0; i < nx; i++)
		{
			y[j] += A[i * ny + j] * tmp[i];
		}
	}
}


void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, int nx, int ny)
{
	int i,j;
	
	for (i= 0; i < ny; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < nx; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < ny; j++)
		{
			tmp[i] = tmp[i] + A[i*ny + j] * x[j];
		}
		
      	for (j = 0; j < ny; j++)
		{
			y[j] = y[j] + A[i*ny + j] * tmp[i];
		}
    }
}


void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, DATA_TYPE* y_outputFromGpu, int nx, int ny)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * nx * ny);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * ny);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * ny);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * nx);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * nx * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * nx, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)nx) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)ny) / ((float)block.x) )), 1);

	t_start = rtclock();
	atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu, nx, ny);
	cudaThreadSynchronize();
	atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu, nx, ny);
	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * nx, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;

	int size = 32; //2048; // [MODIFIED CODE]
	int nx = size, ny = size; // [MODIFIED CODE]

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-size") && i + 1 < argc) {
			size = atoi(argv[++i]);
			if (size < DIM_THREAD_BLOCK_X || size < DIM_THREAD_BLOCK_Y) {
				fprintf(stderr, "Error: size must be >= %d and %d.\n", DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
				exit(1);
			}
			nx = ny = size;
		}
	}

	A = (DATA_TYPE*)malloc(nx*ny*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(ny*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc(ny*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(ny*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(nx*sizeof(DATA_TYPE));

	init_array(x, A, nx, ny);

	GPU_argv_init();
	ataxGpu(A, x, y, tmp, y_outputFromGpu, nx, ny);
	
	// t_start = rtclock();
	// atax_cpu(A, x, y, tmp, nx, ny);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	// compareResults(y, y_outputFromGpu);

	free(A);
	free(x);
	free(y);
	free(y_outputFromGpu);
	free(tmp);

  	return 0;
}

