/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>
#include <cuda.h>

#include "../polybenchUtilFuncts.h"

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
// #define NX 4096
// #define NY 4096

/* Thread block dimensions */
// #define DIM_THREAD_BLOCK_X 32 // 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r, int nx, int ny)
{
	int i, j;

  	for (i = 0; i < nx; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < ny; j++)
		{
      			A[i*ny + j] = ((DATA_TYPE) i*j) / nx;
		}
 	}
	
	for (i = 0; i < ny; i++)
	{
    		p[i] = i * M_PI;
	}
}


// void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu, int nx, int ny)
// {
// 	int i,fail;
// 	fail = 0;

// 	// Compare s with s_cuda
// 	for (i=0; i<nx; i++)
// 	{
// 		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
// 		{
// 			fail++;
// 		}
// 	}

// 	for (i=0; i<ny; i++)
// 	{
// 		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
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


//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, int nx, int ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < ny)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < nx; i++)
		{
			s[j] += A[i * ny + j] * r[i];
		}
	}	
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q, int nx, int ny)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < nx)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < ny; j++)
		{
			q[i] += A[i * ny + j] * p[j];
		}
	}
}


void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q, int nx, int ny)
{
	int i,j;
	
  	for (i = 0; i < ny; i++)
	{
		s[i] = 0.0;
	}

    for (i = 0; i < nx; i++)
    {
		q[i] = 0.0;
		for (j = 0; j < ny; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i*ny + j];
	    		q[i] = q[i] + A[i*ny + j] * p[j];
	  	}
	}
}


void bicgCuda(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q,
			DATA_TYPE* s_outputFromGpu, DATA_TYPE* q_outputFromGpu, int nx, int ny, int dim_thread_block_x)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *q_gpu;
	DATA_TYPE *p_gpu;
	DATA_TYPE *r_gpu;
	DATA_TYPE *s_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * nx * ny);
	cudaMalloc((void **)&r_gpu, sizeof(DATA_TYPE) * nx);
	cudaMalloc((void **)&s_gpu, sizeof(DATA_TYPE) * ny);
	cudaMalloc((void **)&p_gpu, sizeof(DATA_TYPE) * ny);
	cudaMalloc((void **)&q_gpu, sizeof(DATA_TYPE) * nx);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * nx * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(r_gpu, r, sizeof(DATA_TYPE) * nx, cudaMemcpyHostToDevice);
	cudaMemcpy(s_gpu, s, sizeof(DATA_TYPE) * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(p_gpu, p, sizeof(DATA_TYPE) * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(q_gpu, q, sizeof(DATA_TYPE) * nx, cudaMemcpyHostToDevice);

	dim3 block(dim_thread_block_x, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)ny) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)nx) / ((float)block.x) )), 1);

	t_start = rtclock();
	bicg_kernel1<<< grid1, block >>>(A_gpu, r_gpu, s_gpu, nx, ny);
	cudaThreadSynchronize();
	bicg_kernel2<<< grid2, block >>>(A_gpu, p_gpu, q_gpu, nx, ny);
	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaMemcpy(s_outputFromGpu, s_gpu, sizeof(DATA_TYPE) * ny, cudaMemcpyDeviceToHost);
	cudaMemcpy(q_outputFromGpu, q_gpu, sizeof(DATA_TYPE) * nx, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(r_gpu);
	cudaFree(s_gpu);
	cudaFree(p_gpu);
	cudaFree(q_gpu);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;
	DATA_TYPE* s_outputFromGpu;
	DATA_TYPE* q_outputFromGpu;

	int dim_thread_block_x = 32;
	int size = 32; //2048; // [MODIFIED CODE]
	int nx = size, ny = size; // [MODIFIED CODE]

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-size") && i + 1 < argc) {
			size = atoi(argv[++i]);
			nx = ny = size;
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
 	
	A = (DATA_TYPE*)malloc(nx*ny*sizeof(DATA_TYPE));
	r = (DATA_TYPE*)malloc(nx*sizeof(DATA_TYPE));
	s = (DATA_TYPE*)malloc(ny*sizeof(DATA_TYPE));
	p = (DATA_TYPE*)malloc(ny*sizeof(DATA_TYPE));
	q = (DATA_TYPE*)malloc(nx*sizeof(DATA_TYPE));
	s_outputFromGpu = (DATA_TYPE*)malloc(ny*sizeof(DATA_TYPE));
	q_outputFromGpu = (DATA_TYPE*)malloc(nx*sizeof(DATA_TYPE));

	init_array(A, p, r, nx, ny);

	GPU_argv_init();

	bicgCuda(A, r, s, p, q, s_outputFromGpu, q_outputFromGpu, nx, ny, dim_thread_block_x); // [MODIFIED CODE]

	// t_start = rtclock();
	// bicg_cpu(A, r, s, p, q, nx, ny);
	// t_end = rtclock();

	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	// compareResults(s, s_outputFromGpu, q, q_outputFromGpu);

	free(A);
	free(r);
	free(s);
	free(p);
	free(q);
	free(s_outputFromGpu);
	free(q_outputFromGpu);

  	return 0;
}

