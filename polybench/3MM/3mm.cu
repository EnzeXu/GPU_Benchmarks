/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
// # define NI 512
// # define NJ 512
// # define NK 512
// # define NL 512
// # define NM 512

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, int ni, int nj, int nk, int nl, int nm) // [MODIFIED CODE] // void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
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
			B[i*nj + j] = ((DATA_TYPE) i*(j+1)) / nj;
		}
	}
  
	for (i = 0; i < nj; i++)
	{
		for (j = 0; j < nm; j++)
		{
			C[i*nm + j] = ((DATA_TYPE) i*(j+3)) / nl;
		}
	}
  
	for (i = 0; i < nm; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i*nl + j] = ((DATA_TYPE) i*(j+2)) / nk;
		}
	}
}


// void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
// {
// 	int i,j,fail;
// 	fail = 0;

// 	for (i=0; i < ni; i++)
// 	{
// 		for (j=0; j < nl; j++)
// 		{
// 			if (percentDiff(G[i*nl + j], G_outputFromGpu[i*nl + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
}

	
__global__ void mm3_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E, int ni, int nj, int nk, int nl, int nm)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < ni) && (j < nj))
	{
		int k;
		for(k=0; k < nk; k++)
		{
			E[i * nj + j] += A[i * nk + k] * B[k * nj + j];
		}
	}
}

	
__global__ void mm3_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F, int ni, int nj, int nk, int nl, int nm)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < nj) && (j < nl))
	{
		int k;
		for(k=0; k < nm; k++)
		{
			F[i * nl + j] += C[i * nm + k] * D[k * nl +j];
		}
	}
}

	
__global__ void mm3_kernel3(DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G, int ni, int nj, int nk, int nl, int nm)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < ni) && (j < nl))
	{
		int k;
		for(k=0; k < nj; k++)
		{
			G[i * nl + j] += E[i * nj + k] * F[k * nl + j];
		}
	}
}


void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G, int ni, int nj, int nk, int nl, int nm)
{
	int i,j,k;
	
	/* E := A*B */
	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nj; j++)
		{
			E[i*nj + j] = 0;
			for (k = 0; k < nk; ++k)
			{
				E[i*nj + j] += A[i*nk + k] * B[k*nj + j];
			}
		}
	}
		
	/* F := C*D */
	for (i = 0; i < nj; i++)
	{
		for (j = 0; j < nl; j++)
		{
			F[i*nl + j] = 0;
			for (k = 0; k < nm; ++k)
			{
				F[i*nl + j] += C[i*nm + k] * D[k*nl + j];
			}
		}
	}

  	/* G := E*F */
	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nl; j++)
		{
			G[i*nl + j] = 0;
			for (k = 0; k < nj; ++k)
			{
				G[i*nl + j] += E[i*nj + k] * F[k*nl + j];
			}
		}
	}
}


void mm3Cuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, 
		DATA_TYPE* G, DATA_TYPE* G_outputFromGpu, int ni, int nj, int nk, int nl, int nm)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * ni * nk);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * nk * nj);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * nj * nm);
	cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * nm * nl);
	cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * ni * nj);
	cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * nj * nl);
	cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * ni * nl);

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * ni * nk, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * nk * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * nj * nm, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * nm * nl, cudaMemcpyHostToDevice);
	cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * nj * nl, cudaMemcpyHostToDevice);
	cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * ni * nl, cudaMemcpyHostToDevice);	
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)nj) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)ni/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid2((size_t)(ceil( ((float)nl) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)nj/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid3((size_t)(ceil( ((float)nl) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)ni/ ((float)DIM_THREAD_BLOCK_Y) )));

	t_start = rtclock();
	mm3_kernel1<<<grid1,block>>>(A_gpu, B_gpu, E_gpu, ni, nj, nk, nl, nm);
	cudaThreadSynchronize();
	mm3_kernel2<<<grid2,block>>>(C_gpu, D_gpu, F_gpu, ni, nj, nk, nl, nm);
	cudaThreadSynchronize();
	mm3_kernel3<<<grid3,block>>>(E_gpu, F_gpu, G_gpu, ni, nj, nk, nl, nm);
	cudaThreadSynchronize();
	t_end = rtclock();
	cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * ni * nl, cudaMemcpyDeviceToHost);

	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);
	cudaFree(F_gpu);
	cudaFree(G_gpu);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;
	DATA_TYPE* G_outputFromGpu;

	int size = 32; //2048; // [MODIFIED CODE]
	int ni = size, nj = size, nk = size, nl = size, nm = size; // [MODIFIED CODE]

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-size") && i + 1 < argc) {
			size = atoi(argv[++i]);
			if (size < DIM_THREAD_BLOCK_X || size < DIM_THREAD_BLOCK_Y) {
				fprintf(stderr, "Error: size must be >= %d and %d.\n", DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
				exit(1);
			}
			ni = nj = nk = nl = nm = size;
		}
	}

	A = (DATA_TYPE*)malloc(ni*nk*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(nk*nj*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(nj*nm*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(nm*nl*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(ni*nj*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(nj*nl*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(ni*nl*sizeof(DATA_TYPE));
	G_outputFromGpu = (DATA_TYPE*)malloc(ni*nl*sizeof(DATA_TYPE));

	init_array(A, B, C, D, ni, nj, nk, nl, nm);

	GPU_argv_init();

	mm3Cuda(A, B, C, D, E, F, G, G_outputFromGpu, ni, nj, nk, nl, nm);

	// t_start = rtclock();

	// mm3_cpu(A, B, C, D, E, F, G, ni, nj, nk, nl, nm);
	
	// t_end = rtclock();

	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	// compareResults(G, G_outputFromGpu);

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(G_outputFromGpu);

	return 0;
}

