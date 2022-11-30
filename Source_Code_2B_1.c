#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Number of rows and columns in the matrix A
#define NI 4096
#define NJ 4096

// Kernel function to be executed on the GPU
__global__ void ConvolutionKernel(double* A, double* B)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Get the current row index
	int j = blockIdx.y * blockDim.y + threadIdx.y; // Get the current column index

	// Check if the current indices are within the bounds of the matrix and return If not
	if (i >= NI - 1 || j >= NJ - 1) 
		return;

	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	// Perform convolution on the current element of the matrix A
	B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
		    + c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
		    + c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
}

// Function to fill the matrix A with random values
void init(double* A)
{
	int i, j;

	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A[i*NJ + j] = (double)rand()/RAND_MAX;
        	}
    	}
}

int main(int argc, char *argv[])
{
	double*	A;
	double*	B;
	double*	d_A;
	double*	d_B;
	struct timeval	cpu_start, cpu_end;
	int customBlockSize;

	// Insert custom block size
	printf("Give block size: ");
	scanf("%d", &customBlockSize);

	// Get the start time
	gettimeofday(&cpu_start, NULL);

	// Allocate memory for matrix A and B on the host
	A = (double*)malloc(NI*NJ*sizeof(double));
	B = (double*)malloc(NI*NJ*sizeof(double));

	// Initialize the matrix
	init(A);

	// Allocate memory on the device for the input and output matrix d_A and d_B
	cudaMalloc((void**)&d_A, NI*NJ*sizeof(double));
	cudaMalloc((void**)&d_B, NI*NJ*sizeof(double));

	// Copy the input data from the host to the device
	cudaMemcpy(d_A, A, NI*NJ*sizeof(double), cudaMemcpyHostToDevice);

	// Launch the kernel on the device
	// Set the block size for the kernel execution
	// Calculate the grid size based on the block size and the size of the matrices
	dim3 blockSize(customBlockSize, customBlockSize);
	dim3 gridSize((NI + blockSize.x - 1) / blockSize.x, (NJ + blockSize.y - 1) / blockSize.y);

	// Execute the kernel on the device
	ConvolutionKernel<<<gridSize, blockSize>>>(d_A, d_B);

	// Wait for device to synchronize
	cudaDeviceSynchronize();

	// Copy the data from the device back to the host
	cudaMemcpy(B, d_B, NI*NJ*sizeof(double), cudaMemcpyDeviceToHost);

	// Get the end time
	gettimeofday(&cpu_end, NULL);

	// Print the elapsed time
	fprintf(stdout, "Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);
	
	// Free memory
	cudaFree(d_A);
	cudaFree(d_B);
	free(A);
	free(B);
	
	return 0;
}

