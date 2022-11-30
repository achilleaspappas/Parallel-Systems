#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Number of rows and columns in the matrix
#define M 1024
#define N 1024

#define FLOAT_N 3214212.01

// Kernel function to initialize the data array with values
__global__ void init_arrays(double* data)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= 1 && i <= M && j >= 1 && j <= N) {
		data[i*(N+1) + j] = ((double) i*j) / M;
	}
}

// Kernel function to calculate the covariance of the input data matrix 
__global__ void covariance(double* data, double* symmat, double* mean)
{
	int	i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int j1,j2;

	if (i >= 1 && i <= N && j >= 1 && j <= M) {
		if(threadIdx.x == 0) {
			mean[j] = 0.0;
			for (i = 1; i < (N+1); i++) {
				mean[j] += data[i*(M+1) + j];
			}
			mean[j] /= FLOAT_N;
		}
		data[i*(M+1) + j] -= mean[j];
	}

	if(threadIdx.x == 0 && threadIdx.y == 0) {
		j1 = blockIdx.x + 1;
		j2 = blockIdx.y + 1;
		symmat[j1*(M+1) + j2] = 0.0;
		for (i = 1; i < N+1; i++) {
			symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
		}
		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
	}
}

int main(int argc, char *argv[])
{
	double *data;
	double *symmat;
	double *mean;
	double *d_data;
	double *d_symmat;
	double *d_mean;
	struct timeval cpu_start, cpu_end;
	int customBlockSize;

	// Insert custom block size
	printf("Give block size: ");
	scanf("%d", &customBlockSize);

	// Get the start time
	gettimeofday(&cpu_start, NULL);

	// Allocate memory on the host
	data = (double*)malloc((M+1)*(N+1)*sizeof(double));
	symmat = (double*)malloc((M+1)*(M+1)*sizeof(double));
	mean = (double*)malloc((M+1)*sizeof(double));

	// Allocate memory on the device
	cudaMalloc(&d_data, (M+1)*(N+1)*sizeof(double));
	cudaMalloc(&d_symmat, (M+1)*(M+1)*sizeof(double));
	cudaMalloc(&d_mean, (M+1)*sizeof(double));

	// Set the dimensions for the kernel launch
    dim3 blockSize(customBlockSize, customBlockSize);
	dim3 gridSize(1,1);

	// Launch the kernel to initialize the input data
	init_arrays<<<gridSize,blockSize>>>(d_data);

	// Copy the initialized data from the device to the host
	cudaMemcpy(data, d_data, (M+1)*(N+1)*sizeof(double), cudaMemcpyDeviceToHost);

	// Launch the kernel to compute the covariance matrix and mean vector
	covariance<<<gridSize,blockSize>>>(d_data, d_symmat, d_mean);

	// Wait for device to synchronize
	cudaDeviceSynchronize();

	// Copy the computed covariance matrix and mean vector from the device to the host
	cudaMemcpy(symmat, d_symmat, (M+1)*(M+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(mean, d_mean, (M+1)*sizeof(double), cudaMemcpyDeviceToHost);

	// Get the end time and calculate elapsed time
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	// Free memory
	cudaFree(d_data);
	cudaFree(d_symmat);
	cudaFree(d_mean);
	free(data);
	free(symmat);
	free(mean);

  	return 0;
}
