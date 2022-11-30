#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Number of rows and columns in the matrix A
#define NX 4096
#define NY 4096

#ifndef M_PI
#define M_PI 3.14159
#endif

// This kernel initializes the arrays x and A.
// x is initialized to a sequence of values from 0 to NX-1 multiplied by pi.
// A is initialized to a matrix with values that are calculated based on the indices i and j.
__global__ void init_array_kernel(double *x, double *A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        x[i] = i * M_PI;
        A[i*NY + j] = ((double) i*(j)) / NX;
    }
}

// This kernel calculates the product of A and x, and stores the result in y.
// It uses a temporary array tmp to store intermediate results.
__global__ void trans_norm_vector_kernel(double* A, double* x, double* y, double* tmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        tmp[i] = 0;
        tmp[i] = tmp[i] + A[i*NY + j] * x[j];
        y[j] = y[j] + A[i*NY + j] * tmp[i];
    }
}

int main(int argc, char *argv[])
{
    double *host_y;
    double *A, *x, *y, *tmp;
    double *d_A, *d_x, *d_y, *d_tmp;
    struct timeval cpu_start, cpu_end;
    int customBlockSize;

	// Insert custom block size
	printf("Give block size: ");
	scanf("%d", &customBlockSize);

    // Allocate memory on the host
    host_y = (double*)malloc(NY*sizeof(double));

    // Get the start time
    gettimeofday(&cpu_start, NULL);

    // Allocate memory on the device
    cudaMalloc((void**)&A, NX*NY*sizeof(double));
    cudaMalloc((void**)&x, NY*sizeof(double));
    cudaMalloc((void**)&y, NY*sizeof(double));
    cudaMalloc((void**)&tmp, NX*sizeof(double));

    // Set the dimensions for the kernel launch
    dim3 threadsPerBlock(customBlockSize, customBlockSize);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the init_array_kernel to initialize x and A
    init_array_kernel<<<numBlocks, threadsPerBlock>>>(x, A);

    // Wait for device to synchronize
	cudaDeviceSynchronize();

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, NX*NY*sizeof(double));
    cudaMalloc((void**)&d_x, NY*sizeof(double));
    cudaMalloc((void**)&d_y, NY*sizeof(double));
    cudaMalloc((void**)&d_tmp, NX*sizeof(double));

    // Copy the data from the device to the device
    cudaMemcpy(d_A, A, NX*NY*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_x, x, NY*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_y, y, NY*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_tmp, tmp, NX*sizeof(double), cudaMemcpyDeviceToDevice); 
    
    // Launch the trans_norm_vector_kernel to calculate the product of A and x
    trans_norm_vector_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_x, d_y, d_tmp);

    // Wait for device to synchronize
	cudaDeviceSynchronize();

    // Copy the result array y from device to host
    cudaMemcpy(host_y, d_y, NY*sizeof(double), cudaMemcpyDeviceToHost);

    // Get the end time and calculate elapsed time
    gettimeofday(&cpu_end, NULL);
    fprintf(stdout, "Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_tmp);
    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
    cudaFree(tmp);
    free(host_y);

    return 0;
}
