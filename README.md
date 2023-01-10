# Parallel Systems

A university project in openMP and CUDA

## Requirements for OpenMP

- [WSL2 for Windows](https://docs.microsoft.com/en-us/windows/wsl/install) or any Linux distribution 
- [GCC](https://gcc.gnu.org/)

You can install GCC
```
sudo apt install gcc
```

Compile openMP code
```
gcc file.c -o file -fopenmp
```

Execute openMP code 
```
./file
```


## Requirements for CUDA

- NVIDIA Graphics Card
- [NVDIA Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)

Compile CUDA code
```
nvcc file.cu -o file
```

Execute CUDA code 
```
./file
```
