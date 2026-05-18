#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

// Macro for error checking
#define CHK(code)                                                    \
    do                                                               \
    {                                                                \
        if ((code) != cudaSuccess)                                   \
        {                                                            \
            fprintf(stderr, "CUDA error: %s %s %i\n",                \
                    cudaGetErrorString((code)), __FILE__, __LINE__); \
            goto Error;                                              \
        }                                                            \
    } while (0)
//computes exactly 512 threads in total
// with 1024 threads/block: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES ?
const dim3 max_threads(32,8,2);

//4D access of a flattened 1D array
#define IDX4(x,y,z,k) (int) ((x) + ref.width * (y + ref.height * (z + ZPlanes * (k))))
#define IDX3(x,y,z) IDX4(x,y,z,0)
//2D access of a flattened 1D array
#define IDX2(row,col) (int) ((col)+ (ref.width * (row)))