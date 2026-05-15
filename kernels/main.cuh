#pragma once
#include <cuda_runtime.h>
#include <cuda.h>


#include "../src/cam_params.hpp"
#include "../src/constants.hpp"

#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

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

const int max_threads = 256;

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();
std::vector<cv::Mat> naive_sweeping_plane_gpu(cam const ref, std::vector<cam> const &cam_vector, int window = 3);

//4D access of a flattened 1D array
#define IDX4(x,y,z,k) (int) ((x) + ref.width * (y + ref.height * (z + ZPlanes * (k))))
//2D access of a flattened 1D array
#define IDX2(row,col) (int) ((col)+ (ref.width * (row)))

/**
 * GPU compatible camera
 */
struct gpu_cam{

    const char* name;

    double *K,
        *K_inv,
        *R,
        *R_inv,
        *t,
        *t_inv;
    
    int width,
        height,
        size;

    uint8_t *Y; // only Y from YUV space will be needed here

    gpu_cam(cam host) : 
        name(host.name.c_str()),

        K(&host.p.K[0]), 
        K_inv(&host.p.K_inv[0]),
        R(&host.p.R[0]), 
        R_inv(&host.p.R_inv[0]),
        t(&host.p.t[0]), 
        t_inv(&host.p.t_inv[0]),

        width(host.width),
        height(host.height),
        size(host.size),

        Y((uint8_t*)(host.YUV[0].data))
    {};
};