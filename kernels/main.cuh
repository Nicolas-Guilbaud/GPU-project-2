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

int div_up(int x, int y){
    return (x + y - 1)/y;
}

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();
std::vector<cv::Mat> naive_sweeping_plane_gpu(cam const ref, std::vector<cam> const &cam_vector, int window = 3);