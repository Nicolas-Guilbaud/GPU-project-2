#pragma once
#include "../../src/cam_params.hpp"
#include "../../src/constants.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * GPU compatible camera
 */
struct gpu_cam{

    double K[9],
        K_inv[9],
        R[9],
        R_inv[9];
    
    double t[3],
        t_inv[3];
    
    int width,
        height,
        size;

    uint8_t Y[1920*1080]; // only Y from YUV space will be needed here
};