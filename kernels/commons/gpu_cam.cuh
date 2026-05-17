#pragma once
#include "../../src/cam_params.hpp"
#include "../../src/constants.hpp"

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

/**
 * 
 */
inline void convert_cam_array(std::vector<cam> src,gpu_cam* dest){
    size_t size = src.size();
	for(int i = 0; i < size; i++){
		dest[i] = gpu_cam(src.at(i));
		//uncomment if you want to ensure struct holds correct data
        // #include "./debug.cuh"
		// check_gpu_cam(src.at(i),dest[i]);
	}
}