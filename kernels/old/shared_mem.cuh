/*  
    3rd Optimization: 
    store both:

    - the minimal cost
    - the Y image of the reference camera
    
    in shared memory
*/
#pragma once
#include "../../src/cam_params.hpp"
#include "../../src/constants.hpp"
#include "../commons/utils.cuh"

std::vector<cv::Mat> shared_sweeping_plane(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window = 3
);