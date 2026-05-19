/*  
    1st Optimization: 
    store all the parameters of each camera in the constant memory
    (except Y channel of the image, because it does not fit)
*/

#pragma once
#include "../../src/cam_params.hpp"
#include "../../src/constants.hpp"
#include "../commons/utils.cuh"

std::vector<cv::Mat> constant_mem_sweeping_plane(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window = 3
);