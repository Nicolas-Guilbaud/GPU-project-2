/*  
    2nd Optimization: 
    store the Y images of all the cameras in texture memory.
    They are stored as a whole 3D image with 1 channel: [x][y][cam]
*/

#pragma once
#include "../../src/cam_params.hpp"
#include "../../src/constants.hpp"
#include "../commons/utils.cuh"

std::vector<cv::Mat> texture_sweeping_plane(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window = 3
);