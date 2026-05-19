#pragma once
#include "../src/cam_params.hpp"
#include "../src/constants.hpp"
#include "./commons/utils.cuh"

std::vector<cv::Mat> optimized_sweeping_plane(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window = 3
);