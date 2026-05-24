#pragma once
#include "../../src/cam_params.hpp"
#include "../../src/constants.hpp"
#include "../commons/utils.cuh"

enum convol_type{
    NAIVE,
};

std::vector<cv::Mat> single_cam_fast_convol_gpu(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window = 3
);