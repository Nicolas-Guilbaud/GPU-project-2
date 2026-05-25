#pragma once
#include "../../src/cam_params.hpp"
#include "../../src/constants.hpp"
#include "../commons/utils.cuh"

std::vector<cv::Mat> single_cam_homography_approx(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window = 3
);