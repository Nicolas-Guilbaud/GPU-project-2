#pragma once
#include <cuda_runtime.h>
#include <cuda.h>


#include "../src/cam_params.hpp"
#include "../src/constants.hpp"

#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();