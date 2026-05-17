/*
	Naive implementation of the sweeping plane algorithm on a GPU. 
*/
#include "./commons/gpu_cam.cuh"

/**
 * Use to select which choice to run.
 */
enum choice{
    /**
     * Process 1 kernel with multiple elements
     */
    MULTI_ELEMS,
    /**
     * Process 1 kernel per plane & reduce in CPU
     */
    SINGLE_PLANE_CPU,
    /**
     * Process 1 kernel per plane & reduce in GPU
     */
    SINGLE_PLANE_GPU,
    /**
     * Process 1 kernel per camera
     */
    SINGLE_CAMERA,
};

std::vector<cv::Mat> naive_gpu_sweeping_plane(
    cam const ref, 
    std::vector<cam> const cam_vector, 
    choice choice,
    int window = 3
);