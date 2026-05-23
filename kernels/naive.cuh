#include "./naive/multiple_elem.cuh"
#include "./naive/single_cam.cuh"
#include "./naive/single_plane.cuh"
#include "./naive/single_cam_in_place.cuh"

/**
 * Use to select which choice to run.
 */
enum choice{
    /**
     * Process 1! kernel call by computing multiple elements per thread
     */
    MULTI_ELEMS,
    /**
     * Process 1 kernel call per plane
     */
    SINGLE_PLANE,
    /**
     * Process 1 kernel call per camera
     * Compute minimum in host
     */
    SINGLE_CAMERA_CPU,

    /**
     * Process 1 kernel call per camera
     * Compute minimum in place on device (GPU)
     */
    SINGLE_CAMERA_GPU,
};

std::vector<cv::Mat> naive_gpu_sweeping_plane(
    int const ref_idx, 
    std::vector<cam> const cam_vector, 
    choice choice,
    int window = 3
){
    std::vector<cv::Mat> result;
    switch(choice){
        case MULTI_ELEMS:
            result = multi_elem(ref_idx,cam_vector,window);
            break;
        case SINGLE_CAMERA_CPU:
            result = single_cam(ref_idx,cam_vector,window);
            break;
        case SINGLE_CAMERA_GPU:
            result = single_cam_gpu(ref_idx,cam_vector,window);
            break;
        case SINGLE_PLANE:
            result = single_plane(ref_idx,cam_vector,window);
            break;
        default:
            break;
    }
    return result;
}