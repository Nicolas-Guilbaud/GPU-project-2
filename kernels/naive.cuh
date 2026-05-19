#include "./naive/multiple_elem.cuh"
#include "./naive/single_cam.cuh"

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
        case SINGLE_CAMERA:
            result = single_cam(ref_idx,cam_vector,window);
            break;
        default:
            break;
    }
    return result;
}