#include "single_plane.cuh"

#define divup(x,y) (x+y-1/y)

__global__ void single_plane_convol_kernel(
    const int ref_idx,

    const double *K,
    const double *K_inv,

    const double *R,
    const double *R_inv,

    const double *t,
    const double *t_inv,

    const uint8_t *Y_img,

    const int width,
    const int height,
    const int cam_vec_size,

    const uint8_t zIdx,
	const int window,
	float *cost_mat

){

    int x = threadIdx.x + blockIdx.x * blockDim.x,
            y = threadIdx.y + blockIdx.y * blockDim.y,
            cam_idx = threadIdx.z + blockIdx.z * blockDim.z;

    if(x >= width || y >= height || cam_idx >= cam_vec_size || cam_idx == ref_idx){
        return;
    }
    // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
    double z = ZNear * ZFar / (ZNear + (((double)zIdx / (double)ZPlanes) * (ZFar - ZNear)));

    // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
    double X_ref = (K_inv[0+9*ref_idx] * x + K_inv[1+9*ref_idx] * y + K_inv[2+9*ref_idx]) * z;
    double Y_ref = (K_inv[3+9*ref_idx] * x + K_inv[4+9*ref_idx] * y + K_inv[5+9*ref_idx]) * z;
    double Z_ref = (K_inv[6+9*ref_idx] * x + K_inv[7+9*ref_idx] * y + K_inv[8+9*ref_idx]) * z;

    // 3D in ref camera coordinates to 3D world
    double X = R_inv[0+9*ref_idx] * X_ref + R_inv[1+9*ref_idx] * Y_ref + R_inv[2+9*ref_idx] * Z_ref - t_inv[0+3*ref_idx];
    double Y = R_inv[3+9*ref_idx] * X_ref + R_inv[4+9*ref_idx] * Y_ref + R_inv[5+9*ref_idx] * Z_ref - t_inv[1+3*ref_idx];
    double Z = R_inv[6+9*ref_idx] * X_ref + R_inv[7+9*ref_idx] * Y_ref + R_inv[8+9*ref_idx] * Z_ref - t_inv[2+3*ref_idx];

    // 3D world to projected camera 3D coordinates
    double X_proj = R[0+9*cam_idx] * X + R[1+9*cam_idx] * Y + R[2+9*cam_idx] * Z - t[0+3*cam_idx];
    double Y_proj = R[3+9*cam_idx] * X + R[4+9*cam_idx] * Y + R[5+9*cam_idx] * Z - t[1+3*cam_idx];
    double Z_proj = R[6+9*cam_idx] * X + R[7+9*cam_idx] * Y + R[8+9*cam_idx] * Z - t[2+3*cam_idx];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    double x_proj = (K[0+9*cam_idx] * X_proj / Z_proj + K[1+9*cam_idx] * Y_proj / Z_proj + K[2+9*cam_idx]);
    double y_proj = (K[3+9*cam_idx] * X_proj / Z_proj + K[4+9*cam_idx] * Y_proj / Z_proj + K[5+9*cam_idx]);
    double z_proj = Z_proj;
    
    x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
    
    // (ii) calculate cost against reference
    // Calculating cost in a window
    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -window / 2; k <= window / 2; k++)
    {
        for (int l = -window / 2; l <= window / 2; l++)
        {
            if (x + l < 0 || x + l >= width)
                continue;
            if (y + k < 0 || y + k >= height)
                continue;
            if (x_proj + l < 0 || x_proj + l >= width)
                continue;
            if (y_proj + k < 0 || y_proj + k >= height)
                continue;

            int idx_r = (y+k)*width + (x+l) + ref_idx*width*height,
                idx_c = (y_proj+k)*width + (x_proj+l) + cam_idx*width*height;
            
            float ref = Y_img[idx_r],
                curr = Y_img[idx_c];

            // Y
            cost += fabsf(ref-curr);
            cc += 1.0f;
        }
    }
    cost_mat[x + width*(y+height*cam_idx)] = cost/cc;

}

__global__ void single_plane_min_kernel(
    const int ref_idx,
    const int width,
    const int height,
    const int cam_vec_size,
    float* cost_mat
){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
            y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x >= width || y >= height){
        return;
    }

    int min = 255.f;
    
    for(int k = 0; k < cam_vec_size; k++){
        if(ref_idx == k)
            continue;
        
        min = fminf(cost_mat[x + width*(y+height*k)],min);
    }
    cost_mat[x + width*y] = min;
}


std::vector<cv::Mat> single_plane(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window
){
    std::vector<cv::Mat> result(ZPlanes);

    int width = cam_vector.at(0).width,
        height = cam_vector.at(0).height,
        cam_vec_size = cam_vector.size();

    size_t cost_mat_size = cam_vec_size*width*height*sizeof(float),
        mat3x3 = 9*sizeof(double),
        mat3x1 = 3*sizeof(double),
        Y_img = width*height*sizeof(uint8_t);

    //CPU:
    float* host_cost_mat = (float*) malloc(width*height*ZPlanes*sizeof(float));

    //GPU:
    double *dev_K, 
        *dev_K_inv,
        *dev_R,
        *dev_R_inv,
        *dev_t,
        *dev_t_inv;

    uint8_t *dev_Y_img;

    float* dev_cost_mat; //output

    //threads & blocks for matmult & convolution
    int block_x = divup(width,max_threads_512.x),
        block_y = divup(height,max_threads_512.y),
        block_z = divup(cam_vec_size,max_threads_512.z);
    
    dim3 N_blocks(block_x,block_y,block_z);
    //threads & blocks for min
    int block_x_min = divup(width,max_threads_1024.x),
        block_y_min = divup(height,max_threads_1024.y);
    
    dim3 N_block_min(block_x_min,block_y_min);

    cudaSetDevice(0);

    //init GPU pointers
    CHK(cudaMalloc(&dev_K,cam_vec_size*mat3x3));
    CHK(cudaMalloc(&dev_K_inv,cam_vec_size*mat3x3));
    CHK(cudaMalloc(&dev_R,cam_vec_size*mat3x3));
    CHK(cudaMalloc(&dev_R_inv,cam_vec_size*mat3x3));

    CHK(cudaMalloc(&dev_t,cam_vec_size*mat3x1));
    CHK(cudaMalloc(&dev_t_inv,cam_vec_size*mat3x1));

    CHK(cudaMalloc(&dev_Y_img,cam_vec_size*Y_img));

    //output
    CHK(cudaMalloc(&dev_cost_mat,cost_mat_size));

    //cpy cam params to pointers
    for(int k = 0; k < cam_vec_size; k++){
        cam current = cam_vector.at(k);

        //K
        CHK(cudaMemcpy(dev_K+k*9,&current.p.K[0],mat3x3,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dev_K_inv+k*9,&current.p.K_inv[0],mat3x3,cudaMemcpyHostToDevice));
        //R
        CHK(cudaMemcpy(dev_R+k*9,&current.p.R[0],mat3x3,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dev_R_inv+k*9,&current.p.R_inv[0],mat3x3,cudaMemcpyHostToDevice));
        //t
        CHK(cudaMemcpy(dev_t+k*3,&current.p.t[0],mat3x1,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dev_t_inv+k*3,&current.p.t_inv[0],mat3x1,cudaMemcpyHostToDevice));
        //Y
        CHK(cudaMemcpy(dev_Y_img+k*width*height,(uint8_t*) current.YUV[0].data,Y_img,cudaMemcpyHostToDevice));
    }
    
    //run kernel for each plane
    for(int z = 0; z < ZPlanes; z++){
        single_plane_convol_kernel<<<N_blocks,max_threads_512>>>(
            ref_idx,
            dev_K,dev_K_inv,
            dev_R,dev_R_inv,
            dev_t,dev_t_inv,
            dev_Y_img,

            width,height,
            cam_vec_size,
            z,
	        window,
            dev_cost_mat
        );

        //wait for results
        CHK(cudaDeviceSynchronize());

        single_plane_min_kernel<<<N_block_min,max_threads_1024>>>(
            ref_idx,
            width, height,
            cam_vec_size,
            dev_cost_mat
        );

        //wait for results
        CHK(cudaDeviceSynchronize());

        //store result to host & generate matrix
        float* host_cost_zi = host_cost_mat+(z*width*height);

        CHK(cudaMemcpy(host_cost_zi,dev_cost_mat,width*height*sizeof(float),cudaMemcpyDeviceToHost));


        result.at(z) = cv::Mat(
            height, 
            width, 
            CV_32FC1,
            host_cost_zi
        ).clone();

    }

Error:
    //K
    cudaFree(dev_K);
    cudaFree(dev_K_inv);
    //R
    cudaFree(dev_R);
    cudaFree(dev_R_inv);
    //t
    cudaFree(dev_t);
    cudaFree(dev_t_inv);
    //Y
    cudaFree(dev_Y_img);
    //cost_mat
    cudaFree(dev_cost_mat);

    free(host_cost_mat);

    return result;

}