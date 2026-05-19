#include "constant_mem.cuh"

#define divup(x,y) (x+y-1/y)

#define CAM_VEC_SIZE 4

__constant__ double K[9*CAM_VEC_SIZE],K_inv[9*CAM_VEC_SIZE],
                    R[9*CAM_VEC_SIZE],R_inv[9*CAM_VEC_SIZE],
                    t[3*CAM_VEC_SIZE],t_inv[3*CAM_VEC_SIZE];

__global__ void optimized_kernel(
    const int ref_idx,

    const uint8_t *Y_img,

    const int width,
    const int height,

	const int cam_vec_size,
	const int window,
	float *cost_mat
){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,
        zIdx = threadIdx.z + blockIdx.z * blockDim.z;
    
    if(x >= width || y >= height || zIdx >= ZPlanes){
        return;
    }

    float min = 255.f;

    for(int cam_idx = 0; cam_idx < cam_vec_size; cam_idx++){
        if(ref_idx == cam_idx) 
            continue;

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
        cost /= cc;
        min = fminf(cost,min);

        cost_mat[(x + width*(y + height*(zIdx)))] = min;
    }
}

std::vector<cv::Mat> constant_mem_sweeping_plane(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window
){

    std::vector<cv::Mat> result(ZPlanes);

    int width = cam_vector.at(0).width,
        height = cam_vector.at(0).height,
        cam_vec_size = cam_vector.size();
    
    size_t cost_mat_size = ZPlanes*width*height*sizeof(float),
        mat3x3 = 9*sizeof(double),
        mat3x1 = 3*sizeof(double),
        Y_img = width*height*sizeof(uint8_t);

    //CPU
    float* host_cost_mat = (float*) malloc(cost_mat_size);

    //GPU
    double *dev_K, 
        *dev_K_inv,
        *dev_R,
        *dev_R_inv,
        *dev_t,
        *dev_t_inv;

    uint8_t *dev_Y_img;

    float* dev_cost_mat; //output

    //Threads & blocks

    int block_x = divup(width,max_threads_512.x),
        block_y = divup(height,max_threads_512.y),
        block_z = divup(ZPlanes,max_threads_512.z);

    dim3 N_blocks(block_x,block_y,block_z);

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

    //cpy values to pointers
    for(int k = 0; k < cam_vec_size; k++){
        cam current = cam_vector.at(k);

        //K
        CHK(cudaMemcpyToSymbol(K,&current.p.K[0],mat3x3,k*mat3x3,cudaMemcpyHostToDevice));
        CHK(cudaMemcpyToSymbol(K_inv,&current.p.K_inv[0],mat3x3,k*mat3x3,cudaMemcpyHostToDevice));
        //R
        CHK(cudaMemcpyToSymbol(R,&current.p.R[0],mat3x3,k*mat3x3,cudaMemcpyHostToDevice));
        CHK(cudaMemcpyToSymbol(R_inv,&current.p.R_inv[0],mat3x3,k*mat3x3,cudaMemcpyHostToDevice));
        //t
        CHK(cudaMemcpyToSymbol(t,&current.p.t[0],mat3x1,k*mat3x1,cudaMemcpyHostToDevice));
        CHK(cudaMemcpyToSymbol(t_inv,&current.p.t_inv[0],mat3x1,k*mat3x1,cudaMemcpyHostToDevice));
        //Y
        cv::Mat Y = current.YUV[0];
        if(!Y.isContinuous()) goto Error;
        CHK(cudaMemcpy(dev_Y_img+k*width*height,(uint8_t*) Y.data,Y_img,cudaMemcpyHostToDevice));
    }

    // run kernel
    optimized_kernel<<<N_blocks,max_threads_512>>>(
        ref_idx,        // ref
        dev_Y_img,      // Y
        width,          // img size
        height,
        cam_vec_size,   // vec size
        window,         // window
        dev_cost_mat    // output
    );

    //wait for kernel results
    CHK(cudaDeviceSynchronize());

    //retrieve contiguous data
    CHK(cudaMemcpy(host_cost_mat,dev_cost_mat,cost_mat_size,cudaMemcpyDeviceToHost));

    for(int zi = 0; zi < ZPlanes; zi++){

        float* host_cost_zi = host_cost_mat+(zi*width*height);

        result.at(zi) = cv::Mat(
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