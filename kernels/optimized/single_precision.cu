#include "single_precision.cuh"

#define divup(x,y) (((x)+(y)-1)/(y))


__global__ void single_cam_single_precision_kernel(

    const double* K,
    const double* K_inv, //ref
    const double* R,
    const double* R_inv, //ref
    const double* t,
    const double* t_inv, //ref

    const uint8_t* ref_Y_img, //ref
    const uint8_t* current_Y_img,

    const int width,
    const int height,

	const int window,
	float *cost_mat
){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,
        zIdx = threadIdx.z + blockIdx.z * blockDim.z;
    
    if(x >= width || y >= height || zIdx >= ZPlanes){
        return;
    }

    // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
    float z = ZNear * ZFar / (ZNear + (((float)zIdx / (float)ZPlanes) * (ZFar - ZNear)));

    // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
    float X_ref = ((float) K_inv[0] * x + (float) K_inv[1] * y + (float) K_inv[2]) * z;
    float Y_ref = ((float) K_inv[3] * x + (float) K_inv[4] * y + (float) K_inv[5]) * z;
    float Z_ref = ((float) K_inv[6] * x + (float) K_inv[7] * y + (float) K_inv[8]) * z;

    // 3D in ref camera coordinates to 3D world
    float X = (float) R_inv[0] * X_ref + (float) R_inv[1] * Y_ref + (float) R_inv[2] * Z_ref - (float) t_inv[0];
    float Y = (float) R_inv[3] * X_ref + (float) R_inv[4] * Y_ref + (float) R_inv[5] * Z_ref - (float) t_inv[1];
    float Z = (float) R_inv[6] * X_ref + (float) R_inv[7] * Y_ref + (float) R_inv[8] * Z_ref - (float) t_inv[2];

    // 3D world to projected camera 3D coordinates
    float X_proj = (float) R[0] * X + (float) R[1] * Y + (float) R[2] * Z - (float) t[0];
    float Y_proj = (float) R[3] * X + (float) R[4] * Y + (float) R[5] * Z - (float) t[1];
    float Z_proj = (float) R[6] * X + (float) R[7] * Y + (float) R[8] * Z - (float) t[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = ((float) K[0] * X_proj / Z_proj + (float) K[1] * Y_proj / Z_proj + (float) K[2]);
    float y_proj = ((float) K[3] * X_proj / Z_proj + (float) K[4] * Y_proj / Z_proj + (float) K[5]);
    float z_proj = Z_proj;
    
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

            int idx_r = (y+k)*width + (x+l),
                idx_c = (y_proj+k)*width + (x_proj+l);
            
            float ref_Y = ref_Y_img[idx_r],
                curr = current_Y_img[idx_c];

            // Y
            cost += fabsf(ref_Y-curr);
            cc += 1.0f;
        }
    }
    cost/=cc;

    float min = cost_mat[(x + width*(y + height*(zIdx)))];

    cost_mat[(x + width*(y + height*(zIdx)))] = fminf(cost,min);
}

std::vector<cv::Mat> single_cam_fp32_gpu(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window
){
    std::vector<cv::Mat> result(ZPlanes);

    cam ref = cam_vector.at(ref_idx);
    int width = ref.width,
        height = ref.height;

    int cam_vec_size = cam_vector.size();

    //CPU
    std::vector<float> host_cost_mat(width*height*ZPlanes,255.f);

    //GPU
    double *K,*R,*t,            //current
        *K_inv,*R_inv,*t_inv;   //ref
    
    float *dev_cost_mat;        //output
    uint8_t *ref_Y_img, *current_Y_img;

    //threads & blocks:

    int block_x = divup(width,max_threads_512.x),
        block_y = divup(height,max_threads_512.y),
        block_z = divup(ZPlanes,max_threads_512.z);
    
    dim3 N_blocks(block_x,block_y,block_z);

    CHK(cudaSetDevice(0));

    //init pointers
    CHK(cudaMalloc(&dev_cost_mat,width*height*ZPlanes*sizeof(float)));

    CHK(cudaMalloc(&K,9*sizeof(double)));
    CHK(cudaMalloc(&K_inv,9*sizeof(double)));
    CHK(cudaMalloc(&R,9*sizeof(double)));
    CHK(cudaMalloc(&R_inv,9*sizeof(double)));
    CHK(cudaMalloc(&t,3*sizeof(double)));
    CHK(cudaMalloc(&t_inv,3*sizeof(double)));
    CHK(cudaMalloc(&ref_Y_img,width*height*sizeof(uint8_t)));
    CHK(cudaMalloc(&current_Y_img,width*height*sizeof(uint8_t)));

    //cpy ref cam
    CHK(cudaMemcpy(K_inv,&ref.p.K_inv[0],9*sizeof(double),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(R_inv,&ref.p.R_inv[0],9*sizeof(double),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(t_inv,&ref.p.t_inv[0],3*sizeof(double),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(ref_Y_img,ref.YUV[0].data,width*height*sizeof(uint8_t),cudaMemcpyHostToDevice));

    //init cost mat:
    cudaMemcpy(dev_cost_mat,&host_cost_mat[0],width*height*ZPlanes*sizeof(float),cudaMemcpyHostToDevice);

    for(int cam_idx = 0; cam_idx < cam_vec_size; cam_idx++){

        //skip ref cam
        if(ref_idx == cam_idx){
            continue;
        }

        cam curr = cam_vector.at(cam_idx);

        //cpy content of current camera
        CHK(cudaMemcpy(K,&curr.p.K[0],9*sizeof(double),cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(R,&curr.p.R[0],9*sizeof(double),cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(t,&curr.p.t[0],3*sizeof(double),cudaMemcpyHostToDevice));


        CHK(cudaMemcpy(current_Y_img,curr.YUV[0].data,width*height*sizeof(uint8_t),cudaMemcpyHostToDevice));
        //run kernel
        single_cam_single_precision_kernel<<<N_blocks,max_threads_512>>>(
            K,K_inv,
            R,R_inv,
            t,t_inv,
            ref_Y_img,
            current_Y_img,
            width,
            height,
            window,
            dev_cost_mat
        );

        CHK(cudaDeviceSynchronize());
    }

    CHK(cudaMemcpy(&host_cost_mat[0],dev_cost_mat,width*height*ZPlanes*sizeof(float),cudaMemcpyDeviceToHost));

    //copy back in cv mat format
    for(int zi = 0; zi < ZPlanes; zi++){

        float* host_cost_zi = &host_cost_mat[zi*width*height];

        result.at(zi) = cv::Mat(
            height, 
            width, 
            CV_32FC1,
            host_cost_zi
        ).clone();
    }

Error:

    //free cuda pointers
    cudaFree(K);
    cudaFree(K_inv);
    cudaFree(R);
    cudaFree(R_inv);
    cudaFree(t);
    cudaFree(t_inv);
    cudaFree(ref_Y_img);
    cudaFree(current_Y_img);

    cudaFree(dev_cost_mat);

    return result;
}