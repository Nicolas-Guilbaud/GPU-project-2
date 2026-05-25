#include "homography.cuh"

#define divup(x,y) (((x)+(y)-1)/(y))

__global__ void single_cam_homography_kernel(

    const double* A,
    const double* B,
    const double* K,

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

    float X_proj = ((float) A[0]*x + (float) A[1]*y + (float) A[2])*z - (float) B[0],
        Y_proj = ((float) A[3]*x + (float) A[4]*y + (float) A[5])*z - (float) B[1],
        Z_proj = ((float) A[6]*x + (float) A[7]*y + (float) A[8])*z - (float) B[2];

    float x_proj = (float) K[0]* X_proj / Z_proj + (float) K[1]*Y_proj/Z_proj + (float) K[2],
        y_proj = (float) K[3] * X_proj / Z_proj + (float) K[4]*Y_proj/Z_proj + (float) K[5];
    
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

std::vector<cv::Mat> single_cam_homography_gpu(
    int ref_idx,
    std::vector<cam> const cam_vector,
    int window
){
    std::vector<cv::Mat> result(ZPlanes);

    cam ref = cam_vector.at(ref_idx);
    int width = ref.width,
        height = ref.height;

    int cam_vec_size = cam_vector.size();

    //CPUrows
    std::vector<float> host_cost_mat(width*height*ZPlanes,255.f);
    //GPU
    double *dev_A,*dev_B, *dev_K;
    
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
    CHK(cudaMalloc(&dev_A,9*sizeof(double)));
    CHK(cudaMalloc(&dev_B,3*sizeof(double)));
    CHK(cudaMalloc(&dev_K,9*sizeof(double)));


    CHK(cudaMalloc(&ref_Y_img,width*height*sizeof(uint8_t)));
    CHK(cudaMalloc(&current_Y_img,width*height*sizeof(uint8_t)));

    //cpy ref cam
    CHK(cudaMemcpy(ref_Y_img,ref.YUV[0].data,width*height*sizeof(uint8_t),cudaMemcpyHostToDevice));

    //init cost mat:
    cudaMemcpy(dev_cost_mat,&host_cost_mat[0],width*height*ZPlanes*sizeof(float),cudaMemcpyHostToDevice);

    for(int cam_idx = 0; cam_idx < cam_vec_size; cam_idx++){

        //skip ref cam
        if(ref_idx == cam_idx){
            continue;
        }

        cam curr = cam_vector.at(cam_idx);

        /* matmult on CPU: */
        cv::Mat K = cv::Mat(3,3,CV_64F,&curr.p.K[0]), 
            K_inv = cv::Mat(3,3,CV_64F,&ref.p.K_inv[0]), 
            R = cv::Mat(3,3,CV_64F,&curr.p.R[0]), 
            R_inv = cv::Mat(3,3,CV_64F,&ref.p.R_inv[0]), 
            t = cv::Mat(3,1,CV_64F,&curr.p.t[0]), 
            t_inv = cv::Mat(3,1,CV_64F,&ref.p.t_inv[0]);

        cv::Mat host_A = R*R_inv*K_inv,
            host_B = (R*t_inv) + t;

        double *a = host_A.ptr<double>(),
            *b = host_B.ptr<double>(),
            *host_K = K.ptr<double>();

        //copy homogeneous matrices
        cudaMemcpy(dev_A,a,9*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B,b,3*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dev_K,host_K,9*sizeof(double),cudaMemcpyHostToDevice);
        
        CHK(cudaMemcpy(current_Y_img,curr.YUV[0].data,width*height*sizeof(uint8_t),cudaMemcpyHostToDevice));
        //run kernel
        single_cam_homography_kernel<<<N_blocks,max_threads_512>>>(
            dev_A,dev_B,dev_K,
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
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_K);
    cudaFree(ref_Y_img);
    cudaFree(current_Y_img);

    cudaFree(dev_cost_mat);

    return result;
}