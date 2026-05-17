#include "./naive.cuh"
#include "./commons/utils.cuh"

/**
 * Kernel that processes the projection of only 1 camera to the ref. 
 * Must be called multiple times.
 */
__global__ void single_cam_proj_kernel(
    gpu_cam const ref, 
	gpu_cam const current,
	const int window,
    const int cam_idx,
	float *cost_mat
){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,
        zIdx = threadIdx.z + blockIdx.z * blockDim.z; //depth
    
    if(x > ref.width || y > ref.height || current.name == ref.name){
        return;
    }

    // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
    double z = ZNear * ZFar / (ZNear + (((double)z / (double)ZPlanes) * (ZFar - ZNear)));
    
    // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
    double X_ref = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * z;
    double Y_ref = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * z;
    double Z_ref = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * z;
    
    // 3D in ref camera coordinates to 3D world
    double X = ref.R_inv[0] * X_ref + ref.R_inv[1] * Y_ref + ref.R_inv[2] * Z_ref - ref.t_inv[0];
    double Y = ref.R_inv[3] * X_ref + ref.R_inv[4] * Y_ref + ref.R_inv[5] * Z_ref - ref.t_inv[1];
    double Z = ref.R_inv[6] * X_ref + ref.R_inv[7] * Y_ref + ref.R_inv[8] * Z_ref - ref.t_inv[2];
    
    // 3D world to projected camera 3D coordinates
    double X_proj = current.R[0] * X + current.R[1] * Y + current.R[2] * Z - current.t[0];
    double Y_proj = current.R[3] * X + current.R[4] * Y + current.R[5] * Z - current.t[1];
    double Z_proj = current.R[6] * X + current.R[7] * Y + current.R[8] * Z - current.t[2];
    
    // Projected camera 3D coordinates to projected camera 2D coordinates
    double x_proj = (current.K[0] * X_proj / Z_proj + current.K[1] * Y_proj / Z_proj + current.K[2]);
    double y_proj = (current.K[3] * X_proj / Z_proj + current.K[4] * Y_proj / Z_proj + current.K[5]);
    double z_proj = Z_proj;
    
    x_proj = x_proj < 0 || x_proj >= current.width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= current.height ? 0 : roundf(y_proj);
    // (ii) calculate cost against reference
    // Calculating cost in a window
    float cost = 0.0f;
    float cc = 0.0f;
    for (int k = -window / 2; k <= window / 2; k++)
    {
        for (int l = -window / 2; l <= window / 2; l++)
        {
            if (x + l < 0 || x + l >= ref.width)
            continue;
            if (y + k < 0 || y + k >= ref.height)
            continue;
            if (x_proj + l < 0 || x_proj + l >= current.width)
            continue;
            if (y_proj + k < 0 || y_proj + k >= current.height)
            continue;
            
            // Y
            cost += fabsf((float) (ref.Y[IDX2(y+k,x+l)]) - (float) (current.Y[IDX2(y_proj+k,x_proj+l)]));
            cc += 1.0f;
        }
    }
    cost_mat[IDX3(x,y,zIdx)] = cost / cc;
}

/**
 * Kernel that processes the projection of only 1 plane for each camera. 
 * Reduction is done on cpu.
 * Must be called multiple times.
 */
__global__ void single_plane_proj_kernel_cpu(
    gpu_cam const ref, 
	gpu_cam const *cam_vector, 
	const int cam_vec_size,
	const int window,
    const int zIdx,
	float *cost_mat
){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,
        cam_idx = threadIdx.z + blockIdx.z * blockDim.z; //cam
		
    gpu_cam current = cam_vector[cam_idx];
    
    if(x > ref.width || y > ref.height || current.name == ref.name){
        return;
    }

    // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
    double z = ZNear * ZFar / (ZNear + (((double)z / (double)ZPlanes) * (ZFar - ZNear)));
    
    // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
    double X_ref = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * z;
    double Y_ref = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * z;
    double Z_ref = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * z;
    
    // 3D in ref camera coordinates to 3D world
    double X = ref.R_inv[0] * X_ref + ref.R_inv[1] * Y_ref + ref.R_inv[2] * Z_ref - ref.t_inv[0];
    double Y = ref.R_inv[3] * X_ref + ref.R_inv[4] * Y_ref + ref.R_inv[5] * Z_ref - ref.t_inv[1];
    double Z = ref.R_inv[6] * X_ref + ref.R_inv[7] * Y_ref + ref.R_inv[8] * Z_ref - ref.t_inv[2];
    
    // 3D world to projected camera 3D coordinates
    double X_proj = current.R[0] * X + current.R[1] * Y + current.R[2] * Z - current.t[0];
    double Y_proj = current.R[3] * X + current.R[4] * Y + current.R[5] * Z - current.t[1];
    double Z_proj = current.R[6] * X + current.R[7] * Y + current.R[8] * Z - current.t[2];
    
    // Projected camera 3D coordinates to projected camera 2D coordinates
    double x_proj = (current.K[0] * X_proj / Z_proj + current.K[1] * Y_proj / Z_proj + current.K[2]);
    double y_proj = (current.K[3] * X_proj / Z_proj + current.K[4] * Y_proj / Z_proj + current.K[5]);
    double z_proj = Z_proj;
    
    x_proj = x_proj < 0 || x_proj >= current.width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= current.height ? 0 : roundf(y_proj);
    // (ii) calculate cost against reference
    // Calculating cost in a window
    float cost = 0.0f;
    float cc = 0.0f;
    for (int k = -window / 2; k <= window / 2; k++)
    {
        for (int l = -window / 2; l <= window / 2; l++)
        {
            if (x + l < 0 || x + l >= ref.width)
            continue;
            if (y + k < 0 || y + k >= ref.height)
            continue;
            if (x_proj + l < 0 || x_proj + l >= current.width)
            continue;
            if (y_proj + k < 0 || y_proj + k >= current.height)
            continue;
            
            // Y
            cost += fabsf((float) (ref.Y[IDX2(y+k,x+l)]) - (float) (current.Y[IDX2(y_proj+k,x_proj+l)]));
            cc += 1.0f;
        }
    }
    cost_mat[IDX3(x,y,cam_idx)] = cost / cc;
}

/**
 * Kernel that processes the projection of only 1 plane for each camera. 
 * Reduction is done on GPU.
 * Must be called multiple times.
 */
__global__ void single_plane_proj_kernel_gpu(
    gpu_cam const ref, 
	gpu_cam const *cam_vector, 
	const int cam_vec_size,
	const int window,
    const int zIdx,
	float *cost_mat
){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,
        cam_idx = threadIdx.z + blockIdx.z * blockDim.z; //cam
		
    gpu_cam current = cam_vector[cam_idx];
    
    if(x > ref.width || y > ref.height || current.name == ref.name){
        return;
    }

    // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
    double z = ZNear * ZFar / (ZNear + (((double)z / (double)ZPlanes) * (ZFar - ZNear)));
    
    // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
    double X_ref = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * z;
    double Y_ref = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * z;
    double Z_ref = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * z;
    
    // 3D in ref camera coordinates to 3D world
    double X = ref.R_inv[0] * X_ref + ref.R_inv[1] * Y_ref + ref.R_inv[2] * Z_ref - ref.t_inv[0];
    double Y = ref.R_inv[3] * X_ref + ref.R_inv[4] * Y_ref + ref.R_inv[5] * Z_ref - ref.t_inv[1];
    double Z = ref.R_inv[6] * X_ref + ref.R_inv[7] * Y_ref + ref.R_inv[8] * Z_ref - ref.t_inv[2];
    
    // 3D world to projected camera 3D coordinates
    double X_proj = current.R[0] * X + current.R[1] * Y + current.R[2] * Z - current.t[0];
    double Y_proj = current.R[3] * X + current.R[4] * Y + current.R[5] * Z - current.t[1];
    double Z_proj = current.R[6] * X + current.R[7] * Y + current.R[8] * Z - current.t[2];
    
    // Projected camera 3D coordinates to projected camera 2D coordinates
    double x_proj = (current.K[0] * X_proj / Z_proj + current.K[1] * Y_proj / Z_proj + current.K[2]);
    double y_proj = (current.K[3] * X_proj / Z_proj + current.K[4] * Y_proj / Z_proj + current.K[5]);
    double z_proj = Z_proj;
    
    x_proj = x_proj < 0 || x_proj >= current.width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= current.height ? 0 : roundf(y_proj);
    // (ii) calculate cost against reference
    // Calculating cost in a window
    float cost = 0.0f;
    float cc = 0.0f;
    for (int k = -window / 2; k <= window / 2; k++)
    {
        for (int l = -window / 2; l <= window / 2; l++)
        {
            if (x + l < 0 || x + l >= ref.width)
            continue;
            if (y + k < 0 || y + k >= ref.height)
            continue;
            if (x_proj + l < 0 || x_proj + l >= current.width)
            continue;
            if (y_proj + k < 0 || y_proj + k >= current.height)
            continue;
            
            // Y
            cost += fabsf((float) (ref.Y[IDX2(y+k,x+l)]) - (float) (current.Y[IDX2(y_proj+k,x_proj+l)]));
            cc += 1.0f;
        }
    }
    cost_mat[IDX3(x,y,cam_idx)] = cost / cc;
    
    if(cam_idx != 0){
        return;
    }

    __syncthreads();

    float min_val = 255.f;
    for(int k = 0; k < cam_vec_size; k++){
        min_val = fminf(min_val,cost_mat[IDX3(x,y,k)]);
    }
    //store min value in cam_idx = 0
    cost_mat[IDX3(x,y,0)] = min_val;
}

/**
 * Kernel that processes the projection of only 1 plane for each camera. 
 * Reduction is done on GPU.
 * Must be called multiple times.
 */
__global__ void muliple_elem_kernel(
    gpu_cam const ref, 
	gpu_cam const *cam_vector, 
	const int cam_vec_size,
	const int window,
	float *cost_mat
){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,
        zIdx = threadIdx.z + blockIdx.z * blockDim.z;
        
    if(x > ref.width || y > ref.height){
        return;
    }

    float min = 255.f;

    for(int cam_idx = 0; cam_idx < cam_vec_size; cam_idx++){
        gpu_cam current = cam_vector[cam_idx];

        // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
        double z = ZNear * ZFar / (ZNear + (((double)z / (double)ZPlanes) * (ZFar - ZNear)));
        
        // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
        double X_ref = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * z;
        double Y_ref = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * z;
        double Z_ref = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * z;
        
        // 3D in ref camera coordinates to 3D world
        double X = ref.R_inv[0] * X_ref + ref.R_inv[1] * Y_ref + ref.R_inv[2] * Z_ref - ref.t_inv[0];
        double Y = ref.R_inv[3] * X_ref + ref.R_inv[4] * Y_ref + ref.R_inv[5] * Z_ref - ref.t_inv[1];
        double Z = ref.R_inv[6] * X_ref + ref.R_inv[7] * Y_ref + ref.R_inv[8] * Z_ref - ref.t_inv[2];
        
        // 3D world to projected camera 3D coordinates
        double X_proj = current.R[0] * X + current.R[1] * Y + current.R[2] * Z - current.t[0];
        double Y_proj = current.R[3] * X + current.R[4] * Y + current.R[5] * Z - current.t[1];
        double Z_proj = current.R[6] * X + current.R[7] * Y + current.R[8] * Z - current.t[2];
        
        // Projected camera 3D coordinates to projected camera 2D coordinates
        double x_proj = (current.K[0] * X_proj / Z_proj + current.K[1] * Y_proj / Z_proj + current.K[2]);
        double y_proj = (current.K[3] * X_proj / Z_proj + current.K[4] * Y_proj / Z_proj + current.K[5]);
        double z_proj = Z_proj;
        
        x_proj = x_proj < 0 || x_proj >= current.width ? 0 : roundf(x_proj);
        y_proj = y_proj < 0 || y_proj >= current.height ? 0 : roundf(y_proj);
        // (ii) calculate cost against reference
        // Calculating cost in a window
        float cost = 0.0f;
        float cc = 0.0f;
        for (int k = -window / 2; k <= window / 2; k++)
        {
            for (int l = -window / 2; l <= window / 2; l++)
            {
                if (x + l < 0 || x + l >= ref.width)
                continue;
                if (y + k < 0 || y + k >= ref.height)
                continue;
                if (x_proj + l < 0 || x_proj + l >= current.width)
                continue;
                if (y_proj + k < 0 || y_proj + k >= current.height)
                continue;
                
                // Y
                cost += fabsf((float) (ref.Y[IDX2(y+k,x+l)]) - (float) (current.Y[IDX2(y_proj+k,x_proj+l)]));
                cc += 1.0f;
            }
        }
        min = fminf(cost/cc,min);

    }
    
    cost_mat[IDX3(x,y,zIdx)] = min;
}



std::vector<cv::Mat> single_cam(
    gpu_cam const &ref, 
    gpu_cam const *cam_vector, 
    int cam_vec_size,
    int window = 3
){

    size_t cost_mat_size = ref.width*ref.height*ZPlanes*sizeof(float);

    //CPU
    float* host_cost_mat = (float*) malloc(cost_mat_size);
    std::vector<cv::Mat> result(ZPlanes);

    for (int i = 0; i < result.size(); ++i){
		result[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}
    
    //GPU
    float* dev_cost_mat;

    //blocks & threads
    int block_x = div_up(ref.width,max_threads.x),
        block_y = div_up(ref.height,max_threads.y),
        block_z = div_up(ZPlanes,max_threads.y);
    
    dim3 N_blocks(block_x,block_y,block_z);

    cudaMalloc(&dev_cost_mat,cost_mat_size);

    for(int i = 0; i < cam_vec_size; i++){
        single_cam_proj_kernel<<<N_blocks,max_threads>>>(
            ref,
            cam_vector[i],
            window,
            i,
            dev_cost_mat
        );
        cudaDeviceSynchronize();
        cudaMemcpy(&dev_cost_mat,&host_cost_mat,cost_mat_size,cudaMemcpyDeviceToHost);
        
        //update minimal values on CPU
        for(int z = 0; z < ZPlanes; z++){
            for(int x = 0; x < ref.width; x++){
                for(int y = 0; y < ref.height; y++){
                    float min = result[z].at<float>(y,x);
                    result[z].at<float>(y,x) = fmin(min,host_cost_mat[IDX3(x,y,z)]);
                }
            }
        }

    }
    
Error:
    cudaFree(&dev_cost_mat);
    free(host_cost_mat);

    return result;
}


std::vector<cv::Mat> single_plane_cpu(
    gpu_cam const &ref, 
    gpu_cam const *cam_vector, 
    int cam_vec_size,
    int window = 3
){

    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < result.size(); ++i){
		result[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

    size_t cost_mat_size = cam_vec_size*ref.width*ref.height*sizeof(float),
        gpu_cam_vec_size = cam_vec_size*sizeof(gpu_cam);

    //CPU
    float* host_cost_mat = (float*) malloc(cost_mat_size);


    //GPU
    gpu_cam* dev_cam_vec;
    float* dev_cost_mat;

    //Threads & blocks
    dim3 N_threads(max_threads.x,max_threads.y,cam_vec_size);

    int block_x = div_up(ref.width,N_threads.x),
        block_y = div_up(ref.height,N_threads.y);

    dim3 N_blocks(block_x,block_y,1);

    cudaMalloc(&dev_cam_vec,gpu_cam_vec_size);
    cudaMalloc(&dev_cost_mat,cost_mat_size);

    cudaMemcpy(&dev_cam_vec,&cam_vector,gpu_cam_vec_size,cudaMemcpyHostToDevice);

    for(int z = 0; z < ZPlanes; z++){
        // run kernel

        single_plane_proj_kernel_cpu<<<N_blocks,N_threads>>>(
            ref, 
            dev_cam_vec, 
            cam_vec_size,
            window,
            z,
            dev_cost_mat
        );

        //wait for kernel results
        cudaDeviceSynchronize();
        cudaMemcpy(&host_cost_mat,&dev_cost_mat,cost_mat_size,cudaMemcpyDeviceToHost);
        
        for(int x = 0; x < ref.width; x++){
            for(int y = 0; y < ref.height; y++){
                float min = 255.f;
                for(int k = 0; k < cam_vec_size; k++){
                    min = fmin(min, host_cost_mat[IDX3(x,y,k)]);
                }

                result[z].at<float>(y,x) = min;
            }
        }
    }

Error:
    free(host_cost_mat);
    cudaFree(dev_cam_vec);
    cudaFree(dev_cost_mat);

    return result;
}

std::vector<cv::Mat> single_plane_gpu(
    gpu_cam const &ref, 
    gpu_cam const *cam_vector, 
    int cam_vec_size,
    int window = 3
){

    std::vector<cv::Mat> result(ZPlanes);

    size_t cost_mat_size_reduced = ref.width*ref.height*sizeof(float),
        cost_mat_size = cost_mat_size_reduced*cam_vec_size,
        gpu_cam_vec_size = cam_vec_size*sizeof(gpu_cam);

    //CPU
    float* host_cost_mat = (float*) malloc(cost_mat_size_reduced);


    //GPU
    gpu_cam* dev_cam_vec;
    float* dev_cost_mat;

    //Threads & blocks
    dim3 N_threads(max_threads.x,max_threads.y,cam_vec_size);

    int block_x = div_up(ref.width,N_threads.x),
        block_y = div_up(ref.height,N_threads.y);

    dim3 N_blocks(block_x,block_y,1);

    cudaMalloc(&dev_cam_vec,gpu_cam_vec_size);
    cudaMalloc(&dev_cost_mat,cost_mat_size);

    cudaMemcpy(&dev_cam_vec,&cam_vector,gpu_cam_vec_size,cudaMemcpyHostToDevice);

    for(int z = 0; z < ZPlanes; z++){
        // run kernel

        single_plane_proj_kernel_gpu<<<N_blocks,N_threads>>>(
            ref, 
            dev_cam_vec, 
            cam_vec_size,
            window,
            z,
            dev_cost_mat
        );

        //wait for kernel results
        cudaDeviceSynchronize();

        //retrieve contiguous data
        cudaMemcpy(&host_cost_mat,&dev_cost_mat,cost_mat_size_reduced,cudaMemcpyDeviceToHost);

        result.at(z) = cv::Mat(
            ref.height, 
            ref.width, 
            CV_32FC1, 
            host_cost_mat+(z*ref.width*ref.height)
        );
    }

Error:
    free(host_cost_mat);
    cudaFree(dev_cam_vec);
    cudaFree(dev_cost_mat);

    return result;
}

std::vector<cv::Mat> multi_elem(
    gpu_cam const &ref, 
    gpu_cam const *cam_vector, 
    int cam_vec_size,
    int window = 3
){
    std::vector<cv::Mat> result(ZPlanes);

    size_t cost_mat_size = cam_vec_size*ref.width*ref.height*sizeof(float),
        gpu_cam_vec_size = cam_vec_size*sizeof(gpu_cam);

    //CPU
    float* host_cost_mat = (float*) malloc(cost_mat_size);


    //GPU
    gpu_cam* dev_cam_vec;
    float* dev_cost_mat;

    //Threads & blocks

    int block_x = div_up(ref.width,max_threads.x),
        block_y = div_up(ref.height,max_threads.y),
        block_z = div_up(ZPlanes,max_threads.z);

    dim3 N_blocks(block_x,block_y,block_z);

    cudaMalloc(&dev_cam_vec,gpu_cam_vec_size);
    cudaMalloc(&dev_cost_mat,cost_mat_size);

    cudaMemcpy(&dev_cam_vec,&cam_vector,gpu_cam_vec_size,cudaMemcpyHostToDevice);

    // run kernel
    muliple_elem_kernel<<<N_blocks,max_threads>>>(
        ref, 
        dev_cam_vec, 
        cam_vec_size,
        window,
        dev_cost_mat
    );

    //wait for kernel results
    cudaDeviceSynchronize();

    //retrieve contiguous data
    cudaMemcpy(&host_cost_mat,&dev_cost_mat,cost_mat_size,cudaMemcpyDeviceToHost);

    for(int z = 0; z < ZPlanes; z++){
        result.at(z) = cv::Mat(
            ref.height, 
            ref.width, 
            CV_32FC1, 
            host_cost_mat+(z*ref.width*ref.height)
        );
    }

Error:
    free(host_cost_mat);
    cudaFree(dev_cam_vec);
    cudaFree(dev_cost_mat);

    return result;
}

std::vector<cv::Mat> naive_gpu_sweeping_plane(
    cam const ref, 
    std::vector<cam> const cam_vector, 
    choice selection,
    int window
){
    //Transform to gpu-compatible struct
    int cam_vec_size = cam_vector.size();
    const gpu_cam gpu_ref = gpu_cam(ref);
    gpu_cam* gpu_cam_vec = (gpu_cam*) malloc(cam_vec_size*sizeof(gpu_cam));
    convert_cam_array(cam_vector,gpu_cam_vec);

    std::vector<cv::Mat> result;
    
    switch (selection)
    {
    case SINGLE_CAMERA:
        result = single_cam(gpu_ref,gpu_cam_vec,cam_vec_size,window);
        break;
    
    case SINGLE_PLANE_CPU:
        result = single_plane_cpu(gpu_ref,gpu_cam_vec,cam_vec_size,window);
        break;

    case SINGLE_PLANE_GPU:
        result = single_plane_gpu(gpu_ref,gpu_cam_vec,cam_vec_size,window);
        break;
    case MULTI_ELEMS:
        result = multi_elem(gpu_ref,gpu_cam_vec,cam_vec_size,window);
        break;
    }

    free(gpu_cam_vec);
    return result;
}