#include "main.cuh"
#include "../src/constants.hpp"
#include "./debug.cuh"
#include <cstdio>

// Those functions are an example on how to call cuda functions from the main.cpp

int div_up(int x, int y){
    return (x + y - 1)/y;
}


__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

__global__ void naive_gpu(
	gpu_cam const ref, 
	gpu_cam const *cam_vector, 
	const int cam_vec_size,
	const int window, 
	float *cost_mat){

	int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,

		camIdx = threadIdx.z, // [0 -> 255] -> dans 1 block, on a 256 threads en Z
		zIdx = blockIdx.z; // pour chaque block, la camera est la même -> on a 3 blocs
		
		if(x > ref.width || y > ref.height){
			return;
		}
		
		gpu_cam current = cam_vector[camIdx];
		
		if(current.name != ref.name){

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			double z = ZNear * ZFar / (ZNear + (((double)zIdx / (double)ZPlanes) * (ZFar - ZNear)));
			
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
					// U
					// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
					// V
					// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
					cc += 1.0f;
				}
			}
			cost_mat[IDX4(x,y,zIdx,camIdx)] = cost / cc;
		}
		
		
		if(camIdx != 0){
			return;
		}
		
		//thread 0: wait for other threads to finish the projection
		__syncthreads();

		//select minimal cost over camIdx for (x,y,zIdx)
		float min_cost = cost_mat[IDX4(x,y,zIdx,0)];
		for(int k = 0; k < cam_vec_size; k++){
			
			gpu_cam cam_k = cam_vector[k];
			
			//skip ref (cost = 0)
			if(cam_k.name == ref.name){
				continue;
			}
			
			min_cost = fminf(cost_mat[IDX4(x,y,zIdx,k)],min_cost);
		}

		//store minimal cost
		cost_mat[IDX4(x,y,zIdx,0)] = min_cost;
}

std::vector<cv::Mat> naive_sweeping_plane_gpu(
	cam const ref, 
	std::vector<cam> const &cam_vector, 
	int window
){

	//get cam size
	size_t cam_vec_size = cam_vector.size();

	//CPU Side
	std::vector<cv::Mat> host_cost_cube(ZPlanes);
	float* result = (float*) malloc(ZPlanes*ref.width*ref.height*sizeof(float)); //temporary output (could be optimized ?)
	gpu_cam *host_cam = (gpu_cam*) malloc(cam_vec_size*sizeof(gpu_cam));
	//conversion of cameras
	for(int i = 0; i < cam_vec_size; i++){
		host_cam[i] = gpu_cam(cam_vector.at(i));
		//TODO: remove (ensure struct holds correct data) 
		check_gpu_cam(cam_vector.at(i),host_cam[i]);
	}

	//GPU Side
	gpu_cam *dev_cam_vector;
	float *dev_cost_mat; //output (flattened)

	//x = width
	//y = height
	//z = camIdx (blocks) + plane (threads)
	
	dim3 N_threads(max_threads,max_threads,cam_vec_size); //TODO: check si Z supporte 256 threads
	
	int x_blocks = div_up(ref.width,max_threads),
		y_blocks = div_up(ref.height,max_threads),
		z_blocks = ZPlanes;
	
	dim3 N_blocks(x_blocks,y_blocks,z_blocks);

	//init GPU arrays
	CHK(cudaMalloc(&dev_cam_vector,cam_vec_size*sizeof(gpu_cam)));
	CHK(cudaMalloc(&dev_cost_mat,cam_vec_size*ZPlanes*ref.width*ref.height*sizeof(float)));

	DEBUG("%d*%d*%d*%d = %d\n",ref.width,ref.height,ZPlanes,cam_vec_size,cam_vec_size*ZPlanes*ref.width*ref.height);
	//cpy to GPU
	CHK(cudaMemcpy(dev_cam_vector,host_cam,cam_vec_size*sizeof(gpu_cam),cudaMemcpyHostToDevice));

	DEBUG("Blocks: (%d,%d,%d) Threads: (%d,%d,%d)\n",N_blocks.x,N_blocks.y,N_blocks.z,N_threads.x,N_threads.y,N_threads.z);

	naive_gpu<<<N_blocks,N_threads>>>(
		//cameras params
		ref,
		dev_cam_vector,
		cam_vec_size,

		window, //kernel
		dev_cost_mat //result
	);

	//wait for gpu
	CHK(cudaDeviceSynchronize());

	//copy only results of the 1st cam
	CHK(cudaMemcpy(result,dev_cost_mat,ZPlanes*ref.width*ref.height,cudaMemcpyDeviceToHost));
	for(int l = 0; l < ref.width*ref.height*ZPlanes; l++){
		if(result[l] != 0)
			DEBUG("%d : %f \n",l,result[l]);
	}

	for(int z = 0; z < ZPlanes; z++){
		host_cost_cube.at(z) = cv::Mat(ref.height, ref.width, CV_32FC1, result+(z*ref.width*ref.height));
	}

Error:
	cudaFree(&dev_cam_vector);
	cudaFree(&dev_cost_mat);
	free(host_cam);
	free(result);

	return host_cost_cube;
}


void wrap_test_vectorAdd() {
	printf("Vector Add:\n");

	int N = 3;
	int a[] = { 1, 2, 3 };
	int b[] = { 1, 2, 3 };
	int c[] = { 0, 0, 0 };

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);

	dev_test_vecAdd <<<1, N>>> (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < N; ++i) {
		printf("%i + %i = %i\n", a[i], b[i], c[i]);
	}
}