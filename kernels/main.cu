#include "main.cuh"
#include "../src/constants.hpp"
#include <cstdio>

// Those functions are an example on how to call cuda functions from the main.cpp

__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

__global__ void naive_gpu(
	cam const ref, 
	cam const *cam_vector, 
	const int cam_vec_size,
	const int window, 
	cv::Mat *cost_mat){

	int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y,

		zIdx = threadIdx.z, // [0 -> 255] -> dans 1 block, on a 256 threads en Z
		camIdx = blockIdx.z; // pour chaque block, la camera est la même -> on a 3 blocs
		
		if(x > ref.width || y > ref.height){
			return;
		}

		//initialize cost_mat vector
		if(x == 0 && y == 0){
			cost_mat[camIdx*ZPlanes+zIdx] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
		}

		__syncthreads();
		
		cam current = cam_vector[camIdx];
		
		if(current.name != ref.name){

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			double z = ZNear * ZFar / (ZNear + (((double)zIdx / (double)ZPlanes) * (ZFar - ZNear)));
			
			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			double X_ref = (ref.p.K_inv[0] * x + ref.p.K_inv[1] * y + ref.p.K_inv[2]) * z;
			double Y_ref = (ref.p.K_inv[3] * x + ref.p.K_inv[4] * y + ref.p.K_inv[5]) * z;
			double Z_ref = (ref.p.K_inv[6] * x + ref.p.K_inv[7] * y + ref.p.K_inv[8]) * z;
			
			// 3D in ref camera coordinates to 3D world
			double X = ref.p.R_inv[0] * X_ref + ref.p.R_inv[1] * Y_ref + ref.p.R_inv[2] * Z_ref - ref.p.t_inv[0];
			double Y = ref.p.R_inv[3] * X_ref + ref.p.R_inv[4] * Y_ref + ref.p.R_inv[5] * Z_ref - ref.p.t_inv[1];
			double Z = ref.p.R_inv[6] * X_ref + ref.p.R_inv[7] * Y_ref + ref.p.R_inv[8] * Z_ref - ref.p.t_inv[2];
			
			// 3D world to projected camera 3D coordinates
			double X_proj = current.p.R[0] * X + current.p.R[1] * Y + current.p.R[2] * Z - current.p.t[0];
			double Y_proj = current.p.R[3] * X + current.p.R[4] * Y + current.p.R[5] * Z - current.p.t[1];
			double Z_proj = current.p.R[6] * X + current.p.R[7] * Y + current.p.R[8] * Z - current.p.t[2];
			
			// Projected camera 3D coordinates to projected camera 2D coordinates
			double x_proj = (current.p.K[0] * X_proj / Z_proj + current.p.K[1] * Y_proj / Z_proj + current.p.K[2]);
			double y_proj = (current.p.K[3] * X_proj / Z_proj + current.p.K[4] * Y_proj / Z_proj + current.p.K[5]);
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
					cost += fabs(ref.YUV[0].at<uint8_t>(y + k, x + l) - current.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
					// U
					// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
					// V
					// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
					cc += 1.0f;
				}
			}
			cv::Mat cost_mat_current = cost_mat[camIdx*ZPlanes+zIdx];
			cost_mat_current.at<float>(x,y) = cost / cc;
		}
		
		
		if(camIdx != 0){
			return;
		}
		
		//thread 0: wait for other threads to finish the projection
		__syncthreads();

		//object where minimal value will be stored (cam = 0)
		cv::Mat cost_mat_write = cost_mat[zIdx];

		float min_cost = cost_mat_write.at<float>(x,y);
		//select minimal cost over camIdx for (x,y,zIdx)
		for(int k = 0; k < cam_vec_size; k++){
			
			cam cam_k = cam_vector[k];
			
			//skip ref (cost = 0)
			if(cam_k.name == ref.name){
				continue;
			}
			
			cv::Mat cost_mat_current = cost_mat[k*ZPlanes+zIdx];
			
			min_cost = fminf(cost_mat_current.at<float>(x,y),min_cost);
		}

		//store minimal cost
		cost_mat_write.at<float>(x,y) = min_cost;
}

std::vector<cv::Mat> naive_sweeping_plane_gpu(
	cam const ref, 
	std::vector<cam> const &cam_vector, 
	int window = 3
){
	/*
	
	1) on choisit les params de la grid
	2) on init tout
	3) on launch le kernel
	4) ???
	5) Profit
	*/

	dim3 threadSize = dim3(),
		blockSize = dim3();

	//CPU Side
	std::vector<cv::Mat> host_cost_cube(ZPlanes);

	//GPU Side
	cam *dev_cam_vector;
	cv::Mat *dev_cost_mat; //output

	//get GPU size
	size_t cam_vec_size = cam_vector.size();

	//x = width
	//y = height
	//z = camIdx (blocks) + plane (threads)
	
	dim3 threads(max_threads,max_threads,256); //TODO: check si Z supporte 256 threads
	
	int x_blocks = div_up(ref.width,max_threads),
		y_blocks = div_up(ref.height,max_threads),
		z_blocks = cam_vec_size;
	
	dim3 blocks(x_blocks,y_blocks,z_blocks);

	//init GPU arrays
	CHK(cudaMalloc(&dev_cam_vector,cam_vec_size*sizeof(cam)));
	CHK(cudaMalloc(&dev_cost_mat,cam_vec_size*ZPlanes*sizeof(cv::Mat)));

	//TODO: check si simplifiable
	for(int i = 0; i < cam_vec_size; i++){
		CHK(cudaMemcpy(&dev_cam_vector+i,&cam_vector[i],sizeof(cam),cudaMemcpyHostToDevice));
	}

	naive_gpu<<<blocks,threads>>>(
		//cameras params
		ref,
		dev_cam_vector,
		cam_vec_size,

		window, //kernel
		dev_cost_mat //result
	);

	//wait for gpu
	CHK(cudaDeviceSynchronize());

	//copy only results on the 1st cam
	CHK(cudaMemcpy(&host_cost_cube[0],dev_cost_mat,ZPlanes*sizeof(cv::Mat),cudaMemcpyDeviceToHost));


Error:
	cudaFree(&dev_cam_vector);
	cudaFree(&dev_cost_mat);

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