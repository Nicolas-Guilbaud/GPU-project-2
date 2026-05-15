/*
 * This file is used for debug purposes to ensure the conversion
 * from a cam object to a gpu_cam is correct
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include "../src/cam_params.hpp"
#include "../src/constants.hpp"

#define DEBUG(...) fprintf(stderr,__VA_ARGS__)


void print_mat(std::vector<double> &ref, double* obj,std::string title){
    DEBUG("%s: \n",title.c_str());
    for(int i = 0; i < ref.size();i++){
        DEBUG("- (%i): '%f' '%f'\n",i,ref.at(i),obj[i]);
    }
}

void print_cv_data(cv::Mat &ref, uint8_t* obj){
    DEBUG("cols: %d - rows: %d\n",ref.cols,ref.rows);
    
    DEBUG("%s: \n","Diff in Y");

    for(int i = 0; i < ref.rows; i++){
        for(int j = 0; j < ref.cols; j++){
            
            uint8_t ref_val = ref.at<uint8_t>(i,j);
            uint8_t obj_val = obj[i*ref.cols + j];

            int8_t diff = ref_val - obj_val;
            if(diff != 0){
                DEBUG("- (i,j)=(%i,%i): %i\n",i,j,diff);
            }
        }
    }
}

void check_gpu_cam(cam ref, gpu_cam obj){
    
    DEBUG("Names: '%s' '%s'\n",ref.name.c_str(),obj.name);

    print_mat(ref.p.K,obj.K,"K");
    print_mat(ref.p.K_inv,obj.K_inv,"K_inv");
    print_mat(ref.p.R,obj.R,"R");
    print_mat(ref.p.R_inv,obj.R_inv,"R_inv");
    print_mat(ref.p.t,obj.t,"t");
    print_mat(ref.p.t_inv,obj.t_inv,"t_inv");

    DEBUG("width: '%d' '%d'\n",ref.width,obj.width);
    DEBUG("height: '%d' '%d'\n",ref.height,obj.height);
    DEBUG("size: '%d' '%d'\n",ref.size,obj.size);

    print_cv_data(ref.YUV[0],obj.Y);
}