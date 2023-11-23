#include <stdio.h>
#include "timer.h"

struct Matrix {
  int channels;
  int width;
  int height;
  int stride;
  double* elements;
}; 

struct Filter {
    int fh;
    int fw;
    int c;
    int k;
    double* weights;
};

Matrix MakeDeviceMatrix(Matrix M, bool copy){
  // Create a new matrix in device memory.
  Matrix newDeviceMatrix;
  newDeviceMatrix.width = M.width;
  newDeviceMatrix.stride = M.width;
  newDeviceMatrix.height = M.height;
  newDeviceMatrix.channels = M.channels;
  size_t size = M.channels * M.width * M.height * sizeof(double);
  cudaMalloc((void**) &newDeviceMatrix.elements, size);
  if (copy)
    cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
  return newDeviceMatrix;
}

Filter MakeDeviceFilter(Filter F, bool copy){
    // Create a new filter in device memory.
    Filter newDeviceFilter;
    newDeviceFilter.fh = F.fh;
    newDeviceFilter.fw = F.fw;
    newDeviceFilter.c = F.c;
    newDeviceFilter.k = F.k;
    size_t size = F.fh * F.fw * F.c * F.k * sizeof(double);
    cudaMalloc((void**) &newDeviceFilter.weights, size);
    if (copy)
        cudaMemcpy(newDeviceFilter.weights, F.weights, size, cudaMemcpyHostToDevice);
    return newDeviceFilter;
}

Matrix MakeHostMatrix(int channels, int width, int height, bool use_zeros){
  //row major 
  Matrix newHostMatrix;
  newHostMatrix.width = width;
  newHostMatrix.stride = width;
  newHostMatrix.height = height;
  newHostMatrix.channels = channels;
  size_t size = channels * width * height * sizeof(double);
  newHostMatrix.elements = (double*)malloc(size);
  int grid_size = width * height;
  int nrows = height;
  int ncols = width;
  for(int c=0; c < channels; c++){
    for (int x = 0; x < nrows; x++){
        for (int y=0; y < ncols; y++){
            if (use_zeros){
                newHostMatrix.elements[c*grid_size + x * width + y] = 0;
            }else{
                double elem = c * (x + y);
                newHostMatrix.elements[c*grid_size + x * width + y] = elem;
            }
            
        }
    }
  }
  return newHostMatrix;
}

Matrix PadMatrix(Matrix M, int padding){
    Matrix newMatrix;
    newMatrix.width = M.width + 2 * padding;
    newMatrix.stride = M.width + 2 * padding;
    newMatrix.height = M.height + 2 * padding;
    newMatrix.channels = M.channels;
    size_t size = M.channels * newMatrix.width * newMatrix.height * sizeof(double);
    newMatrix.elements = (double*)malloc(size);
    int new_grid_size = newMatrix.width * newMatrix.height;
    int old_grid_size = M.width * M.height;
    for(int c=0; c < M.channels; c++){
        for (int x=0; x < newMatrix.height; x++){
            for (int y=0; y < newMatrix.width; y++){
               /*
               for case 3, 1024, 1024, this gets triggered when x/y = 0 or 
               */
               if(x < padding || x >= M.height+padding || y < padding || y >= M.width+padding){
                   newMatrix.elements[c * new_grid_size + x * newMatrix.width + y] = 0;
               }
               else{
                   int orig_x = x - padding;
                   int orig_y = y - padding;
                //    if (x == 1025){
                //     printf("should not be here");
                //    }
                   double t_elem = M.elements[c * old_grid_size + orig_x * M.width + orig_y];
                   newMatrix.elements[c * new_grid_size + x * newMatrix.width + y] = t_elem;
                   
               }
            }
        }
    }
    return newMatrix;
}

Filter MakeHostFilter(int c_in, int k_out, int fw, int fh){
    //row major
    Filter newHostFilter;
    newHostFilter.c = c_in;
    newHostFilter.k = k_out;
    newHostFilter.fw = fw;
    newHostFilter.fh = fh;
    size_t size = c_in * k_out * fw * fh * sizeof(double);
    newHostFilter.weights = (double*)malloc(size);
    int grid_size = fw * fh;
    for(int k=0; k < k_out; k++){
        for (int c=0; c < c_in; c++){
            for (int i=0; i < fh; i++){
                for (int j=0; j < fw; j++){
                    double elem = (c + k) * (i + j);
                    //k * c_in * grid_size takes you to the correct out filter 
                    // c * grid_size takes you to the right channel 
                    // j *fw takes you to the right row 
                    // i takes you to the right column
                    newHostFilter.weights[k * c_in * grid_size + c * grid_size + i * fw + j] = elem;
                }
            }
        }
    }
    return newHostFilter;
}

// void PrintChannel(Matrix M, int channel ){
//     int grid_size = M.width * M.height;
//     for(int x= 0; x < M.height; x++){
//         for(int y=0; y < M.width; y++){
//             printf("%.0f,", M.elements[grid_size* channel + x * M.width + y]);
//         }
//         printf("\n");
//     }
// }

// void PrintFilter(Filter F, int k ){
//     int grid_size = F.fw * F.fh;
//     for (int c=0; c < F.c; c++){
//         printf("FChannel: %d\n", c);
//         for (int i=0; i < F.fh; i++){
//             for (int j=0; j < F.fw; j++){
//                 printf("%.0f,", F.weights[k * F.c * grid_size + c * grid_size + i * F.fw + j]);
//             }
//             printf("\n");
//         }
//     }
// }

__global__ void NaiveConvKernel(Matrix M, Filter F, Matrix O){
    /*
    keep it simple: each thread is responsible for one element in the output, 
    so we use a single 1024 x 1 thread block, and then a 64 x 1024 grid of blocks.
    
    */

    int row_idx = blockIdx.y; 
    int col_idx = threadIdx.x;
    int c_out_idx = blockIdx.x;
    int elem_per_filter = F.fw * F.fh * F.c;

    //if(c_out_idx == 0 && row_idx == 0 && col_idx == 0) PrintFilter(F, 0)
    
    double* Msub;
    double* Fsub;
    Msub = &M.elements[row_idx * M.width + col_idx];// only need to worry about getting into the first channel 
    Fsub = &F.weights[c_out_idx * elem_per_filter ]; //only need to get to start of right out channel
    
    double sum = 0;
    for (int c = 0; c < 3 ; c++){
        for (int x = 0; x < 3; x++){
            for (int y = 0; y < 3; y++){
                sum += Msub[c * (1026*1026) + x * 1026 + y] * Fsub[c * (3*3) + x * 3 + y];
            }
        }
    }
    // if (c_out_idx == 0 && row_idx == 0 && col_idx == 0){
    //     printf("sum: %f\n", sum);
    // };

    O.elements[c_out_idx * (1024*1024) + row_idx * 1024 + col_idx] = sum;
}



int main(){
    // borrowed timing code from Part A
    Matrix h_mat = MakeHostMatrix(3, 1024, 1024, false);
    Matrix h_padded_mat = PadMatrix(h_mat, 1);
    Matrix h_out_mat = MakeHostMatrix(64, 1024, 1024, true);
    Filter h_filt = MakeHostFilter(3, 64, 3, 3);
    
    //PrintChannel(h_mat, 2);

    Matrix d_mat = MakeDeviceMatrix(h_padded_mat, true);
    Filter d_filt = MakeDeviceFilter(h_filt, true);
    Matrix d_out_mat = MakeDeviceMatrix(h_out_mat, false);
    dim3 dimBlock(1024);
    dim3 dimGrid(64,1024);
    // warmup
    NaiveConvKernel<<<dimGrid, dimBlock>>>(d_mat, d_filt, d_out_mat);
    cudaDeviceSynchronize();

    // run for real 
    initialize_timer();
    start_timer();
    NaiveConvKernel<<<dimGrid, dimBlock>>>(d_mat, d_filt, d_out_mat);
    cudaThreadSynchronize() ;
    stop_timer();
    double time = elapsed_time();
    cudaMemcpy(h_out_mat.elements, d_out_mat.elements, 64 * 1024 * 1024 * sizeof(double), cudaMemcpyDeviceToHost);
    double checksum = 0;
    for(int i = 0; i < 64 * 1024 * 1024; i++){
        checksum += h_out_mat.elements[i];
    }
    printf("%.1f, %.3f\n", checksum, time*1000);
    //PrintChannel(h_out_mat, 0);
    return 0 ;
}