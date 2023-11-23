#include <stdio.h>

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

void PrintChannel(Matrix M, int channel ){
    int grid_size = M.width * M.height;
    for(int x= 0; x < M.height; x++){
        for(int y=0; y < M.width; y++){
            printf("%.0f,", M.elements[grid_size* channel + x * M.width + y]);
        }
        printf("\n");
    }
}

void PrintFilter(Filter F, int k ){
    int grid_size = F.fw * F.fh;
    for (int c=0; c < F.c; c++){
        printf("FChannel: %d\n", c);
        for (int i=0; i < F.fh; i++){
            for (int j=0; j < F.fw; j++){
                printf("%.0f,", F.weights[k * F.c * grid_size + c * grid_size + i * F.fw + j]);
            }
            printf("\n");
        }
    }
}

__global__ void TiledConvKernel(Matrix M, Filter F, Matrix O){
    
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int c_out_channel = blockIdx.z;
    int block_width = 30;
    int block_height = 30;
    int global_row = block_row * block_height+ thread_row;
    int global_col = block_col * block_width + thread_col;
    /*
    problem: given that the block width is 30, and the thread block length is 32, 
    we are getting collisions , at least when writing but likelly else
    // need to do 30 by 30, not 32 by 32 
    */
   
    double* Msub; 
    // like before, because we always are starting at the first channel of the image
    Msub = &M.elements[block_row * (block_width * M.width) + block_col*block_width];
    
    __shared__ double shared_Msub[3*32*32];
    // each thread reads in at a single position, but 3 times along the channel dimension
    
    for (int c = 0; c < 3; c++){
        for( int os =0; os < 2; os++){
            int total_thread_idx = thread_row *30 + thread_col + os * (30*30);
            int row_in_Msub = total_thread_idx / 32;
            int col_in_Msub = total_thread_idx % 32;
            if (total_thread_idx < 32*32){
                shared_Msub[c*(32*32) + total_thread_idx] = Msub[c*(M.width*M.height) + row_in_Msub * M.width + col_in_Msub];
            }
        }
    }
    __syncthreads();
    __shared__ double shared_Fsub[64*3*3*3];
    
    for( int i =0; i<2; i++){
        int flat_thread_index = thread_row * 30 + thread_col  + i*(30*30);
        if (flat_thread_index < 64*3*3*3){
            
            shared_Fsub[flat_thread_index] = F.weights[flat_thread_index];
        }
    }
    __syncthreads();

    // if (thread_row == 0 && thread_col == 0 && block_row ==0 && block_col == 0 && c_out_channel == 0){
    //     //PrintChannel(M,1);
    //     for (int c = 0; c < 3; c++){
    //         printf("channel: %d\n", c);
    //         for (int x = 0; x < 32; x++){
    //             for (int y = 0; y < 32 ; y++){
    //                 printf("%.0f,", shared_Msub[c*(32*32) + x * 32 + y]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    

    double sum = 0;
    for( int c = 0; c < 3 ;c++){
        for (int x = 0; x < 3; x++){
            for (int y = 0; y < 3 ; y++){
                //c_out_channel * (3*3*3) + c * (3*3) + x * 3 +y
                //         nfilter          inchannel   row   col      
                if (global_row < 1024 && global_col < 1024){ //  only update if in bounds
                    double weight = shared_Fsub[c_out_channel * (3*3*3) + c * (3*3) + x * 3 +y];
                    //c*(32*32) + (thread_row + x) * 32 + (thread_col + y)
                    //c*(32*32) right channel 
                    //(thread_row + x) - thread row takes us to the right row, and then x takes us to the right row in the filter operation
                    //(thread_col + y) - thread col takes us to the right col, and then y takes us to the right col in the filter operation
                    double elem = shared_Msub[c*(32*32) + (thread_row + x) * 32 + (thread_col + y)];
                    sum += weight * elem;
                }
            }
        }
    }
    
    //write to the output
    if (global_row < 1024 && global_col < 1024){
        int idx = c_out_channel * (O.width * O.height) + global_row * 1024 + global_col;
        if (idx==30){
            //printf("writing to 30 : %f\n", sum);
            //printf("thread_row: %d, thread_col: %d, block_row: %d, block_col: %d, c_out_channel: %d\n", thread_row, thread_col, block_row, block_col, c_out_channel);
            //printf("global_col: %d, global_row: %d\n", global_col, global_row);
        }
        O.elements[c_out_channel * (O.width * O.height) + global_row * 1024 + global_col] = sum;
    }
    
    //if (thread_row == 0 && thread_col == 0 && block_row ==0 && block_col == 1 && c_out_channel == 0) printf("sum: %f\n", sum);


    // if (thread_row == 0 && thread_col == 0 && block_row ==0 && block_col == 0 && c_out_channel == 0){
    // printf("shared_Fsub: \n");
    // for(int c = 0; c<3; c++){
    //     printf("channel: %d\n", c);
    //     for(int x=0; x<3; x++){
    //         for(int y=0;  y<3; y++){
    //             //printf("%d,", c*9 + x*3 + y);
    //             printf("%.0f,", shared_Fsub[c*9 + x*3 + y]);
    //         }
    //         printf("\n");
    //     }
    // }
    // printf("F: \n");
    // for(int c = 0; c<3; c++){
    //     printf("channel: %d\n", c);
    //     for(int x=0; x<3; x++){
    //         for(int y=0;  y<3; y++){
    //             printf("%.0f,", F.weights[c*9 + x*3 + y]);
    //         }
    //         printf("\n");
    //     }
    // }

    // }
    
    
}

int main(int argc, char *argv[]){

    Matrix h_mat = MakeHostMatrix(3, 1024, 1024, false);
    Matrix h_padded_mat = PadMatrix(h_mat, 1);
    Matrix h_out_mat = MakeHostMatrix(64, 1024, 1024, true);
    Filter h_filt = MakeHostFilter(3, 64, 3, 3);
    
    //PrintChannel(h_padded_mat, 1);

    Matrix d_mat = MakeDeviceMatrix(h_padded_mat, true);
    Filter d_filt = MakeDeviceFilter(h_filt, true);
    Matrix d_out_mat = MakeDeviceMatrix(h_out_mat, false);
    
    // 
    dim3 dimBlock(30,30);
    //1024/30 = 34.13 ~ 35
    dim3 dimGrid(35,35,64);
    TiledConvKernel<<<dimGrid, dimBlock>>>(d_mat, d_filt, d_out_mat);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_mat.elements, d_out_mat.elements, 64 * 1024 * 1024 * sizeof(double), cudaMemcpyDeviceToHost);
    double checksum = 0;
    for(int i = 0; i < 64 * 1024 * 1024; i++){
        checksum += h_out_mat.elements[i];
    }
    printf("checksum: %f\n", checksum);
    //int cout_channel = std::atoi(argv[1]);
    //PrintChannel(h_out_mat, cout_channel );
    return 0 ;
}