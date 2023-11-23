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

__device__ void PrintChannel(Matrix M, int channel ){
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
    /*
    More complicated version. 
    Read in a 32x32 sized tile; This writes to a 30x30 output space
    when advancing the tile, so will need to account for this 
    1024/30 =~ 35, so we'll have a 35 x 35 grid of blocks 
    we'll need to have some checks in place to make sure we don't read out of bounds

    can we store all filters in mem?
    3*3*3*64 *8= 13824 = 14kb
    32*32*3*8 = 27864 = 28kb
    I think the shared mem size for ampere GPUs is 164 kb, so this should be fine
    Like before, each thread performs a single convulution
    */
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int c_out_channel = blockIdx.z;
    int block_width = 30;
    int block_height = 30;
    int global_row = block_row * block_height+ thread_row;
    int global_col = block_col * block_width + thread_col;
    //if (thread_row == 0 && thread_col == 0 && block_row ==0 && block_col == 0 && c_out_channel == 0) PrintChannel(M, 1);
   

    double* Msub; 
    // like before, because we always are starting at the first channel of the image
    Msub = &M.elements[block_row * M.width + block_col*block_width];
    // if (thread_row == 0 && thread_col == 0 && block_row ==0 && block_col == 0 && c_out_channel == 0){
    //     int t_channel = 1; 
    //     int grid_size = M.width * M.height;
    //     printf("Msub: \n");
    //     for (int x = 0; x<32; x++){
    //         for(int y = 0; y < 32; y++){
    //             printf("%.0f,", Msub[t_channel * grid_size + x*M.width + y]);
    //         }
    //         printf("\n");
    //     }
    // }
    // next read the filters into shared memory:
    // lets try not using the ndim array, and jsut a flat array to make things easier 
    //__shared__ double shared_Fsub[64*3*3*3];
    // this is probably not the best way to do this, bu
    __shared__ double shared_Msub[3*32*32];
    // each thread reads in at a single position, but 3 times along the channel dimension
    for(int c = 0; c < 3 ; c++){
        if (global_row < 1024 && global_col < 1024){ //  only update if in bounds
            shared_Msub[c*(32*32) + thread_row * 32 + thread_col] = Msub[c*(M.width*M.height) + thread_row * M.width + thread_col];
        }
    }
    __syncthreads();

    // if (thread_row == 0 && thread_col == 0 && block_row ==0 && block_col == 0 && c_out_channel == 0){
    //     int t_channel = 1; 
    //     printf("shared_Msub: \n");
    //     for (int x = 0; x<32; x++){
    //         for(int y = 0; y < 32; y++){
    //             printf("%.0f,", shared_Msub[(32*32) + x*32 + y]);
    //         }
    //         printf("\n");
    //     }
    // }

    // now read in the filters. b/c we are loading in all filters, we don't need to worry about a seond sub, or jumping rows
    // paralellize reads over the c_out, c_in, and fw dimensions 
    __shared__ double shared_Fsub[64*3*3*3];
    int flat_thread_index = thread_row * 32 + thread_col;
    for( int i =0; i<2; i++){
        if (i * flat_thread_index < 64*3*3*3){
            shared_Fsub[i*flat_thread_index] = F.weights[i*flat_thread_index];
        }
    }
    __syncthreads();

    // now do the actual convolution part
    // each thread will perform a single convolution
    // we'll need to loop over the channel dimension
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
        O.elements[c_out_channel * (O.width * O.height) + global_row * 1024 + global_col] = sum;
    }
    


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

int main(){
    Matrix h_mat = MakeHostMatrix(3, 1024, 1024, false);
    Matrix h_padded_mat = PadMatrix(h_mat, 1);
    Matrix h_out_mat = MakeHostMatrix(64, 1024, 1024, true);
    Filter h_filt = MakeHostFilter(3, 64, 3, 3);
    
    //PrintChannel(h_padded_mat, 0);

    Matrix d_mat = MakeDeviceMatrix(h_padded_mat, true);
    Filter d_filt = MakeDeviceFilter(h_filt, true);
    Matrix d_out_mat = MakeDeviceMatrix(h_out_mat, false);
    
    // 
    dim3 dimBlock(32,32);
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
    //PrintChannel(h_out_mat, 0);
    return 0 ;
}