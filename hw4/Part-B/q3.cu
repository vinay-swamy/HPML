
///
/// vecadd.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-03
/// Last Modified: 2011-03-03 DVN
///
/// Add two Vectors A and B in C on GPU using
/// a kernel defined according to vecAddKernel.h
/// Students must not modify this file. The GTA
/// will grade your submission using an unmodified
/// copy of this file.
/// 

// Includes
#include <stdio.h>
#include "timer.h"
#include "vecaddKernel.h"

// Defines
// #define GridWidth 60
// #define BlockWidth 128

// Variables for host and device vectors.

float* d_A; 
float* d_B; 
float* d_C; 

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    int K; // number of values per thread
    int GridWidth; //Grid size
    int BlockWidth; //Block size
    int N; //Vector size
    int ValuesPerThread; //Number of values added by each thread

	// Parse arguments.
    if(argc != 4){
     printf("Usage: GridWidth, BlockWidth, K ", argv[0]);
     printf("ValuesPerThread is the number of values added by each thread.\n");
     printf("Total vector size is 128 * 60 * this value.\n");
     return 0;
    } else {
      sscanf(argv[3], "%d", &K);
      sscanf(argv[1], "%d", &GridWidth);
      sscanf(argv[2], "%d", &BlockWidth);

    }      

    // Determine the number of threads .
    // N is the total number of values to be in a vector
    
    // size_t is the total number of bytes for a vector.
    int n_elem = K * 1000000;
    //printf("Total vector size: %d\n", n_elem); 
    int size = n_elem * sizeof(int);
    int total_threads = GridWidth * BlockWidth;
    ValuesPerThread = n_elem / total_threads;
    N = ValuesPerThread * GridWidth * BlockWidth;

    // Tell CUDA how big to make the grid and thread blocks.
    // Since this is a vector addition problem,
    // grid and thread block are both one-dimensional.
    dim3 dimGrid(GridWidth);                    
    dim3 dimBlock(BlockWidth);                 

    // Allocate vectors in device memory.
    cudaError_t error;
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMallocManaged(&d_A, size);
    cudaMallocManaged(&d_B, size);
    cudaMallocManaged(&d_C, size);

    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     d_A[i] = (float)i;
     d_B[i] = (float)(N-i);   
    }
    

    // Warm up
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, ValuesPerThread);

    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, ValuesPerThread);

    cudaDeviceSynchronize();

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

    // Compute floating point operations per second.
    int nFlops = N;
    double nFlopsPerSec = nFlops/time;
    double nGFlopsPerSec = nFlopsPerSec*1e-9;

	// Compute transfer rates.
    int nBytes = 3*4*N; // 2N words in, 1N word out
    double nBytesPerSec = nBytes/time;
    double nGBytesPerSec = nBytesPerSec*1e-9;

	// Report timing data.
    // printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
    //          time, nGFlopsPerSec, nGBytesPerSec);
     
    // Copy result from device memory to host memory
    

    // Verify & report result
    for (i = 0; i < N; ++i) {
        float val = d_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("", (i == N) ? "" : "FAILED");
    printf("%lf", time);
    // Clean up and exit.
    Cleanup(true);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);


        
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}


