///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

#define BLOCK_SIZE 16
#define WORK_FACTOR 4

 

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
  /*
  Notes:
  Everything is in Row major format. 
  since we have A x B = C, We access the rows of A and the cols of B
  As we adavance along the rows of A, we update values in increasing *cols* of C 
  Similarly, as we advance along the cols of B, we update values in increasing *rows* of C

  when in order to copmute 4 operations, we need to increase the amount of data we load in (footprint?), while keeping the block size 
  constant. 16*16 = 256, so if each thread needs to do 4 ops, each block will need to load in 256*4 = 1024 elements.
  this can be acheived by loading in 32*32 elements, and having each thread do 4 ops
  
  */

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  // Addition: change from using block size to footprint size 
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors

    // A.stride * BLOCK_SIZE * block_row gets us the rigth block, and the first position in the right row, 
    // and BLOCK_SIZE * m gets the right starting position of the submatrix within that row.
    // B.stride * BLOCK_SIZE takes us to a column in B 

    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Now each thread copies 4 elements into memory. We should use a coalesced transaction for this 
    // but we are not for now
    // A sub is 32 x 32 matrx, so we have 1024 elements, so I think we load in like we did for the vecadd
    // bc the matrix is just one giant vector
    // this logic locks us into using a bl
    int loop_stride = 1024;
    int loop_end = 1024*4;

    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e)
      //NOTE: shared memory does not require coalescing(coalesing is specifically for offchip<>onchip memory transactions)
       Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  Csub[thread_row * C.stride + thread_col] = Cvalue;
}

