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
#include <stdio.h>

#define FOOTPRINT_SIZE BLOCK_SIZE

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
  /*
  Notes:
  Everything is in Row major format. 
  since we have A x B = C, We access the rows of A and the cols of B
  As we adavance along the rows of A, we update values in increasing *cols* of C 
  Similarly, as we advance along the cols of B, we update values in increasing *rows* of C
  
  */

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  // 16x16 thread blocks
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  // grid dim: dim3 dimGrid(B.width/dimension, A.height/dimension), dimension = FOOTPRINT_SIZE  = BLOCK_SIZE;
  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / BLOCK_SIZE); ++m){
    // Get Asub and Bsub descriptors

    
    
    // A.stride * BLOCK_SIZE * block_row gets us the rigth block, and the first position in the right row, 
    // and BLOCK_SIZE * m gets the right starting position of the submatrix within that row.
    // B.stride * BLOCK_SIZE takes us to a column in B 

    Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread copies just one element of shared_A and one element of shared_B
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
    __syncthreads();
    
    

    // Synchronize to ensure all elements are read
   

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e){
      //NOTE: shared memory does not require coalescing(coalesing is specifically for offchip<>onchip memory transactions)

       
    
       Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];
       }
    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
    // if(thread_row == 2 && thread_col == 2 && block_row == 0 && block_col == 0){

    //       printf("Cvalue: %f, ",Cvalue );

    //     }
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  // if(thread_row == 2 && thread_col == 2 && block_row == 0 && block_col == 0){

  //         printf("cval: %f\n", Cvalue);

  //       }
  Csub[thread_row * C.stride + thread_col] = Cvalue;
  // __syncthreads();
  
}

