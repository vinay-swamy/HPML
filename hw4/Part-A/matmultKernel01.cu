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
#define BLOCK_SIZE 16
#define WORK_FACTOR 4
#define FOOTPRINT_SIZE 32 // this is probably not needed when compiling via the flag in the make file 

 

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
  int threads_per_block = BLOCK_SIZE * BLOCK_SIZE;
  int sharedmatrix_rows_read_per_block = threads_per_block / FOOTPRINT_SIZE;
  int submatrix_stride = BLOCK_SIZE*BLOCK_SIZE;
  //make the FOOTPRINT_SIZE a function of the block size and the work factor
  // comment out for now
  //float n_ops = BLOCK_SIZE *BLOCKSIZE * WORK_FACTOR;
  //int FOOTPRINT_SIZE = (int)sqrtf(n_ops);


  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  // Addition: change from using block size to footprint size
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes WORK_FACTOR elements of Csub in its copy of CValue
  //declare empty array of cvalues
  float c_arr[WORK_FACTOR];
  for (int i = 0; i < WORK_FACTOR; ++i) {
    c_arr[i] = 0.0;
  }


  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    // A.stride * BLOCK_SIZE * block_row gets us the rigth block, and the first position in the right row, 
    // and BLOCK_SIZE * m gets the right starting position of the submatrix within that row.
    // B.stride * BLOCK_SIZE takes us to a column in B 

    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m]; //get to the right row, then slide right to the right column
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col]; // get to the right column then slide down to the right row



    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Challenge: shared matrices use row/colums indexing, while global ones use pointer indexing; need to account for both;
    //**shared**
    // we have a thread block of 16x16, and a matrix of 32x32. This means that each thread block reads 256 contiguous elements
    // 256/32 = 8; so each thread block reads 8 rows of 32 elements each; do we need to account for any changes int the col dimension? 
    // No; consider it as having each block mapped to its position in the submatrix - it will cover exactly 1/4th of the matrix as a rectangle
    // so incrementing the row and not the column will just slide this rectangle down the matrix, and not change its shape
    //**sub**
    // note that in this copy, because we have already pointed B to the right start of the submatrix, we can use the same copy logic for A and B
    // submatrix are 32x32 rows of an orignal 1024*1024 matrix.
    // thread_row * A.stride advances pointer to the right row, and then col_idx advances it to the right column
    // as the loop iterates, we want to jump the rows by the number of element in 1/4 of the matrix, or 8*32 = 256

    
    for(int i=0; i<2; i++){
      for (int j = 0; j< 2; j++){
        int row_idx = thread_row + i*16;
        int col_idx = thread_col + j*16;
        // end case: i=1, j=1, thread_row=15, thread_col=15
        // row_idx = 15+16 = 31, col_idx = 15+16
        shared_A[row_idx][col_idx] = Asub[row_idx * A.stride + col_idx];
        // the same logic should work for B
        shared_B[row_idx][col_idx] = Bsub[row_idx * A.stride + col_idx];
      }
    
    }
    // Synchronize to ensure all elements are read
    __syncthreads();


    
    /*
    we have a 16x16 thread block, and a 32x32 submatrix. therefore the submatrix can be chunked into 4 "subblocks". Given that we want 4 outputs, each block need 
    
    */

    #pragma unroll
    for (int i = 0; i < WORK_FACTOR; ++i) {
      int mat_x;
      int mat_y;

      if(thread_row % 2 == 0){ //is even
         mat_x = thread_row/ 2;
         mat_y = thread_col;
         
      } else{ // is odd
        mat_x = (thread_row - 1) / 2;
        mat_y = 16 + thread_col;
      };
      mat_x = mat_x + i*8;
      /*
      End case: i=3, thread_row = 15
      
      mat_x = (15-1)/2 + 3*8 = 7 + 24 = 31 GOOD
      */

      for(int e=0; e<FOOTPRINT_SIZE; ++e){
        // if(m==0 && thread_row == 2 && thread_col == 2 && block_row == 0 && block_col == 0 && i==0){

        //   printf("Aval: %f, Bval %f\n",shared_A[thread_row][e] , shared_B[e][thread_col] );

        // }
        c_arr[i]+= shared_A[mat_x][e] * shared_B[e][mat_y];
      }
        //Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];

      // Synchronize to ensure all Cvalues have been incremented
      // before reading in the next shared_A AND shared_B BLOCKS
      __syncthreads();
      // if( thread_row == 2 && thread_col == 2  && block_row==0 &&  i==0){

      //     printf("\nBlockCol:%d\n Cvalue: %f, ",block_col, c_arr[i]);

      //   }
    }
  }

  // Write Csub to GLOBAL memory.
  // Csub has been already allocated 

  for (int i = 0; i < WORK_FACTOR; ++i) {
      int mat_x;
      int mat_y;

      if(thread_row % 2 == 0){ //is even
         mat_x = thread_row/ 2;
         mat_y = thread_col;
         
      } else{ // is odd
        mat_x = (thread_row - 1) / 2;
        mat_y = 16 + thread_col;
      };
      mat_x = mat_x + i*8;
      //Csub[mat_x][mat_y] = c_arr[i]
      Csub[mat_x * C.stride + mat_y ] = c_arr[i];
  }
  
}

