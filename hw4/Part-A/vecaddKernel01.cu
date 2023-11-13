
/// NOTE: N is not the length of the vector, but the number operations to do per thread
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x ;
    int threadStartIndex = blockStartIndex + threadIdx.x ;
    int stride = blockDim.x * gridDim.x;
    int threadEndIndex  = threadStartIndex + stride*N ;
    int i;

    for( i=threadStartIndex; i<threadEndIndex; i+=stride ){
        C[i] = A[i] + B[i];
    }
}
