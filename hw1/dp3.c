#include <mkl_cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float bdp(long N, float *pA, float *pB) {
  float R = cblas_sdot(N, pA, 1, pB, 1);
return R; }

float dp(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j++)
        R += pA[j]*pB[j];
    return R;
}
/*  */

int main(int argc, char *argv[]) {

    long dim = atol(argv[1]);
    float *pA = (float *)malloc(dim * sizeof(float)); 
    float *pB = (float *)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        pA[i] = 1.0;
        pB[i]=1.0;
    }
    int n_iter = atoi(argv[2]); 
    int warmup = n_iter/2 ;
    float dp_out=0.0;
    double mean = 0.0;
    for (int i = 0; i < n_iter; i++) {
        struct timespec f_start;
        struct timespec f_end;
        clock_gettime(CLOCK_MONOTONIC, &f_start);
        float tmp = bdp(dim, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &f_end);
        if(i > warmup){
            double nsec_tdiff = (f_end.tv_nsec - f_start.tv_nsec)/1000000000.0;
            double sec_tdiff = f_end.tv_sec -f_start.tv_sec;
            double total_time = sec_tdiff + nsec_tdiff;
            
            mean+=total_time;
            
        }
        dp_out = tmp;
    }
    mean = mean/warmup;
    // report GFlops bc its easier to read 
    double flops = 2*dim / mean/1000000000;
    // wants GB per second so cancels out nanoseconds
    double bandwidth = 8*dim / mean/1000000000 ;
    printf("N: %ld\tT: %.12f sec\tB: %.12f GB/sec\tF: %.12f GFLOP/s\n",dim,  mean, bandwidth,flops);
    printf("dp_out: %.12f\n", dp_out);
    free(pA); // Free dynamically allocated memory
    free(pB);
    return 0; 
}
//gcc -O3 -Wall -o dp3 -I /opt/intel/oneapi/mkl/2022.2.0/include dp3.c -L /opt/intel/oneapi/mkl/2022.2.0/lib -lmkl_rt
