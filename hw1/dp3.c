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

int main(int argc, char *argv[]) {

    long dim = atol(argv[1]);
    float *pA = (float *)malloc(dim * sizeof(float)); 
    float *pB = (float *)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        pA[i] = 1.0;
        pB[i]=1.0;
    }
    int n_iter = atoi(argv[2]); 
    float mean = 0.0;
    for (int i = 0; i < n_iter; i++) {
        struct timespec f_start;
        struct timespec f_end;
        clock_gettime(CLOCK_MONOTONIC, &f_start);
        //float R = dp(dim, pA, pB);
        float R = bdp(dim, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &f_end);
        float tdiff = f_end.tv_nsec - f_start.tv_nsec;
        mean+=tdiff;
    }
    mean = mean/n_iter;
    printf("average run time: %f \n", mean);
    free(pA); // Free dynamically allocated memory
    free(pB);
    return 0; 
}
//gcc -O3 -Wall -o dp3 -I /opt/intel/oneapi/mkl/2022.2.0/include dp3.c -L /opt/intel/oneapi/mkl/2022.2.0/lib -lmkl_rt
