#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dp(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j++){
        R += pA[j]*pB[j];
    }
    return R;
}

int main(int argc, char *argv[]) {

    long dim = atol(argv[1]);
    float *pA = (float *)malloc(dim * sizeof(float)); 
    float *pB = (float *)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        pA[i] = 1.0;
        pB[i] = 1.0;
    }
    int n_iter = atoi(argv[2]); 
    int warmup = n_iter/2 ;

    double mean = 0.0;
    float tally = 0.0;
    float dp_out = 0.0;
    for (int i = 0; i < n_iter; i++) {
        struct timespec f_start;
        struct timespec f_end;
        clock_gettime(CLOCK_MONOTONIC, &f_start);
        float tmp = dp(dim, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &f_end);
        tally+=tmp;
        tally = 0;
        if(i > warmup){
            double nsec_tdiff = (f_end.tv_nsec - f_start.tv_nsec)/1000000000.0;
            double sec_tdiff = f_end.tv_sec -f_start.tv_sec;
            double total_time = sec_tdiff + nsec_tdiff;
            mean+=total_time;
            //printf("%.12f\n", mean);
            
        }
        dp_out = tmp;
    }
    //printf("%f", tmp);
    warmup = warmup; 
    mean = mean/warmup;
    //printf("%i\n", warmup);
    //printf("%.12f\n", mean);
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
//gcc -O3 -Wall -o dp1 dp1.c