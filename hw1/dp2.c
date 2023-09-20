#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB) {
  float R = 0.0;
  int j;
  for (j=0;j<N;j+=4)
    R += pA[j]*pB[j] + pA[j+1]*pB[j+1] \
           + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
return R; }

int main(int argc, char *argv[]) {

    long dim = atol(argv[1]);
    float *pA = (float *)malloc(dim * sizeof(float)); 
    float *pB = (float *)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        pA[i] = 1.0;
        pB[i]=1.0;
    }
    int n_iter = atoi(argv[2]); 
    float mean = 0;
    for (int i = 0; i < n_iter; i++) {
        struct timespec f_start;
        struct timespec f_end;
        clock_gettime(CLOCK_MONOTONIC, &f_start);
        float R = dpunroll(dim, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &f_end);
        float tdiff = f_end.tv_nsec - f_start.tv_nsec;
        mean+=tdiff;
    }
    mean = mean/n_iter;
    printf("average runtime: %f \n", mean);
    free(pA); // Free dynamically allocated memory
    free(pB);
    return 0; 
}