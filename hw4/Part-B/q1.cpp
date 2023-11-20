#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std::chrono;
using namespace std;

int main(int argc, char* argv[]) {
    // Check if command line argument is passed correctly
    if (argc < 2) {
        cout << "bad inupt" << endl ;
        return 1;
    }

    int K = atoi(argv[1]);
    int n_elem = K * 1000000;
    int size = n_elem * sizeof(int);

    // Allocate memory for the arrays
    int* inp1 = (int*)malloc(size );
    int* inp2 = (int*)malloc(size);
    int* output = (int*)malloc(size);

    // Initialize arrays 
    for (int i = 0; i < n_elem; ++i) {
        inp1[i] = 1; 
        inp2[i] = 1; 
    }

    // Measure the start time
    auto start = high_resolution_clock::now();

    // Add elements of the arrays
    for (size_t i = 0; i < n_elem; ++i) {
        output[i] = inp1[i] + inp2[i];
    }

    // Measure the end time
    auto end = high_resolution_clock::now();

    // Calculate and print the time taken to add the arrays
    auto rt = duration_cast<microseconds>(end - start);
    float runtime = (float)rt.count()/1000000;
    printf("%f", runtime);

    // Free the allocated memory
    free(inp1);
    free(inp2);
    free(output);

    return 0;
}
