///
/// KMillionAddition.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Created: 2023-04-17
/// Add two Vectors A and B in C on CPU

// Includes
#include <iostream>
#include <chrono>
#include <cstdlib> 
#include <cmath>
#include "timer.h"

using namespace std;

// Defines
#define GridWidth 60
#define BlockWidth 128

// Variables for host and device vectors.
float* h_A; 
float* h_B; 
float* h_C; 

// Utility Functions
void Cleanup(bool);

int main(int argc, char* argv[]) {
  // correcting usage
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " K" << endl;
    return 1;
  }
  const int K = stoi(argv[1]);
  const int K_million = K * 1000000; 
  const int N = K_million;
  size_t size = K_million * sizeof(float);

  // Tell CUDA how big to make the grid and thread blocks.
  // Since this is a vector addition problem,
  // grid and thread block are both one-dimensional.          

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup(false);
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup(false);
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup(false);


    // Allocate vectors in device memory.
    // cudaError_t error;
    // error = cudaMalloc((void**)&d_A, size);
    // if (error != cudaSuccess) Cleanup(false);
    // error = cudaMalloc((void**)&d_B, size);
    // if (error != cudaSuccess) Cleanup(false);
    // error = cudaMalloc((void**)&d_C, size);
    // if (error != cudaSuccess) Cleanup(false);


    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     h_A[i] = (float)i;
     h_B[i] = (float)(N-i);   
    }

  // timing only the addition
  auto start_time = chrono::steady_clock::now();
  for (i = 0; i < K_million; i++) {
    h_C[i] = h_A[i] + h_B[i];
  }
  auto end_time = chrono::steady_clock::now();
  auto duration = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();  
  // printing result
  cout << "Runtime: " << duration << " s" << endl;
  
  // Verify & report result
    for (i = 0; i < K_million; ++i) {
        float val = h_C[i];
        if (fabs(val - K_million) > 1e-5)
            break;
    }
    if (i == K_million)
        printf("PASSED\n");
    else
        printf("FAILED\n");

  // Clean up and exit.
    Cleanup(true);
  return 0;
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);

    fflush( stdout);
    fflush( stderr);

    exit(0);
}