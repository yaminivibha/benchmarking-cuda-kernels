#include <iostream>
#include <chrono>
#include <cstdlib> 
#include <cmath>
#include "timer.h"
#include "AdditionKernel.h"

using namespace std;

// Defines
#define GridWidth 60
#define BlockWidth 128

// Variables for host and device vectors.
float* h_A; 
float* h_B; 
float* h_C; 
float* d_A; 
float* d_B; 
float* d_C; 

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);


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
  dim3 dimGrid(GridWidth);                    
  dim3 dimBlock(BlockWidth);                 

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup(false);
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup(false);
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup(false);


    // Allocate vectors in device memory.
    cudaError_t error;
    error = cudaMalloc((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);


    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     h_A[i] = (float)i;
     h_B[i] = (float)(N-i);   
    }
   // Copy host vectors h_A and h_B to device vectores d_A and d_B
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);

    // Warm up
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, ValuesPerThread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, ValuesPerThread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();


  // timing only the addition
  auto start_time = chrono::steady_clock::now();
  for (int i = 0; i < K_million; i++) {
    result[i] = arr1[i] + arr2[i];
  }
  auto end_time = chrono::steady_clock::now();
auto duration = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();  
  // printing result
  cout << "Runtime: " << duration << " s" << endl;
  
  // Verify & report result
  int i;
    for (i = 0; i < K_million; ++i) {
        float val = result[i];
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
    cudaError_t error;

    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);

    error = cudaDeviceReset();

    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");

    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}