#include <iostream>
#include <chrono>
#include <cstdlib> 
#include <cmath>

// Includes
#include <stdio.h>
#include "timer.h"
#include "AdditionKernel.h"

// Defines
#define GridWidth 60
#define BlockWidth 128

     

using namespace std;

int main(int argc, char* argv[]) {
  // correcting usage
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " K" << endl;
    return 1;
  }
  const int K = stoi(argv[1]);
  const int K_million = K * 1000000; 
  int* arr1 = new int[K_million];
  int* arr2 = new int[K_million];
  int* result = new int[K_million];

  dim3 dimGrid(GridWidth);                    
  dim3 dimBlock(BlockWidth);   
  for (int i = 0; i < K_million; i++) {
    arr1[i] = (float)i;
    arr2[i] = (float)(K_million-i);   
  }
    // Warm up
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, ValuesPerThread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

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

  // freeing memory
  free(arr1);
  free(arr2);
  free(result);
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