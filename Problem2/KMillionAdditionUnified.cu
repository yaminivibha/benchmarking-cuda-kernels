///
/// KMillionAddition.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Created: 2023-04-17
/// Add two Vectors A and B in C on GPU using
/// a kernel defined according to AdditionKernel.h
/// 

// Includes
#include <iostream>
#include <chrono>
#include <cstdlib> 
#include <cmath>
#include <stdio.h>
#include "timer.h"
#include "AdditionKernel.h"

// Namespaces
using namespace std;

// Variables for host and device vectors.
float* d_A; 
float* d_B; 
float* d_C; 

// Variables for grid and block width
int GridWidth;
int BlockWidth;

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

// Host code performs setup and calls the kernel.
int main(int argc, char* argv[]) {
    int K; // multiple of millions
    int K_million; // vector size
    int situation; // situation number

    // Parse arguments.
    if (argc != 3) {
        printf("Usage: %s K Scenario \n", argv[0]);
        printf("Situtation 1: Using one block with 1 thread \n");
        printf("Situtation 2: Using one block with 256 threads \n");
        printf("Situtation 3: Using multiple blocks etc... \n");
        exit(0);
    }
    else {
        sscanf(argv[1], "%d", &K);
        sscanf(argv[2], "%d", &situation);
        K_million = K * 1000000; 
    }  

    // defining Grid and Block width by situation
    if (situation == 1){
        BlockWidth = 1;
        GridWidth = 1;
    }
    else if (situation == 2){
        BlockWidth = 1;
        GridWidth = 256;
    }
    
    size_t size = K_million * sizeof(float);
    
    // Tell CUDA how big to make the grid and thread blocks.
    // Since this is a vector addition problem,
    // grid and thread block are both one-dimensional.
    dim3 dimGrid(GridWidth);                    
    dim3 dimBlock(BlockWidth);     

    // Allocate vectors in device memory.
    cudaError_t error;
    error = cudaMallocManaged((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize vectors d_A and d_B
    int i;
    for(i=0; i<K_million; ++i){
     d_A[i] = (float)i;
     d_B[i] = (float)(K_million-i);   
    }    
    
    // Warm up
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, K_million);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, K_million);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

    // Compute floating point operations per second.
    int nFlops = K_million;
    double nFlopsPerSec = nFlops/time;
    double nGFlopsPerSec = nFlopsPerSec*1e-9;

	// Compute transfer rates.
    int nBytes = 3*4*K_million; // 2N words in, 1N word out
    double nBytesPerSec = nBytes/time;
    double nGBytesPerSec = nBytesPerSec*1e-9;

	// Report timing data.
    printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
             time, nGFlopsPerSec, nGBytesPerSec);

    // // Copy result from device memory to host memory
    // error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    // if (error != cudaSuccess) Cleanup(false);

    // Verify & report result
    for (i = 0; i < K_million; ++i) {
        float val = d_C[i];
        if (fabs(val - K_million) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == K_million) ? "PASSED" : "FAILED");

    // Clean up and exit.
    Cleanup(true);
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