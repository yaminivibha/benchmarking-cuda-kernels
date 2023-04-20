///
/// KMillionAddition.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Created: 2023-04-17
/// Add two Vectors A and B in C on GPU using
/// a kernel defined according to AdditionKernel.h

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
float* h_A; 
float* h_B; 
float* h_C; 
float* d_A; 
float* d_B; 
float* d_C; 

// Variables for grid and block width
int GridWidth;
int BlockWidth;
int ValuesPerThread;

// Variables for input parameters, situation, K and K_million
int K; // multiple of millions
double K_million; // vector size
int situation; // situation number
// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

// Host code performs setup and calls the kernel.
int main(int argc, char* argv[]) {
    // Parse arguments.
    if (argc != 3) {
        printf("Usage: %s <K> <Situation> \n", argv[0]);
        printf("Situtation 1: Using one block with 1 thread \n");
        printf("Situtation 2: Using one block with 256 threads \n");
        printf("Situtation 3: Using multiple blocks etc... \n");
        exit(0);
    }
    
    else {
        sscanf(argv[1], "%d", &K);
        sscanf(argv[2], "%d", &situation);
        if (K == 1){
            K_million = 1000192;
        }
        else if (K == 5){
            K_million = 5000192;
        }
        else if (K==10){
            K_million = 10000128;
        }
        else if (K==50){
            K_million = 50000128;
        }
        else if (K==100){
            K_million = 100000000;
        }
        else{
            printf("Error: K must be 1, 5, 10, 50 or 100.\n");
            exit(0);
        }
    }  

    // defining Grid and Block width by situation
    if (situation == 1){
        BlockWidth = 1;
        GridWidth = 1;
        ValuesPerThread = K_million;
    }
    else if (situation == 2){
        BlockWidth = 1;
        GridWidth = 256;
        ValuesPerThread = K_million / 256;
    }
    else if (situation == 3){
        BlockWidth = K_million / 256;
        GridWidth = 256;
        ValuesPerThread = 1;
    }
    else{
        printf("Error: Situation must be 1, 2 or 3.\n");
        exit(0);
    }
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
    for(i=0; i<K_million; ++i){
     h_A[i] = (float)i;
     h_B[i] = (float)(K_million-i);   
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

    // Copy result from device memory to host memory
    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) Cleanup(false);

    printf("Vector size: %f\n", K_million);
    // Verify & report result
    for (i = 0; i < K_million; ++i) {
        float val = h_C[i];
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