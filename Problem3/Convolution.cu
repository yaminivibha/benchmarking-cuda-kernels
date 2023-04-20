///
/// Convolution.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-19 DVN
///
/// Yamini Ananth

// Includes
#include <stdio.h>
#include "timer.h"

// Defines
#define C 3
#define H 1024
#define W 1024
#define K 64
#define FH 3
#define FW 3
#define P 1
#define OUT_FOOTPRINT 16
#define BLOCK_SIZE 16

#ifndef __MMKERNEL__
#define __MMKERNEL__
#endif

#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// Matrix Structure declaration
typedef struct {
  int channels;
  int height;
  int width;
  int stride_channel;
  int stride_height;
  double* elements;
} Matrix;


// function declaration
double checkSum(Matrix M);

__global__ void ConvKernel(const Matrix input_matrix, const Matrix* filters, Matrix result){
  // total K * H * W threads
  // each thread computes one output pixel at resuls[k, x, y]
  int x = blockIdx.x;
  int y = blockIdx.y;
  int k = threadIdx.x;
  Matrix filter = filters[k];

  double result_val = 0;
  // Convolution
  // for each channel, do convolution. Convolution defined as sum of element-wise multiplication of filter and input matrix
  for(int c = 0; c < C; c++) {
    for(int j = 0; j < FH; j++) {
      for(int i = 0; i < FW; i++) {
        result_val += filter.elements[c * filter.stride_channel + (FH - 1 - i) * filter.stride_height + (FW - 1 - j)] * input_matrix.elements[c * input_matrix.stride_channel + (x + i) * input_matrix.stride_height + (y + j)];
      }
    }
  }
    // Indexing into the result
  result.elements[k * result.stride_channel + x * result.stride_height + y] = result_val;
}

__global__ void ConvKernelTiled(const Matrix input_matrix, const Matrix* filters, Matrix result){
  int BlockRow = blockIdx.x;
  int BlockCol = blockIdx.y;
  int k = blockIdx.z;

  int x = threadIdx.x;
  int y = threadIdx.y;

  Matrix filter = filters[k];

  double* resultSub = &result.elements[BlockRow * OUT_FOOTPRINT * result.stride_height + BlockCol * OUT_FOOTPRINT];
  double* input_matrixSub = &input_matrix.elements[BlockRow * OUT_FOOTPRINT * input_matrix.stride_height + BlockCol * OUT_FOOTPRINT];
  
  // Shared memory for input matrix
  __shared__ double shared_input_matrix[C][OUT_FOOTPRINT + 2 * P][OUT_FOOTPRINT + 2 * P];

  // Load input matrix into shared memory
  #pragma unroll
  for(int c = 0; c < C; c++) {
    shared_input_matrix[c][x][y] = input_matrixSub[c * input_matrix.stride_channel + x * input_matrix.stride_height + y];
  }
  // synchronize to make sure the matrix is loaded
  __syncthreads();
   // verify if the thread is inside the footprint
  if(x < P || x >= OUT_FOOTPRINT + P || y < P || y >= OUT_FOOTPRINT + P) {
    return;
  }
  // calculate the new x and y
  int new_x = x - P;
  int new_y = y - P;
  // calculate the result
  double result_val = 0;
  for(int c = 0; c < C; c++) {
    for(int j = 0; j < FH; j++) {
      for(int i = 0; i < FW; i++) {
        result_val += filter.elements[c * filter.stride_channel + (FH - 1 - i) * filter.stride_height + (FW - 1 - j)] * shared_input_matrix[c][new_x + i][new_y + j];
      }
    }
  }
  // store the result
  resultSub[k * result.stride_channel + new_x * result.stride_height + new_y] = result_val;
}

Matrix createDeviceMatrix(const Matrix M, bool copy){
  // Create a new matrix in device memory.
  Matrix newDeviceMatrix;
  newDeviceMatrix.channels = M.channels;
  newDeviceMatrix.height = M.height;
  newDeviceMatrix.width = M.width;
  newDeviceMatrix.stride_channel = M.stride_channel;
  newDeviceMatrix.stride_height = M.stride_height;
  size_t size = M.channels * M.height * M.width * sizeof(double);
  cudaMallocManaged((void**) &newDeviceMatrix.elements, size);
  if (copy)
    cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
  return newDeviceMatrix;
}

// Create filters in device memory.
Matrix* createDeviceFilters(const Matrix* filters, bool copy){
  Matrix* newDeviceFilters;
  cudaMallocManaged((void**) &newDeviceFilters, K * sizeof(Matrix));
  if(copy) {
    for(int k = 0; k < K; k++) {
        newDeviceFilters[k] = createDeviceMatrix(filters[k], copy);
    }
  }
  return newDeviceFilters;
}

// Create matrix in host memory.
Matrix createHostMatrix(int channels, int height, int width){
  Matrix newHostMatrix;
  newHostMatrix.channels = channels;
  newHostMatrix.height = height;
  newHostMatrix.width = width;
  newHostMatrix.stride_channel = newHostMatrix.height * newHostMatrix.width;
  newHostMatrix.stride_height = newHostMatrix.width;
  size_t size = newHostMatrix.channels * newHostMatrix.height * newHostMatrix.width * sizeof(double);
  newHostMatrix.elements = (double*)malloc(size);
  return newHostMatrix;
}

// Create input matrix stored in host memory.
Matrix createIMatrix() {
  Matrix input_matrix = createHostMatrix(C, H + 2 * P, W + 2 * P);
  for(int c = 0; c < input_matrix.channels; c++) {
    for(int h = 0; h < input_matrix.height; h++) {
        for(int w = 0; w < input_matrix.width; w++) {
            if(h == 0 || h == input_matrix.height - 1 || w == 0 || w == input_matrix.width - 1) {
                input_matrix.elements[c * input_matrix.stride_channel + h * input_matrix.stride_height + w] = 0;
            }
            else {
                input_matrix.elements[c * input_matrix.stride_channel + h * input_matrix.stride_height + w] = c * ((h - P) + (w - P));
            }
        }
    }
  }
  return input_matrix;
}

// Create all K filter matrices on host
Matrix* createFilterMatrices() {
  Matrix* filters = (Matrix*) malloc(K * sizeof(Matrix));
  for(int k = 0; k < K; k++) {
    Matrix filter = createHostMatrix(C, FH, FW);
    for(int c = 0; c < filter.channels; c++) {
        for(int h = 0; h < filter.height; h++) {
            for(int w = 0; w < filter.width; w++) {
                filter.elements[c * filter.stride_channel + h * filter.stride_height + w] = (c + k) * (h + w);
            }
        }
    }
    filters[k] = filter;
  }
  return filters;
}

// Compute and check the checksum of the result matrix
double checkSum(Matrix M) {

  // expected result
  double checksum = 122756344698240;
  // actual result
  double checksum_computed = 0;
  
  for(int k = 0; k < K; k++) {
    for(int h = 0; h < M.height; h++) {
        for(int w = 0; w < M.width; w++) {
            checksum_computed += M.elements[k * M.stride_channel + h * M.stride_height + w];
        }
    }
  }
    return checksum_computed;
}

// Host code for convolution.
void Conv(const Matrix input_matrix, const Matrix* filters, Matrix result){

  // Create device data structures.
  Matrix input_matrix_device = createDeviceMatrix(input_matrix, true);
  Matrix* filters_device = createDeviceFilters(filters, true);
  Matrix result_device = createDeviceMatrix(result, false);

  // Set up grid
  dim3 dimBlock(K);
  dim3 dimGrid(H, W);

  // Warm-up
  ConvKernel<<<dimGrid, dimBlock>>>(input_matrix_device, filters_device, result_device);

  // Synchronize
  cudaThreadSynchronize();

  // Set up timer
  initialize_timer();
  start_timer();

  // True computation
  ConvKernel<<<dimGrid, dimBlock>>>(input_matrix_device, filters_device, result_device);
 
  //synchronization
  cudaThreadSynchronize() ;

  // Stop timer
  stop_timer();
  double time = elapsed_time();
  printf("C1\n");
  
  // Compute checksum and print time
  double checksum;
  checksum = checkSum(result);
  printf( "Checksum: %f Time: %f (sec)\n", checksum, time);

  // Copy the result to the host memory from device memory
  size_t size = result.channels * result.height * result.width * sizeof(double);
  cudaMemcpy(result.elements, result_device.elements, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(input_matrix_device.elements);
  for(int k = 0; k < K; k++) {
    cudaFree(filters_device[k].elements);
  }
  cudaFree(filters_device);
  cudaFree(result_device.elements);
}

// Host code for convolution - tiled
void ConvTiled(const Matrix input_matrix, const Matrix* filters, Matrix result){

  // Create device data structures.
  Matrix input_matrix_device = createDeviceMatrix(input_matrix, true);
  Matrix* filters_device = createDeviceFilters(filters, true);
  Matrix result_device = createDeviceMatrix(result, false);

  // Set up grid
  dim3 dimBlock(OUT_FOOTPRINT + 2 * P, OUT_FOOTPRINT + 2 * P);
  dim3 dimGrid(H / OUT_FOOTPRINT, W / OUT_FOOTPRINT, K);

  // Warm up
  ConvKernelTiled<<<dimGrid, dimBlock>>>(input_matrix_device, filters_device, result_device);

  // Synchronize to make sure everyone is done in the warmup.
  cudaThreadSynchronize();

  // Set up timer
  initialize_timer();
  start_timer();

  // True computation
  ConvKernelTiled<<<dimGrid, dimBlock>>>(input_matrix_device, filters_device, result_device);
 
  // Synchronize 
  cudaThreadSynchronize() ;

  // Stop timer
  stop_timer();
  double time = elapsed_time();

  // printing checksum and time
  printf("C2\n");
  double checksum;
  checksum = checkSum(result);
  printf( "Checksum: %f Time: %f (sec)\n", checksum, time);
  
  // Copy the result to the host memory from device memory
  size_t size = result.channels * result.height * result.width * sizeof(double);
  cudaMemcpy(result.elements, result_device.elements, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(input_matrix_device.elements);
  for(int k = 0; k < K; k++) {
    cudaFree(filters_device[k].elements);
  }
  cudaFree(filters_device);
  cudaFree(result_device.elements);
}

int main() {

  // Create matrices in host.
  Matrix input_matrix = createIMatrix();
  Matrix* filters = createFilterMatrices();
  Matrix result = createHostMatrix(K, H, W);

  // Perform CUDA matrix Multiplication
  // MatMul is a host function that calls
  // the device kernel MatMulKernel and
  // times its performance.
  Conv(input_matrix,filters,result);
  ConvTiled(input_matrix,filters,result);
  // Free allocated memory.
  free(input_matrix.elements);
  for(int k = 0; k < K; k++) {
    free(filters[k].elements);
  }
  free(filters);
  free(result.elements);
}
