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
// Matrices have 3 channels (by default)
// height = y
// width = x
typedef struct {
  int channels;
  int height;
  int width;
  double* elements;
} Matrix;

//Declaring functions
__global__ void ConvKernelTiled(const Matrix input_matrix, const Matrix* filters, Matrix result);
__global__ void ConvKernel(const Matrix input_matrix, const Matrix* filters, Matrix result);
void ConvTiled(const Matrix input_matrix, const Matrix* filters, Matrix result);
void Convolution(const Matrix input_matrix, const Matrix* filters, Matrix result);
Matrix createHostMatrix(int channels, int height, int width);
Matrix createInputMatrix();
Matrix createDeviceMatrix(const Matrix M);
void Copy(const Matrix M, const Matrix newDeviceMatrix);
Matrix* createFilterMatrices();
Matrix* createDeviceFilters(const Matrix* filters);
double checkSum(Matrix M);

int main() {

  // Create matrices in host.
  Matrix input = createInputMatrix();
  Matrix* filters = createFilterMatrices();
  
  Matrix result;
  result.channels = K;
  result.height = H;
  result.width = W;
  size_t size = result.channels * result.height * result.width * sizeof(double);
  result.elements = (double*)malloc(size);

  // Part C1
  Convolution(input, filters, result);
  
  // Part C2
  ConvTiled(input, filters, result);
  
  // Free allocated memory.
  free(input.elements);
  for(int k = 0; k < K; k++) {
    free(filters[k].elements);
  }
  free(filters);
  free(result.elements);
}

// Create matrix in host memory.
Matrix createHostMatrix(int channels, int height, int width){
  Matrix h;
  h.channels = channels;
  h.height = height;
  h.width = width;
  size_t size = h.channels * h.height * h.width * sizeof(double);
  h.elements = (double*)malloc(size);
  return h;
}

// Create input matrix stored in host memory.
Matrix createInputMatrix() {
  Matrix input = createHostMatrix(C, H + 2 * P, W + 2 * P);
  for(int c = 0; c < input.channels; c++) {
    for(int h = 0; h < input.height; h++) {
        for(int w = 0; w < input.width; w++) {
            if(h == 0 || h == input.height - 1 || w == 0 || w == input.width - 1) {
              // set the border to 0 (padding)
                input.elements[c * input.height * input.width + h * input.width + w] = 0;
            }
            else {
              // set the rest of the matrix to c * (h + w)
                input.elements[c * input.height * input.width + h * input.width + w] = c * ((h - P) + (w - P));
            }
        }
    }
  }
  return input;
}

Matrix createDeviceMatrix(const Matrix M){
  // Create a new matrix in device memory.
  Matrix newDeviceMatrix;
  newDeviceMatrix.channels = M.channels;
  newDeviceMatrix.height = M.height;
  newDeviceMatrix.width = M.width;
  size_t size = M.channels * M.height * M.width * sizeof(double);
  cudaMallocManaged((void**) &newDeviceMatrix.elements, size);
  return newDeviceMatrix;
}

void Copy(const Matrix M, const Matrix newDeviceMatrix){
    size_t size = M.channels * M.height * M.width * sizeof(double);
    cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
}


// Create all K filter matrices on host
Matrix* createFilterMatrices() {
  Matrix* filters = (Matrix*) malloc(K * sizeof(Matrix));
  for(int k = 0; k < K; k++) {
    Matrix filter = createHostMatrix(C, FH, FW);
    for(int c = 0; c < filter.channels; c++) {
        for(int h = 0; h < filter.height; h++) {
            for(int w = 0; w < filter.width; w++) {
                filter.elements[c * filter.height * filter.width + h * filter.width + w] = (c + k) * (h + w);
            }
        }
    }
    filters[k] = filter;
  }
  return filters;
}
// Create filters in device memory.
Matrix* createDeviceFilters(const Matrix* filters){
  Matrix* newDeviceFilters;
  cudaMallocManaged((void**) &newDeviceFilters, K * sizeof(Matrix));
  return newDeviceFilters;
}

// Compute and check the checksum of the result matrix
double checkSum(Matrix M) {
  // actual result
  double checksum_computed = 0;
  
  for(int k = 0; k < K; k++) {
    for(int h = 0; h < M.height; h++) {
        for(int w = 0; w < M.width; w++) {
            checksum_computed += M.elements[k * M.height * M.width + h * M.width + w];
        }
    }
  }
    return checksum_computed;
}

// Host code for convolution.
void Convolution(const Matrix input_matrix, const Matrix* filters, Matrix result){

  // Create device data structures.
  Matrix input_matrix_device = createDeviceMatrix(input_matrix);
  Copy(input_matrix, input_matrix_device)
  Matrix* filters_device = createDeviceFilters(filters);
  Copy(filters, filters_device);
  Matrix result_device = createDeviceMatrix(result);

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
  checksum = checkSum(result_device);
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
  Matrix input_matrix_device = createDeviceMatrix(input_matrix);
  Copy(input_matrix, input_matrix_device);
  Matrix* filters_device = createDeviceFilters(filters);
  Copy(filters, filters_device);
  Matrix result_device = createDeviceMatrix(result);

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
        result_val += filter.elements[c * filter.height * filter.width + (FH - 1 - i) * filter.width + (FW - 1 - j)] * input_matrix.elements[c * input_matrix.height * input_matrix.width + (x + i) * input_matrix.width + (y + j)];
      }
    }
  }
    // Indexing into the result
  result.elements[k * result.height * result.width + x * result.width + y] = result_val;
}

__global__ void ConvKernelTiled(const Matrix input_matrix, const Matrix* filters, Matrix result){
  int BlockRow = blockIdx.x;
  int BlockCol = blockIdx.y;
  int k = blockIdx.z;

  int x = threadIdx.x;
  int y = threadIdx.y;

  Matrix filter = filters[k];

  double* temp_result = &result.elements[BlockRow * OUT_FOOTPRINT * result.width + BlockCol * OUT_FOOTPRINT];
  double* temp_input_matrix = &input_matrix.elements[BlockRow * OUT_FOOTPRINT * input_matrix.width + BlockCol * OUT_FOOTPRINT];
  
  // Shared memory for input matrix
  __shared__ double shared_input_matrix[C][OUT_FOOTPRINT + 2 * P][OUT_FOOTPRINT + 2 * P];

  // Load input matrix into shared memory
  #pragma unroll
  for(int c = 0; c < C; c++) {
    shared_input_matrix[c][x][y] = temp_input_matrix[c * input_matrix.height * input_matrix.width + x * input_matrix.width + y];
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
        result_val += filter.elements[c * filter.height * filter.width + (FH - 1 - i) * filter.width + (FW - 1 - j)] * shared_input_matrix[c][new_x + i][new_y + j];
      }
    }
  }
  // store the result
  temp_result[k * result.height * result.width + new_x * result.width + new_y] = result_val;
}
