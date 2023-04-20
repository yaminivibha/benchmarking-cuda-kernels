///
/// matmult.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-19 DVN
///
/// Do not modify this file. The GTA will grade your
/// code using the master copy of this file, not your
/// copy, so any modifications you make will not play
/// a role in the grading.
///

// Includes
#include <stdio.h>
#include "timer.h"
#include "convKernel.h"

// Defines
#define epsilon (float)1e-4
#define verbose 0


Matrix MakeDeviceMatrix(const Matrix M, bool copy){
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

Matrix* MakeDeviceFilters(const Matrix* filters, bool copy){
  // Create a new matrix in device memory.
  Matrix* newDeviceFilters;
  cudaMallocManaged((void**) &newDeviceFilters, K * sizeof(Matrix));
  if(copy) {
    for(int k = 0; k < K; k++) {
        newDeviceFilters[k] = MakeDeviceMatrix(filters[k], copy);
    }
  }
  
  return newDeviceFilters;
}

// Create a matrix in host memory.
Matrix MakeHostMatrix(int channels, int height, int width){
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
  Matrix input_matrix = MakeHostMatrix(C, H + 2 * P, W + 2 * P);
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
    Matrix filter = MakeHostMatrix(C, FH, FW);
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

// Print channel 0 of a matrix stored in host memory.
void printMatrix0(Matrix M, const char* name) {
  printf("\n%s \n",name);
  for(int y=0; y<M.height; y++){
   for(int x=0; x<M.width; x++) {
      printf("%f ", M.elements[y * M.width + x]);
   }
   printf("\n");
  }
}

// Compute and check the checksum of the result matrix
// expected to be 122756344698240
void checkResult(Matrix M) {

  double checksum = 122756344698240;
  double checksum_M = 0;
  
  for(int k = 0; k < K; k++) {
    for(int h = 0; h < M.height; h++) {
        for(int w = 0; w < M.width; w++) {
            checksum_M += M.elements[k * M.stride_channel + h * M.stride_height + w];
        }
    }
  }

  if(fabs(checksum - checksum_M)> epsilon * checksum) {
    printf("\n\nTEST FAILED\n");
    printf("computed checksum: %lf\n", checksum_M);
    printf("actual checksum: %lf\n", checksum);
  }
}

// Host code for convolution.
void Conv(const Matrix input_matrix, const Matrix* filters, Matrix result){

  // Create device data structures.
  Matrix input_matrix_device = MakeDeviceMatrix(input_matrix, true);
  Matrix* filters_device = MakeDeviceFilters(filters, true);
  Matrix result_device = MakeDeviceMatrix(result, false);

  // Define grid topology
  dim3 dimBlock(OUT_FOOTPRINT + 2 * P, OUT_FOOTPRINT + 2 * P);
  dim3 dimGrid(H / OUT_FOOTPRINT, W / OUT_FOOTPRINT, K);

  // Invoke kernel for warm up
  ConvKernel<<<dimGrid, dimBlock>>>(input_matrix_device, filters_device, result_device);

  // Synchronize to make sure everyone is done in the warmup.
  cudaThreadSynchronize();

  // Set up timer
  initialize_timer();
  start_timer();

  // Invoke kernel for real
  ConvKernel<<<dimGrid, dimBlock>>>(input_matrix_device, filters_device, result_device);
 
  // Synchronize to make sure everyone is done.
  cudaThreadSynchronize() ;

  // Compute and report the timing results

  stop_timer();
  double time = elapsed_time();

  printf( "Grid Dimensions: %dx%d \n",dimGrid.x,dimGrid.y);
  printf( "Block Dimensions: %dx%d \n",dimBlock.x,dimBlock.y);
  
  printf( "Time: %lf (sec)\n", time);

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

//
// main
//
int main(int argc, char** argv) {

  // Create matrices in host.
  Matrix input_matrix = createIMatrix();
  Matrix* filters = createFilterMatrices();
  Matrix result = MakeHostMatrix(K, H, W);
 
  // debugging
//   if(verbose){
//     printMatrix(host_A, "host_A");
//     printMatrix(host_B, "host_B");
//   }

  // Perform CUDA matrix Multiplication
  // MatMul is a host function that calls
  // the device kernel MatMulKernel and
  // times its performance.
  Conv(input_matrix,filters,result);

  // Verify that the result is correct.
  checkResult(result);
  
  // Free allocated memory.
  free(input_matrix.elements);
  for(int k = 0; k < K; k++) {
    free(filters[k].elements);
  }
  free(filters);
  free(result.elements);
}
