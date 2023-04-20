#include "convKernel.h"

// Define a gpu kernel to perform convolution
// of input_matrix *conv* filters = result.
__global__ void ConvKernel(const Matrix input_matrix, const Matrix* filters, Matrix result){
  int block_row = blockIdx.x;
  int block_col = blockIdx.y;
  int k = blockIdx.z;

  int x = threadIdx.x;
  int y = threadIdx.y;

  Matrix filter = filters[k];

  double* resultSub = &result.elements[block_row * OUT_FOOTPRINT * result.stride_height + block_col * OUT_FOOTPRINT];
  double* input_matrixSub = &input_matrix.elements[block_row * OUT_FOOTPRINT * input_matrix.stride_height + block_col * OUT_FOOTPRINT];
  
  __shared__ double shared_input_matrix[C][OUT_FOOTPRINT + 2 * P][OUT_FOOTPRINT + 2 * P];

  #pragma unroll
  for(int c = 0; c < C; c++) {
    shared_input_matrix[c][x][y] = input_matrixSub[c * input_matrix.stride_channel + x * input_matrix.stride_height + y];
  }

  __syncthreads();

  if(x < P || x >= OUT_FOOTPRINT + P || y < P || y >= OUT_FOOTPRINT + P) {
    return;
  }

  int new_x = x - P;
  int new_y = y - P;

  double result_val = 0;
  for(int c = 0; c < C; c++) {
    for(int j = 0; j < FH; j++) {
      for(int i = 0; i < FW; i++) {
        result_val += filter.elements[c * filter.stride_channel + (FH - 1 - i) * filter.stride_height + (FW - 1 - j)] * shared_input_matrix[c][new_x + i][new_y + j];
      }
    }
  }

  resultSub[k * result.stride_channel + new_x * result.stride_height + new_y] = result_val;
}

