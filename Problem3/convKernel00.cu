#include "convKernel.h"

// Define a gpu kernel to perform convolution
// of input_matrix *conv* filters = result.
__global__ void ConvKernel(const Matrix input_matrix, const Matrix* filters, Matrix result){
  // total K * H * W threads
  // each thread computes one output pixel at resuls[k, x, y]
  int x = blockIdx.x;
  int y = blockIdx.y;
  int k = threadIdx.x;
  Matrix filter = filters[k];

  double result_val = 0;
  for(int c = 0; c < C; c++) {
    for(int j = 0; j < FH; j++) {
      for(int i = 0; i < FW; i++) {
        result_val += filter.elements[c * filter.stride_channel + (FH - 1 - i) * filter.stride_height + (FW - 1 - j)] * input_matrix.elements[c * input_matrix.stride_channel + (x + i) * input_matrix.stride_height + (y + j)];
      }
    }
  }

  result.elements[k * result.stride_channel + x * result.stride_height + y] = result_val;
}

