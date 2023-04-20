///
/// KMillionAddition.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Created: 2023-04-17
/// Yamini Ananth
/// Perform a simple convolution in CUDA
/// 3x3

// Includes
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

// Defines
#define epsilon (float)1e-4
#define verbose 0

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define P 1
#define K 64

typedef struct {
    int c;
    int width;
    int height;
    int stride;
    double* elements;
} Tensor;

Tensor MakeDeviceTensor(Tensor M, bool copy){
  // Create a new matrix in device memory.
  Tensor newDeviceTensor;
  newDeviceTensor.c = M.c;
  newDeviceTensor.width = M.width;
  newDeviceTensor.stride = M.width;
  newDeviceTensor.height = M.height;
  size_t size = M.c * M.width * M.height * sizeof(double);
  double* elements;
  cudaMalloc((void**) &newDeviceTensor.elements, size);
  if (copy)
    cudaMemcpy(newDeviceTensor.elements, M.elements, size, cudaMemcpyHostToDevice);
  return newDeviceTensor;
}

// Create a tensor in host memory.
Tensor MakeHostTensor(int c, int h, int w){
    Tensor newHostTensor;
    newHostTensor.c = c;
    newHostTensor.width = w;
    newHostTensor.height = h;
    size_t size = c * h * w * sizeof(double);
    newHostTensor.elements = (double*)malloc(size);
    return newHostTensor;
}

void Convo(const Tensor I0, Tensor O){
  Tensor device_I0 = MakeDeviceTensor(I0, true);
  Tensor host_O = MakeHostTensor(K, W, H);
  Tensor device_O = MakeDeviceTensor(host_O, false);

  // Define grid topology
  dim3 dimBlock(16, 16);
  dim3 dimGrid((W/16, H/16));

  cudaThreadSynchronize();

  // Set up timer

  initialize_timer();
  start_timer();
  double time = elapsed_time();

  convolution<<dimGrid, dimBlock>>(I0, O);
  cudaThreadSynchronize();

  printf("Standard convolution");
  
  cudaMemcpy(host_O, device_O, K*H*W * sizeof(double), cudaMemcpyDeviceToHost);
  stop_timer();
  double time = elapsed_time();
  printf("Checksum: %d", checksum(host_O));
  printf( "Time: %lf (sec)", time);
}


double checksum(Tensor T){
  double cs = 0;
  for(int i = 0; i < T.height * T.width * T.c; i++){
    cs += T.elements[i];
  }
  return cs;
}


// Initialize dummy data in a tensor stored in host memory.
void initTensor(Tensor T, bool horizontal) {
  for(int z=0; z < T.c; z++){
    for(int y=0; y<T.height; y++) {
      for(int x=0; x<T.width; x++) {
        T.elements[z*T.height*T.width + y*T.width + x] = z * (x+y);
      }
    }
  }
}

// Initialize dummy data in a tensor stored in host memory.
void initTensorPadded(Tensor T) {
  for(int z=0; z < T.c; z++){
    for(int y=0; y<T.height; y++) {
      for(int x=0; x<T.width; x++) {
        if(x == 0 || y == 0 || x == T.width-1 || y == T.height-1){
          T.elements[z*T.height * T.width + y*T.width + x] = 0;
        }
        else{
          T.elements[z*T.height*T.width + y*T.width + x] = z * (x+y);
        }
      }
    }
  }
}

__global__ void convolution(Tensor I0, Tensor O){
  int thread_col = threadIdx.x + blockIdx.x + blockDim.x;
  int thread_row = threadIdx.y + blockIdx.y + blockDim.y;

  double *Oadd = &O.elements;

  if(thread_row < H && thread_col < W){
    for(int k = 0; i<K; i++){
      double o = 0;
      for(int c = 0; c < C; ++c){
        for(int i = 0; i<FH; ++i){
          for(int j = 0; j<FW; ++j){
            if(thread_col + j < W+2 && thread_row + i < H+2){
              o += (k+c)*(FW-1-i + FH-1-j) * I0.elements[c * I0.width * I0.height + (thread_row + i) * I0.width + thread_col + j];
            }
          }
        }
      }
      Oadd[k*W*H + thread_row*W + thread_col] = o;
    }
  }
}

int main() {

  // Grid dimension
  int num_blocks;
  // Matrix dimensions in multiples of FOOTPRINT_SIZE
  // Matrices will be of size data_size * data_size
  int data_size;

  // Create matrices in host.
  Tensor host_I0 = MakeHostTensor(C, H+2, W+2);
  initTensorPadded(host_I0);
  Tensor host_O = MakeHostTensor(K, W, H);


  // Initialize values in host A and B
  
  Convo(host_I0,host_O);

}
  

