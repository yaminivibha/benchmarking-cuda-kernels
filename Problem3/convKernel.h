/// matmultKernel.h
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-19 DVN
/// Yamini Ananth
/// Kernels defined with this header must 
/// CUDA: A x B = C where ABC are matrices
///

#ifndef __MMKERNEL__
#define __MMKERNEL__

// Defines for size of thread block and data computed by a thread block
#define BLOCK_SIZE 16
#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// Creating a struct to hold the matrix data
typedef struct {
  int channels;
  int height;
  int width;
  int stride_channel;
  int stride_height;
  double* elements;
} Matrix;

// Forward declaration of the kernel function that performs the work.
__global__ void ConvKernel(const Matrix, const Matrix*, Matrix);

#endif

