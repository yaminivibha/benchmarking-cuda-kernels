#include <stdio.h>

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadEndIndex   = tid + N;
    while (tid < threadEndIndex) {
        C[tid] = A[tid] + B[tid];
        tid = tid + blockDim.x * gridDim.x;
    }
}
