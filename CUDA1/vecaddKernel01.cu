#include <stdio.h>

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadEndIndex   = threadStartIndex + N;
    printf("tid = %d\n", tid);
    while (tid < threadEndIndex) {
        C[tid] = A[tid] + B[tid];
        printf("C[tid] = %f\n", C[tid]);
        printf("A[tid] = %f\n", A[tid]);
        printf("B[tid] = %f\n", B[tid]); 
        tid = tid + blockDim.x * gridDim.x;
        printf("next tid = %d\n", tid);
    }
}
