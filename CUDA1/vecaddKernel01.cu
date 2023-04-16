

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x * N);
    int threadEndIndex   = threadStartIndex + N;
    int i;
    int stride = blockDim.x;
    
    for (i = threadStartIndex; i < threadEndIndex; i += stride) {
        C[i] = A[i] + B[i];
    }
}