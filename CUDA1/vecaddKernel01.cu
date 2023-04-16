__global__ void AddVectors(const float* A, const float* B, float* C, int ValuesPerThread)
{
    int N = blockIdx.x * blockDim.x * ValuesPerThread;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadEndIndex   = tid + N;
    while (tid < threadEndIndex) {
        C[tid] = A[tid] + B[tid];
        tid = tid + blockDim.x * gridDim.x;
    }
}
