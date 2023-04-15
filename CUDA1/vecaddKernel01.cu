

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        C[tid] = A[tid] + B[tid];
        tid += blockDim.x * gridDim.x;
    }
}