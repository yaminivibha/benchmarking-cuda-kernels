

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < N) {
        C[tid] = A[tid] + B[tid];
        tid = tid + blockDim.x * gridDim.x;
    }
}
