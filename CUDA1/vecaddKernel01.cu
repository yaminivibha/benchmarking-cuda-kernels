

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        C[tid] = A[tid] + B[tid];
    }
}
