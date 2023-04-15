

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}