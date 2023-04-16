__global__ void vectorSum(const float *A, const float *B, float *C, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int chunk_size = blockDim.x;
    int chunk_id = blockIdx.x;

    for (int i = chunk_id * chunk_size + threadIdx.x; i < size; i += chunk_size * gridDim.x) {
        C[i] = A[i] + B[i];
    }
}

