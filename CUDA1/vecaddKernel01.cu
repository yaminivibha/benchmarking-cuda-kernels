
__global__ void AddVectors(const float *A, const float *B, float *C, int N) {
    int i = threadIdx.x +blockIdx.x * blockDim.x;

    while(i < N * gridDim.x * blockDim.x){
        C[i] = A[i] + B[i];
        i += gridDim.x * blockDim.x;
    }
}