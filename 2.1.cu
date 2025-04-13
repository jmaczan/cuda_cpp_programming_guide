#include <stdio.h>

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    float *A, *B, *C;
    int N = 1000000;
    size_t size = N * sizeof(float);

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    cudaDeviceSynchronize();
    printf("C[1] = %f\n", C[1]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}