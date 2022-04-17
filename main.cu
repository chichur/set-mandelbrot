#include <iostream>

void checkDevice();

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}


int main() {

    checkDevice();

    int N = 9;
    size_t size = sizeof(float) * N;

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size) ;
    float* h_C = (float*)malloc(size);

    float A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    memcpy(h_A, A, size);
    memcpy(h_B, B, size);

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    VecAdd<<<1, N>>>(d_A, d_B, d_C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float f = 1.0f;
    for (int i = 0; i < N; i++) {
        printf("%.0f %.0f %.0f\n", h_B[i], h_A[i], h_C[i]);
//        std::cout << h_A[i] << std::endl;
    }


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

void checkDevice() {
    printf("-------------- Driver information --------------\n");

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    int driver_version = 0, runtime_version = 0;

    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    printf("Driver Version: %d\n"
           "Runtime Version: %d\n",
           driver_version, runtime_version);
    printf("------------------------------------------------\n");
}
