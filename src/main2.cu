#include <mma.h>
#include <iostream>

using namespace nvcuda;

#define DIM_SIZE 1024 * 90 // 1024 * 60

__global__ void wmma_tensor(half *mA, half *mB, float *mC)
{
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, mA, 16);
    wmma::load_matrix_sync(b_frag, mB, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(mC, c_frag, 16, wmma::mem_row_major);
}

__global__ void wmma_kernel(half *mA, half *mB, float *mC, long width, long height)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Map 1D position to 2D grid
    unsigned int row = idx / width;
    unsigned int col = idx % width;

    if (idx < width * height)
    {
        unsigned int pos = row * width + col;
        for (int i = 0; i < width; i++)
            mC[pos] += __half2float(mA[row * width + i]) * __half2float(mB[i * width + col]);
    }
}

__global__ void hadamard_kernel(half *mA, half *mB, float *mC, long width, long height) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Map 1D position to 2D grid
    unsigned int row = idx / width;
    unsigned int col = idx % width;

    if (idx < width * height)
    {
        unsigned int pos = row * width + col;
        mC[pos] = __half2float(mA[pos]) * __half2float(mB[pos]);
    }
}

int main()
{
    cudaEvent_t start, stop;
    long width = DIM_SIZE;
    long height = DIM_SIZE;
    int blocks_num = (width * height + 32 - 1) / 32;

    float *d_c, *h_c;
    half *d_a, *h_a, *d_b, *h_b;
    h_a = (half *)malloc(height * width * sizeof(half));
    h_b = (half *)malloc(height * width * sizeof(half));
    h_c = (float *)malloc(height * width * sizeof(float));

    cudaMalloc(&d_a, height * width * sizeof(half));
    cudaMalloc(&d_b, height * width * sizeof(half));
    cudaMalloc(&d_c, height * width * sizeof(float));

    for (long i = 0; i < height; i++) {
        for (long j = 0; j < width; j++) {
            h_a[i * width + j] = 1;
            h_b[i * width + j] = __float2half(static_cast<float>(i));
        }
    }

    cudaMemcpy(d_a, h_a, height * width * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, height * width * sizeof(half), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    // wmma_tensor<<<blocks_num, 32>>>(d_a, d_b, d_c);
    // wmma_kernel<<<blocks_num, 32>>>(d_a, d_b, d_c, width, height);
    hadamard_kernel<<<blocks_num, 32>>>(d_a, d_b, d_c, width, height);
    cudaEventCreate(&stop);

    cudaMemcpy(h_c, d_c, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < height; i++) {
    //     for (int j = 0; j < width; j++) {
    //         std::cout << h_c[i * width + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    float milliseconds = 0;
    cudaEventRecord(start);
    cudaEventSynchronize(start);    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;
}

