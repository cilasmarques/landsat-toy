#include <mma.h>
#include <iostream>

using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Linear warp index
    // int warpId = blockIdx.x * blockDim.x + threadIdx.x;
    // int warpM = (warpId / ((N + WMMA_N - 1) / WMMA_N));
    // int warpN = (warpId % ((N + WMMA_N - 1) / WMMA_N));

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);


    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K)
    {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N)
    {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
        for (int i = 0; i < c_frag.num_elements; i++)
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

__global__ void wmma_ker(half *a, half *b, float *c)
{
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

int main()
{
    float *d_c, *h_c;
    half *d_a, *h_a, *d_b, *h_b;
    h_c = new float[16 * 16];
    h_b = new half[16 * 16];
    h_a = new half[16 * 16];

    cudaMalloc(&d_a, 16 * 16 * sizeof(half));
    cudaMalloc(&d_b, 16 * 16 * sizeof(half));
    cudaMalloc(&d_c, 16 * 16 * sizeof(float));

    for (int i = 0; i < 16 * 16; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    cudaMemcpy(d_a, h_a, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);

    // wmma_ker<<<8, 32>>>(d_a, d_b, d_c);
    wmma_example<<<8, 32>>>(d_a, d_b, d_c, 16, 16, 16, 2.0f, 0.0f);

    cudaMemcpy(h_c, d_c, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16 * 16; i++)
        std::cout << h_c[i] << ",";
    std::cout << std::endl;
}