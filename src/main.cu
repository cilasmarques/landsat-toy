#include <mma.h>
#include <iostream>
#include <cutensor.h>
#include <vector>
#include <cassert>

using namespace nvcuda;

#define DIM_SIZE 16 // 1024 * 90

cudaStream_t stream;
cutensorHandle_t handle;
cutensorTensorDescriptor_t descA;
cutensorTensorDescriptor_t descB;
cutensorTensorDescriptor_t descC;
cutensorPlan_t tensor_plan_binary_mul;


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

__global__ void wmma_kernel(half *mA, half *mB, float *mC, int width, int height)
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

__global__ void hadamard_kernel(half *mA, half *mB, float *mC, int width, int height)
{
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

void hadamard_tensor(int width, int height)
{
    cutensorCreate(&handle);
    cudaStreamCreate(&stream);

    int dim_num = 2;
    std::vector<int> axis{'m', 'n'};
    std::vector<int64_t> axis_dim = {height, width};

    const uint32_t kAlignment = 128;

    // Define descriptors
    cutensorCreateTensorDescriptor(handle, &descA, dim_num, axis_dim.data(), NULL, CUTENSOR_R_16F, kAlignment);
    cutensorCreateTensorDescriptor(handle, &descB, dim_num, axis_dim.data(), NULL, CUTENSOR_R_16F, kAlignment);
    cutensorCreateTensorDescriptor(handle, &descC, dim_num, axis_dim.data(), NULL, CUTENSOR_R_32F, kAlignment);

    // Create tensors
    cutensorOperationDescriptor_t desc;
    const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_16F;
    cutensorCreateElementwiseBinary(handle, &desc,
                                    descA, axis.data(), CUTENSOR_OP_IDENTITY,
                                    descB, axis.data(), CUTENSOR_OP_IDENTITY,
                                    descC, axis.data(),
                                    CUTENSOR_OP_MUL, descCompute);

    // Optional (but recommended): ensure that the scalar type is correct.
    cutensorDataType_t scalarType;
    cutensorOperationDescriptorGetAttribute(handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void *)&scalarType, sizeof(scalarType));
    assert(scalarType == CUTENSOR_R_16F);

    // Set the algorithm to use
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    cutensorPlanPreference_t planPref;
    cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE);

    // Query workspace estimate
    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    cutensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, &workspaceSizeEstimate);

    // Create Plan
    cutensorCreatePlan(handle, &tensor_plan_binary_mul, desc, planPref, workspaceSizeEstimate);
}

int main()
{
    cudaEvent_t start, stop;
    int width = DIM_SIZE;
    int height = DIM_SIZE;
    int blocks_num = (width * height + 32 - 1) / 32;

    float *d_c, *h_c;
    half *d_a, *h_a, *d_b, *h_b;
    h_a = (half *)malloc(height * width * sizeof(half));
    h_b = (half *)malloc(height * width * sizeof(half));
    h_c = (float *)malloc(height * width * sizeof(float));

    cudaMalloc(&d_a, height * width * sizeof(half));
    cudaMalloc(&d_b, height * width * sizeof(half));
    cudaMalloc(&d_c, height * width * sizeof(float));

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            h_a[i * width + j] = 1;
            h_b[i * width + j] = i;
        }
    }

    cudaMemcpy(d_a, h_a, height * width * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, height * width * sizeof(half), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    // wmma_tensor<<<blocks_num, 32>>>(d_a, d_b, d_c);
    // wmma_kernel<<<blocks_num, 32>>>(d_a, d_b, d_c, width, height);
    // hadamard_kernel<<<blocks_num, 32>>>(d_a, d_b, d_c, width, height);

    int pos1 = 1;
    int pos0 = 0;
    hadamard_tensor(width, height);
    cutensorElementwiseBinaryExecute(handle, tensor_plan_binary_mul, (void *)&pos1, d_a, (void *)&pos1, d_b, d_c, stream);

    cudaEventCreate(&stop);

    cudaMemcpy(h_c, d_c, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            std::cout << h_c[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    float milliseconds = 0;
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;
}
