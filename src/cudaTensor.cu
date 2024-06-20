#include <iostream>
#include <cutensor.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <assert.h>

using namespace std::chrono;

// Define the matrix size
#define MATRIX_HEIGTH 32768
#define MATRIX_WIDTH 32768

int main()
{
  system_clock::time_point begin, end;
  int64_t initial_time, final_time, general_time;

  // Create a matrix
  float *matrixA = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
  float *matrixB = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
  std::cout << "Matrix size: " << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << std::endl;

  // Initialize the matrix
  for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  {
    matrixA[i] = 2.0f;
    matrixB[i] = 2.0f;
  }

  // Define the handle
  cutensorHandle_t handle;
  cutensorCreate(&handle);

  // Create the tensor descriptors
  int quantidades_de_dimensoes = 2;
  int64_t tamanho_de_cada_dimensao[] = {MATRIX_HEIGTH, MATRIX_WIDTH};
  cutensorTensorDescriptor_t descA, descB, descC;
  cutensorStatus_t status;
  status = cutensorCreateTensorDescriptor(handle, &descA, quantidades_de_dimensoes, tamanho_de_cada_dimensao, NULL, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
  status = cutensorCreateTensorDescriptor(handle, &descB, quantidades_de_dimensoes, tamanho_de_cada_dimensao, NULL, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);

  // Allocate the memory on the device
  float *d_matrixA, *d_matrixB, *d_matrixC;
  cudaMalloc(&d_matrixA, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float));
  cudaMalloc(&d_matrixB, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float));
  cudaMalloc(&d_matrixC, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float));

  // Copy the data to the device
  cudaMemcpy(d_matrixA, matrixA, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrixB, matrixB, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

  // Create the tensor operations
  const int32_t eixosA[] = {MATRIX_HEIGTH, MATRIX_WIDTH};
  const int32_t eixosB[] = {MATRIX_HEIGTH, MATRIX_WIDTH};
  const int32_t eixosC[] = {MATRIX_HEIGTH, MATRIX_WIDTH};

  cutensorOperationDescriptor_t desc;
  cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  cutensorCreateContraction(
      handle,
      &desc,
      descA, eixosA, CUTENSOR_OP_IDENTITY, // Indica que não há transformação adicional em A antes da operação binária.
      descB, eixosB, CUTENSOR_OP_IDENTITY, // Indica que não há transformação adicional em B antes da operação binária.
      NULL, NULL, CUTENSOR_OP_IDENTITY,    // Indica que não há transformação adicional em C antes da operação binária.
      descC, eixosC, descCompute);

  // Ensure that the scalar type is correct.
  cutensorDataType_t scalarType;
  cutensorOperationDescriptorGetAttribute(handle,
                                          desc,
                                          CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                          (void *)&scalarType,
                                          sizeof(scalarType));

  assert(scalarType == CUTENSOR_R_32F);

  // Define scaling factors
  typedef float floatTypeCompute;
  floatTypeCompute alpha = (floatTypeCompute)1.1f;
  floatTypeCompute beta = (floatTypeCompute)0.f;

  // Set the algorithm to use
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  cutensorPlanPreference_t planPref;
  cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE);

  // Estimate the workspace size
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  cutensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, &workspaceSizeEstimate);

  // Create the plan
  cutensorPlan_t plan;
  cutensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate);

  // Optional: Query information about the created plan
  uint64_t actualWorkspaceSize = 0;
  cutensorPlanGetAttribute(handle,plan,CUTENSOR_PLAN_REQUIRED_WORKSPACE,&actualWorkspaceSize,sizeof(actualWorkspaceSize));
  assert(actualWorkspaceSize <= workspaceSizeEstimate);

  void *work = nullptr;
  if (actualWorkspaceSize > 0)
  {
    cudaMalloc(&work, actualWorkspaceSize);
    assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Execute the operation
  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cutensorContract(handle, plan, (void *)&alpha, d_matrixA, d_matrixB, (void *)&beta, d_matrixC, d_matrixC, work, actualWorkspaceSize, stream);

  end = system_clock::now();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUBLAS CORE - TOTAL TIME (ns): " << general_time << std::endl;
  // ======== RUN CUBLAS TENSOR =========

  // Copy the result back to the host
  float *matrixC = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
  cudaMemcpy(matrixC, d_matrixC, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the result
  // std::cout << "Matrix C: " << std::endl;
  // for (int i = 0; i < MATRIX_HEIGTH; i++)
  // {
  //   for (int j = 0; j < MATRIX_WIDTH; j++)
  //   {
  //     std::cout << matrixC[i * MATRIX_WIDTH + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // Free the memory
  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);

  return 0;
}