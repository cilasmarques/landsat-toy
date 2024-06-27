/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>

#include <unordered_map>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cutensor.h>

using namespace std::chrono;

#define MATRIX_HEIGTH 128*64
#define MATRIX_WIDTH 128*64

#define HANDLE_ERROR(x)                                   \
  {                                                       \
    const auto err = x;                                   \
    if (err != CUTENSOR_STATUS_SUCCESS)                   \
    {                                                     \
      printf("Error: %s\n", cutensorGetErrorString(err)); \
      exit(-1);                                           \
    }                                                     \
  };

#define HANDLE_CUDA_ERROR(x)                          \
  {                                                   \
    const auto err = x;                               \
    if (err != cudaSuccess)                           \
    {                                                 \
      printf("Error: %s\n", cudaGetErrorString(err)); \
      exit(-1);                                       \
    }                                                 \
  };

struct GPUTimer
{
  GPUTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  ~GPUTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start()
  {
    cudaEventRecord(start_, 0);
  }

  float seconds()
  {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }

private:
  cudaEvent_t start_, stop_;
};

/**********************
 * Computing: C_{m,n} = alpha * A_{m,n} B_{m,n} + beta * C_{m,n}
 **********************/
int main()
{
  system_clock::time_point begin, end;
  int64_t general_time;

  // Define os eixos
  int tamanhoEixoA = 2;
  int tamanhoEixoB = 2;
  int tamanhoEixoC = 2;
  std::vector<int> indicesEixoA{'m', 'n'};
  // std::vector<int> indicesEixoB{'m', 'n'}; // Produto Escalar
  std::vector<int> indicesEixoB{'n', 'j'}; // Produto de hadamard
  std::vector<int> indicesEixoC{'m', 'n'};
  std::vector<int64_t> dimensoesEixoA = {MATRIX_HEIGTH, MATRIX_WIDTH};
  std::vector<int64_t> dimensoesEixoB = {MATRIX_HEIGTH, MATRIX_WIDTH};
  std::vector<int64_t> dimensoesEixoC = {MATRIX_HEIGTH, MATRIX_WIDTH};

  // Aloca memória
  void *A_d, *B_d, *C_d;
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&A_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&B_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&C_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));

  float *A = (float *)malloc(sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH);
  float *B = (float *)malloc(sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH);
  float *C = (float *)malloc(sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH);

  // Alignment of the global-memory device pointers (bytes)
  const uint32_t kAlignment = 128;
  assert(uintptr_t(A_d) % kAlignment == 0);
  assert(uintptr_t(B_d) % kAlignment == 0);
  assert(uintptr_t(C_d) % kAlignment == 0);

  // Inicializa a matrix
  for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  {
    A[i] = 4.0f;
    B[i] = 2.0f;
    C[i] = 0.0f;
  }

  // Copia os dados para o dispositivo
  HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));

  // Define o handle cutensor
  cutensorHandle_t handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  // Define os descriptors
  cutensorTensorDescriptor_t descA;
  cutensorTensorDescriptor_t descB;
  cutensorTensorDescriptor_t descC;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, tamanhoEixoA, dimensoesEixoA.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descB, tamanhoEixoB, dimensoesEixoB.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, tamanhoEixoC, dimensoesEixoC.data(), NULL, CUTENSOR_R_32F, kAlignment));

  // Create Contraction Descriptor
  cutensorOperationDescriptor_t desc;
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  HANDLE_ERROR(cutensorCreateContraction(handle,
                                         &desc,
                                         descA, indicesEixoA.data(), /* unary operator A*/ CUTENSOR_OP_IDENTITY,
                                         descB, indicesEixoB.data(), /* unary operator B*/ CUTENSOR_OP_IDENTITY,
                                         descC, indicesEixoC.data(), /* unary operator C*/ CUTENSOR_OP_IDENTITY,
                                         descC, indicesEixoC.data(),
                                         descCompute));

  // Optional (but recommended): ensure that the scalar type is correct.
  cutensorDataType_t scalarType;
  HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void *)&scalarType, sizeof(scalarType)));

  assert(scalarType == CUTENSOR_R_32F);
  typedef float floatTypeCompute;
  floatTypeCompute alpha = (floatTypeCompute)1;
  floatTypeCompute beta = (floatTypeCompute)0;

  // Set the algorithm to use
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  cutensorPlanPreference_t planPref;
  HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE));

  // Query workspace estimate
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, &workspaceSizeEstimate));

  // Create Contraction Plan
  cutensorPlan_t plan;
  HANDLE_ERROR(cutensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate));

  // Optional: query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  HANDLE_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize, sizeof(actualWorkspaceSize)));
  assert(actualWorkspaceSize <= workspaceSizeEstimate);

  // Define the workspace
  void *work = nullptr;
  if (actualWorkspaceSize > 0)
  {
    HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
    assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
  }

  // Execute
  cudaStream_t stream;
  HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

  begin = system_clock::now();

  HANDLE_ERROR(cutensorContract(handle, plan, (void *)&alpha, A_d, B_d, (void *)&beta, C_d, C_d, work, actualWorkspaceSize, stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUTENSOR," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;

  // Print the result
  float *C_aux = (float *)malloc(sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH);
  HANDLE_CUDA_ERROR(cudaMemcpy(C_aux, C_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));

  // Print the result
  // std::cout << "Matrix C: " << std::endl;
  // for (int i = 0; i < dimensoesEixoC[0]; i++)
  // {
  //   for (int j = 0; j < dimensoesEixoC[1]; j++)
  //   {
  //     std::cout << C_aux[i * dimensoesEixoC[1] + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  HANDLE_ERROR(cutensorDestroy(handle));
  HANDLE_ERROR(cutensorDestroyPlan(plan));
  HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
  HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

  if (A)
    free(A);
  if (B)
    free(B);
  if (C)
    free(C);
  if (A_d)
    cudaFree(A_d);
  if (B_d)
    cudaFree(B_d);
  if (C_d)
    cudaFree(C_d);
  if (work)
    cudaFree(work);

  return 0;
}