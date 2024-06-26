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

int main()
{
  system_clock::time_point begin, end;
  int64_t general_time;

  typedef float floatTypeA;
  typedef float floatTypeB;
  typedef float floatTypeC;

  /**********************
   * Computing: C_{m,n} = alpha * A_{m,n} B_{m,n} + beta * C_{m,n}
   **********************/

  // Define o tamanho de cada dimensão
  std::vector<int64_t> extentA = {10000, 10000};
  std::vector<int64_t> extentB = {10000, 10000};
  std::vector<int64_t> extentC = {10000, 10000};

  // Define a quantidade de dimensões dos tensores
  int nmodeA = 2;
  int nmodeB = 2;
  int nmodeC = 2;

  /**********************
   * Allocating data
   **********************/
  size_t elementsA = extentA[0] * extentA[1];
  size_t elementsB = extentB[0] * extentB[1];
  size_t elementsC = extentC[0] * extentC[1];

  size_t sizeA = sizeof(floatTypeA) * elementsA;
  size_t sizeB = sizeof(floatTypeB) * elementsB;
  size_t sizeC = sizeof(floatTypeC) * elementsC;
  printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC) / 1024. / 1024. / 1024);

  void *A_d, *B_d, *C_d;
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&A_d, sizeA));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&B_d, sizeB));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&C_d, sizeC));

  floatTypeA *A = (floatTypeA *)malloc(sizeof(floatTypeA) * elementsA);
  floatTypeB *B = (floatTypeB *)malloc(sizeof(floatTypeB) * elementsB);
  floatTypeC *C = (floatTypeC *)malloc(sizeof(floatTypeC) * elementsC);

  if (A == NULL || B == NULL || C == NULL)
  {
    printf("Error: Host allocation of A or C.\n");
    return -1;
  }

  /*******************
   * Initialize data
   *******************/

  for (int64_t i = 0; i < elementsA; i++)
    A[i] = 4;
  for (int64_t i = 0; i < elementsB; i++)
    B[i] = 2;
  for (int64_t i = 0; i < elementsC; i++)
    C[i] = 0;

  // print A
  printf("A size: %ld\n", sizeA);

  HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

  const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
  assert(uintptr_t(A_d) % kAlignment == 0);
  assert(uintptr_t(B_d) % kAlignment == 0);
  assert(uintptr_t(C_d) % kAlignment == 0);

  /*************************
   * cuTENSOR
   *************************/

  cutensorHandle_t handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  /**********************
   * Create Tensor Descriptors
   **********************/

  cutensorTensorDescriptor_t descA;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                              &descA,
                                              nmodeA,
                                              extentA.data(),
                                              NULL, /*stride*/
                                              CUTENSOR_R_32F, kAlignment));

  cutensorTensorDescriptor_t descB;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                              &descB,
                                              nmodeB,
                                              extentB.data(),
                                              NULL, /*stride*/
                                              CUTENSOR_R_32F, kAlignment));

  cutensorTensorDescriptor_t descC;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                              &descC,
                                              nmodeC,
                                              extentC.data(),
                                              NULL, /*stride*/
                                              CUTENSOR_R_32F, kAlignment));

  /*******************************
   * Create Contraction Descriptor
   *******************************/
  std::vector<int> modeC{'m', 'n'};
  std::vector<int> modeA{'m', 'n'};
  std::vector<int> modeB{'m', 'n'};

  cutensorOperationDescriptor_t desc;
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  HANDLE_ERROR(cutensorCreateContraction(handle,
                                         &desc,
                                         descA, modeA.data(), /* unary operator A*/ CUTENSOR_OP_IDENTITY,
                                         descB, modeB.data(), /* unary operator B*/ CUTENSOR_OP_IDENTITY,
                                         descC, modeC.data(), /* unary operator C*/ CUTENSOR_OP_IDENTITY,
                                         descC, modeC.data(),
                                         descCompute));

  /*****************************
   * Optional (but recommended): ensure that the scalar type is correct.
   *****************************/

  cutensorDataType_t scalarType;
  HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                                                       desc,
                                                       CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                       (void *)&scalarType,
                                                       sizeof(scalarType)));

  assert(scalarType == CUTENSOR_R_32F);
  typedef float floatTypeCompute;
  floatTypeCompute alpha = (floatTypeCompute)1;
  floatTypeCompute beta = (floatTypeCompute)0;

  /**************************
   * Set the algorithm to use
   ***************************/

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  HANDLE_ERROR(cutensorCreatePlanPreference(
      handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));

  /**********************
   * Query workspace estimate
   **********************/

  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                                             desc,
                                             planPref,
                                             workspacePref,
                                             &workspaceSizeEstimate));

  /**************************
   * Create Contraction Plan
   **************************/

  cutensorPlan_t plan;
  HANDLE_ERROR(cutensorCreatePlan(handle,
                                  &plan,
                                  desc,
                                  planPref,
                                  workspaceSizeEstimate));

  /**************************
   * Optional: Query information about the created plan
   **************************/

  // query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                                        plan,
                                        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                        &actualWorkspaceSize,
                                        sizeof(actualWorkspaceSize)));

  // At this point the user knows exactly how much memory is need by the operation and
  // only the smaller actual workspace needs to be allocated
  assert(actualWorkspaceSize <= workspaceSizeEstimate);

  void *work = nullptr;
  if (actualWorkspaceSize > 0)
  {
    HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
    assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
  }

  /**********************
   * Run
   **********************/

  cudaStream_t stream;
  HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

  begin = system_clock::now();

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, A_d, B_d,
                                (void *)&beta, C_d, C_d,
                                work, actualWorkspaceSize, stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUTENSOR CORE - TOTAL TIME (ns): " << general_time << std::endl;

  // // print the result in a aux variable
  // floatTypeC *C_aux = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

  // // Copy the result back to the host
  // HANDLE_CUDA_ERROR(cudaMemcpy(C_aux, C_d, sizeC, cudaMemcpyDeviceToHost));

  // // Print the result
  // std::cout << "Matrix C: " << std::endl;
  // for (int i = 0; i < extentC[0]; i++)
  // {
  //     for (int j = 0; j < extentC[1]; j++)
  //     {
  //         std::cout << C_aux[i * extentC[1] + j] << " ";
  //     }
  //     std::cout << std::endl;
  // }

  /*************************/

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