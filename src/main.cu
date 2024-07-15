#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <tiffio.h>
#include <string>
#include <assert.h>
#include <iostream>

#include "parameters.h"

#include <cuda_runtime.h>
#include <cutensor.h>

using namespace std::chrono;

// Define the matrix size
#define MATRIX_HEIGTH 35
#define MATRIX_WIDTH 35

float *band1 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *band2 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *band3 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *band4 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *band5 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *band6 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *band7 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *band8 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

float *grenscale1 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *grenscale2 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *grenscale3 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *grenscale4 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *grenscale5 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *grenscale6 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *grenscale7 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *grenscale8 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

float *brescale1 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *brescale2 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *brescale3 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *brescale4 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *brescale5 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *brescale6 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *brescale7 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *brescale8 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

float *radiance1 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *radiance2 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *radiance3 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *radiance4 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *radiance5 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *radiance6 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *radiance7 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *radiance8 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

float *reflectance1 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *reflectance2 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *reflectance3 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *reflectance4 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *reflectance5 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *reflectance6 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *reflectance7 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
float *reflectance8 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

void serial(Sensor sensor, MTL mtl)
{
  int64_t general_time;
  system_clock::time_point begin, end;

  // ======== RUN =========
  begin = system_clock::now();

  for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  {
    radiance1[i] = band1[i] * sensor.parameters[1][sensor.GRESCALE] + sensor.parameters[1][sensor.BRESCALE];
    radiance2[i] = band2[i] * sensor.parameters[2][sensor.GRESCALE] + sensor.parameters[2][sensor.BRESCALE];
    radiance3[i] = band3[i] * sensor.parameters[3][sensor.GRESCALE] + sensor.parameters[3][sensor.BRESCALE];
    radiance4[i] = band4[i] * sensor.parameters[4][sensor.GRESCALE] + sensor.parameters[4][sensor.BRESCALE];
    radiance5[i] = band5[i] * sensor.parameters[5][sensor.GRESCALE] + sensor.parameters[5][sensor.BRESCALE];
    radiance6[i] = band6[i] * sensor.parameters[6][sensor.GRESCALE] + sensor.parameters[6][sensor.BRESCALE];
    radiance7[i] = band7[i] * sensor.parameters[7][sensor.GRESCALE] + sensor.parameters[7][sensor.BRESCALE];
    radiance8[i] = band8[i] * sensor.parameters[8][sensor.GRESCALE] + sensor.parameters[8][sensor.BRESCALE];
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUDA CORE," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;
  // ======== RUN =========

  // ======== RUN =========
  // begin = system_clock::now();

  // const float sin_sun = sin(mtl.sun_elevation * PI / 180);

  // for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  // {
  //   reflectance1[i] = radiance1[i] / sin_sun;
  //   reflectance2[i] = radiance2[i] / sin_sun;
  //   reflectance3[i] = radiance3[i] / sin_sun;
  //   reflectance4[i] = radiance4[i] / sin_sun;
  //   reflectance5[i] = radiance5[i] / sin_sun;
  //   reflectance6[i] = radiance6[i] / sin_sun;
  //   reflectance7[i] = radiance7[i] / sin_sun;
  //   reflectance8[i] = radiance8[i] / sin_sun;
  // }

  // end = system_clock::now();
  // general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  // std::cout << "CUDA CORE," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;
  // ======== RUN =========
}

void cutensor(Sensor sensor, MTL mtl)
{
  int64_t general_time;
  system_clock::time_point begin, end;

  // Define os eixos
  int tamanhoEixoA = 2;
  int tamanhoEixoB = 2;
  int tamanhoEixoC = 2;
  std::vector<int> indicesEixoA{'m', 'n'};
  std::vector<int> indicesEixoB{'m', 'n'};
  std::vector<int> indicesEixoC{'m', 'n'};
  std::vector<int64_t> dimensoesEixoA = {MATRIX_HEIGTH, MATRIX_WIDTH};
  std::vector<int64_t> dimensoesEixoB = {MATRIX_HEIGTH, MATRIX_WIDTH};
  std::vector<int64_t> dimensoesEixoC = {MATRIX_HEIGTH, MATRIX_WIDTH};

  // Aloca memória
  void *band1_d, *band2_d, *band3_d, *band4_d, *band5_d, *band6_d, *band7_d, *band8_d;
  void *brescale1_d, *brescale2_d, *brescale3_d, *brescale4_d, *brescale5_d, *brescale6_d, *brescale7_d, *brescale8_d;
  void *grenscale1_d, *grenscale2_d, *grenscale3_d, *grenscale4_d, *grenscale5_d, *grenscale6_d, *grenscale7_d, *grenscale8_d;
  void *radiance1_d, *radiance2_d, *radiance3_d, *radiance4_d, *radiance5_d, *radiance6_d, *radiance7_d, *radiance8_d;

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&band8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));

  // Alignment of the global-memory device pointers (bytes)
  const uint32_t kAlignment = 128;
  assert(uintptr_t(band1_d) % kAlignment == 0);
  assert(uintptr_t(band2_d) % kAlignment == 0);
  assert(uintptr_t(band3_d) % kAlignment == 0);
  assert(uintptr_t(band4_d) % kAlignment == 0);
  assert(uintptr_t(band5_d) % kAlignment == 0);
  assert(uintptr_t(band6_d) % kAlignment == 0);
  assert(uintptr_t(band7_d) % kAlignment == 0);
  assert(uintptr_t(band8_d) % kAlignment == 0);

  assert(uintptr_t(radiance1_d) % kAlignment == 0);
  assert(uintptr_t(radiance2_d) % kAlignment == 0);
  assert(uintptr_t(radiance3_d) % kAlignment == 0);
  assert(uintptr_t(radiance4_d) % kAlignment == 0);
  assert(uintptr_t(radiance5_d) % kAlignment == 0);
  assert(uintptr_t(radiance6_d) % kAlignment == 0);
  assert(uintptr_t(radiance7_d) % kAlignment == 0);
  assert(uintptr_t(radiance8_d) % kAlignment == 0);

  // Copia os dados para o dispositivo
  HANDLE_CUDA_ERROR(cudaMemcpy(band1_d, band1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band2_d, band2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band3_d, band3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band4_d, band4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band5_d, band5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band6_d, band6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band7_d, band7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band8_d, band8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));

  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale1_d, grenscale1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale2_d, grenscale2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale3_d, grenscale3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale4_d, grenscale4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale5_d, grenscale5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale6_d, grenscale6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale7_d, grenscale7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale8_d, grenscale8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));

  HANDLE_CUDA_ERROR(cudaMemcpy(brescale1_d, brescale1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale2_d, brescale2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale3_d, brescale3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale4_d, brescale4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale5_d, brescale5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale6_d, brescale6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale7_d, brescale7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale8_d, brescale8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));

  HANDLE_CUDA_ERROR(cudaMemcpy(radiance1_d, radiance1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance2_d, radiance2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance3_d, radiance3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance4_d, radiance4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance5_d, radiance5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance6_d, radiance6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance7_d, radiance7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance8_d, radiance8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));

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
  // C = alpha * OP(A) * OP(B) + beta * OP(C)
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
  // radiance_vector = 1 * bands_resampled * sensor.parameters + 1 * sensor.parameters[1][sensor.BRESCALE]
  cudaStream_t stream;
  HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

  typedef float floatTypeCompute;
  floatTypeCompute alpha = (floatTypeCompute)1;
  floatTypeCompute beta = (floatTypeCompute)1;

  begin = system_clock::now();

  // ==== Formula (contraction)
  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band1_d, grenscale1_d,
                                (void *)&beta, brescale1_d,
                                radiance1_d,
                                work,
                                actualWorkspaceSize,
                                stream));

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band2_d, grenscale2_d,
                                (void *)&beta, brescale2_d,
                                radiance2_d,
                                work,
                                actualWorkspaceSize,
                                stream));

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band3_d, grenscale3_d,
                                (void *)&beta, brescale3_d,
                                radiance3_d,
                                work,
                                actualWorkspaceSize,
                                stream));

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band4_d, grenscale4_d,
                                (void *)&beta, brescale4_d,
                                radiance4_d,
                                work,
                                actualWorkspaceSize,
                                stream));

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band5_d, grenscale5_d,
                                (void *)&beta, brescale5_d,
                                radiance5_d,
                                work,
                                actualWorkspaceSize,
                                stream));

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band6_d, grenscale6_d,
                                (void *)&beta, brescale6_d,
                                radiance6_d,
                                work,
                                actualWorkspaceSize,
                                stream));

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band7_d, grenscale7_d,
                                (void *)&beta, brescale7_d,
                                radiance7_d,
                                work,
                                actualWorkspaceSize,
                                stream)); 

  HANDLE_ERROR(cutensorContract(handle,
                                plan,
                                (void *)&alpha, band8_d, grenscale8_d,
                                (void *)&beta, brescale8_d,
                                radiance8_d,
                                work,
                                actualWorkspaceSize,
                                stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUTENSOR," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;

  // Copy data back to host
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance1, radiance1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance2, radiance2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance3, radiance3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance4, radiance4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance5, radiance5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance6, radiance6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance7, radiance7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance8, radiance8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cutensorDestroy(handle));
  HANDLE_ERROR(cutensorDestroyPlan(plan));
  HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
  HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
  TIFF *bands_resampled[8];

  // Sensor setup
  MTL mtl = MTL("./input/small/MTL.txt");
  Sensor sensor = Sensor(mtl.number_sensor, mtl.year);

  // TIFFs Setup
  std::string bands_paths[] = {
      "./input/small/B2.tif",
      "./input/small/B3.tif",
      "./input/small/B4.tif",
      "./input/small/B5.tif",
      "./input/small/B6.tif",
      "./input/small/B10.tif",
      "./input/small/B7.tif",
      "./input/small/final_tal.tif"};

  for (int i = 0; i < 8; i++)
  {
    std::string path_tiff_base = bands_paths[i];
    bands_resampled[i] = TIFFOpen(path_tiff_base.c_str(), "rm");
  }

  uint16_t sample_format;
  uint32_t height, width;
  TIFFGetField(bands_resampled[1], TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(bands_resampled[1], TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(bands_resampled[1], TIFFTAG_SAMPLEFORMAT, &sample_format);

  for (int i = 0; i < 8; i++)
  {
    for (int line = 0; line < height; line++)
    {
      TIFF *curr_band = bands_resampled[i];
      tdata_t band_line_buff = _TIFFmalloc(TIFFScanlineSize(curr_band));
      unsigned short curr_band_line_size = TIFFScanlineSize(curr_band) / width;
      TIFFReadScanline(curr_band, band_line_buff, line);

      for (int col = 0; col < width; col++)
      {
        float value = 0;
        memcpy(&value, static_cast<unsigned char *>(band_line_buff) + col * curr_band_line_size, curr_band_line_size);

        switch (i)
        {
        case 0:
          band1[line * width + col] = value;
          break;
        case 1:
          band2[line * width + col] = value;
          break;
        case 2:
          band3[line * width + col] = value;
          break;
        case 3:
          band4[line * width + col] = value;
          break;
        case 4:
          band5[line * width + col] = value;
          break;
        case 5:
          band6[line * width + col] = value;
          break;
        case 6:
          band7[line * width + col] = value;
          break;
        case 7:
          band8[line * width + col] = value;
          break;
        default:
            break;  
        }
      }
      _TIFFfree(band_line_buff);
    }
  }

  for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  {
    grenscale1[i] = sensor.parameters[1][sensor.GRESCALE];
    grenscale2[i] = sensor.parameters[2][sensor.GRESCALE];
    grenscale3[i] = sensor.parameters[3][sensor.GRESCALE];
    grenscale4[i] = sensor.parameters[4][sensor.GRESCALE];
    grenscale5[i] = sensor.parameters[5][sensor.GRESCALE];
    grenscale6[i] = sensor.parameters[6][sensor.GRESCALE];
    grenscale7[i] = sensor.parameters[7][sensor.GRESCALE];
    grenscale8[i] = sensor.parameters[8][sensor.GRESCALE];

    brescale1[i] = sensor.parameters[1][sensor.BRESCALE];
    brescale2[i] = sensor.parameters[2][sensor.BRESCALE];
    brescale3[i] = sensor.parameters[3][sensor.BRESCALE];
    brescale4[i] = sensor.parameters[4][sensor.BRESCALE];
    brescale5[i] = sensor.parameters[5][sensor.BRESCALE];
    brescale6[i] = sensor.parameters[6][sensor.BRESCALE];
    brescale7[i] = sensor.parameters[7][sensor.BRESCALE];
    brescale8[i] = sensor.parameters[8][sensor.BRESCALE];
  }

  // ======== RUN =========
  // serial(sensor, mtl);
  cutensor(sensor, mtl);
  // ======== RUN =========

  // Print radiance1
  for (int i = 0; i < width; i++)
  {
    for (int j = 0; j < height; j++)
    {
      std::cout << radiance1[j * width + i] << " ";
    }
    std::cout << std::endl;
  }

  for (int i = 0; i < 8; i++)
  {
    TIFFClose(bands_resampled[i]);
  }

  return 0;
}