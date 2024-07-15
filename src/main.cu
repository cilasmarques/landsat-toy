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
#define MATRIX_HEIGTH 6502
#define MATRIX_WIDTH 6502 // 7295

float *all1 = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

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

struct cutensorStruct
{
  cutensorPlan_t plan;
  void *work;
  uint64_t actualWorkspaceSize;
};

cutensorStruct createPlanWork(cutensorHandle_t handle, cutensorOperationDescriptor_t desc)
{
  // Optional (but recommended): ensure that the scalar type is correct.
  cutensorDataType_t scalarType;
  HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void *)&scalarType, sizeof(scalarType)));
  assert(scalarType == CUTENSOR_R_32F);

  // Set the algorithm to use
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  cutensorPlanPreference_t planPrefContraction;
  HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPrefContraction, algo, CUTENSOR_JIT_MODE_NONE));

  // Query workspace estimate
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, desc, planPrefContraction, workspacePref, &workspaceSizeEstimate));

  // Create Contraction Plan
  cutensorPlan_t plan;
  HANDLE_ERROR(cutensorCreatePlan(handle, &plan, desc, planPrefContraction, workspaceSizeEstimate));

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

  return {plan, work, actualWorkspaceSize};
}

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
  begin = system_clock::now();
  const float sin_sun = sin(mtl.sun_elevation * PI / 180);

  for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  {
    reflectance1[i] = radiance1[i] / sin_sun;
    reflectance2[i] = radiance2[i] / sin_sun;
    reflectance3[i] = radiance3[i] / sin_sun;
    reflectance4[i] = radiance4[i] / sin_sun;
    reflectance5[i] = radiance5[i] / sin_sun;
    reflectance6[i] = radiance6[i] / sin_sun;
    reflectance7[i] = radiance7[i] / sin_sun;
    reflectance8[i] = radiance8[i] / sin_sun;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUDA CORE," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;
  // ======== RUN =========
}

void cutensor(Sensor sensor, MTL mtl)
{
  float alpha, beta;
  int64_t general_time;
  system_clock::time_point begin, end;

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
  std::cout << "Alocou 1" << std::endl;

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&brescale8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  std::cout << "Alocou 2" << std::endl;

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&grenscale8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  std::cout << "Alocou 3" << std::endl;

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&radiance8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  std::cout << "Alocou 4" << std::endl;

  // Copia os dados para o dispositivo
  HANDLE_CUDA_ERROR(cudaMemcpy(band1_d, band1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band2_d, band2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band3_d, band3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band4_d, band4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band5_d, band5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band6_d, band6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band7_d, band7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(band8_d, band8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  std::cout << "Copiou 1" << std::endl;

  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale1_d, grenscale1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale2_d, grenscale2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale3_d, grenscale3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale4_d, grenscale4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale5_d, grenscale5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale6_d, grenscale6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale7_d, grenscale7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(grenscale8_d, grenscale8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  std::cout << "Copiou 2" << std::endl;

  HANDLE_CUDA_ERROR(cudaMemcpy(brescale1_d, brescale1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale2_d, brescale2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale3_d, brescale3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale4_d, brescale4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale5_d, brescale5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale6_d, brescale6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale7_d, brescale7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(brescale8_d, brescale8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  std::cout << "Copiou 3" << std::endl;

  HANDLE_CUDA_ERROR(cudaMemcpy(radiance1_d, radiance1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance2_d, radiance2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance3_d, radiance3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance4_d, radiance4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance5_d, radiance5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance6_d, radiance6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance7_d, radiance7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance8_d, radiance8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  std::cout << "Copiou 4" << std::endl;

  // Define os eixos
  int tamanhoEixo = 2;
  std::vector<int> indicesEixo{'m', 'n'};
  std::vector<int64_t> dimensoesEixo = {MATRIX_HEIGTH, MATRIX_WIDTH};

  // Define o handle cutensor
  cutensorHandle_t handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  // Define a stream
  cudaStream_t stream;
  HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

  // Alignment of the global-memory device pointers (bytes)
  const uint32_t kAlignment = 128;

  // Define os descriptors
  cutensorTensorDescriptor_t descA;
  cutensorTensorDescriptor_t descB;
  cutensorTensorDescriptor_t descC;
  cutensorTensorDescriptor_t descD;
  cutensorTensorDescriptor_t descE;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, tamanhoEixo, dimensoesEixo.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descB, tamanhoEixo, dimensoesEixo.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, tamanhoEixo, dimensoesEixo.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descD, tamanhoEixo, dimensoesEixo.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descE, tamanhoEixo, dimensoesEixo.data(), NULL, CUTENSOR_R_32F, kAlignment));

  // Contraction Descriptors
  // C = alpha * OP(A) * OP(B) + beta * OP(C)
  cutensorOperationDescriptor_t descContraction;
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  HANDLE_ERROR(cutensorCreateContraction(handle,
                                         &descContraction,
                                         descA, indicesEixo.data(), /* unary operator A*/ CUTENSOR_OP_IDENTITY,
                                         descB, indicesEixo.data(), /* unary operator B*/ CUTENSOR_OP_IDENTITY,
                                         descC, indicesEixo.data(), /* unary operator C*/ CUTENSOR_OP_IDENTITY,
                                         descC, indicesEixo.data(),
                                         descCompute));
  cutensorStruct planContraction = createPlanWork(handle, descContraction);

  // Permutation Descriptors
  // B = alpha * OP(A)
  // cutensorOperationDescriptor_t descDivision;
  // HANDLE_ERROR(cutensorCreatePermutation(handle, &descDivision, descD, indicesEixo.data(), CUTENSOR_OP_IDENTITY, descE, indicesEixo.data(), descCompute));
  // cutensorStruct planDivision = createPlanWork(handle, descDivision);

  // ====== RUN RADIANCE ======
  // radiance_vector = 1 * bands_resampled * sensor.parameters[sensor.GRENSCALE] + 1 * sensor.parameters[sensor.BRESCALE]
  alpha = 1; beta = 1;
  begin = system_clock::now();

  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band1_d, grenscale1_d, (void *)&beta, brescale1_d, radiance1_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band2_d, grenscale2_d, (void *)&beta, brescale2_d, radiance2_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band3_d, grenscale3_d, (void *)&beta, brescale3_d, radiance3_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band4_d, grenscale4_d, (void *)&beta, brescale4_d, radiance4_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band5_d, grenscale5_d, (void *)&beta, brescale5_d, radiance5_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band6_d, grenscale6_d, (void *)&beta, brescale6_d, radiance6_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band7_d, grenscale7_d, (void *)&beta, brescale7_d, radiance7_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, band8_d, grenscale8_d, (void *)&beta, brescale8_d, radiance8_d, planContraction.work, planContraction.actualWorkspaceSize, stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUTENSOR," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;
  // ====== RUN RADIANCE ======

  // Copy data back to host
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance1, radiance1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance2, radiance2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance3, radiance3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance4, radiance4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance5, radiance5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance6, radiance6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance7, radiance7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(radiance8, radiance8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));

  // Free device memory
  HANDLE_CUDA_ERROR(cudaFree(band1_d));
  HANDLE_CUDA_ERROR(cudaFree(band2_d));
  HANDLE_CUDA_ERROR(cudaFree(band3_d));
  HANDLE_CUDA_ERROR(cudaFree(band4_d));
  HANDLE_CUDA_ERROR(cudaFree(band5_d));
  HANDLE_CUDA_ERROR(cudaFree(band6_d));
  HANDLE_CUDA_ERROR(cudaFree(band7_d));
  HANDLE_CUDA_ERROR(cudaFree(band8_d));

  HANDLE_CUDA_ERROR(cudaFree(grenscale1_d));
  HANDLE_CUDA_ERROR(cudaFree(grenscale2_d));
  HANDLE_CUDA_ERROR(cudaFree(grenscale3_d));
  HANDLE_CUDA_ERROR(cudaFree(grenscale4_d));
  HANDLE_CUDA_ERROR(cudaFree(grenscale5_d));
  HANDLE_CUDA_ERROR(cudaFree(grenscale6_d));
  HANDLE_CUDA_ERROR(cudaFree(grenscale7_d));
  HANDLE_CUDA_ERROR(cudaFree(grenscale8_d));

  HANDLE_CUDA_ERROR(cudaFree(brescale1_d));
  HANDLE_CUDA_ERROR(cudaFree(brescale2_d));
  HANDLE_CUDA_ERROR(cudaFree(brescale3_d));
  HANDLE_CUDA_ERROR(cudaFree(brescale4_d));
  HANDLE_CUDA_ERROR(cudaFree(brescale5_d));
  HANDLE_CUDA_ERROR(cudaFree(brescale6_d));
  HANDLE_CUDA_ERROR(cudaFree(brescale7_d));
  HANDLE_CUDA_ERROR(cudaFree(brescale8_d));
  std::cout << "Desaloca memoria rad" << std::endl;

  // Alocate memory
  void *all1_d;
  void *reflectance1_d, *reflectance2_d, *reflectance3_d, *reflectance4_d, *reflectance5_d, *reflectance6_d, *reflectance7_d, *reflectance8_d;

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&all1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&reflectance8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH));
  std::cout << "Alocou 5" << std::endl;

  HANDLE_CUDA_ERROR(cudaMemcpy(all1_d, all1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance1_d, reflectance1, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance2_d, reflectance2, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance3_d, reflectance3, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance4_d, reflectance4, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance5_d, reflectance5, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance6_d, reflectance6, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance7_d, reflectance7, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance8_d, reflectance8, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyHostToDevice));
  std::cout << "Copiou 5" << std::endl;

  // ====== RUN REFLECTANCE ======
  // radiance_vector = 1 * bands_resampled * sensor.parameters[sensor.GRENSCALE] + 1 * sensor.parameters[sensor.BRESCALE]
  alpha = 1 / sin(mtl.sun_elevation * PI / 180);
  beta = 0;
  begin = system_clock::now();

  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance1_d, all1_d, (void *)&beta, all1_d, reflectance1_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance2_d, all1_d, (void *)&beta, all1_d, reflectance2_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance3_d, all1_d, (void *)&beta, all1_d, reflectance3_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance4_d, all1_d, (void *)&beta, all1_d, reflectance4_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance5_d, all1_d, (void *)&beta, all1_d, reflectance5_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance6_d, all1_d, (void *)&beta, all1_d, reflectance6_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance7_d, all1_d, (void *)&beta, all1_d, reflectance7_d, planContraction.work, planContraction.actualWorkspaceSize, stream));
  HANDLE_ERROR(cutensorContract(handle, planContraction.plan, (void *)&alpha, radiance8_d, all1_d, (void *)&beta, all1_d, reflectance8_d, planContraction.work, planContraction.actualWorkspaceSize, stream));

  // reflectance = alpha * radiance
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance1_d, reflectance1_d, stream));
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance2_d, reflectance2_d, stream));
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance3_d, reflectance3_d, stream));
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance4_d, reflectance4_d, stream));
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance5_d, reflectance5_d, stream));
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance6_d, reflectance6_d, stream));
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance7_d, reflectance7_d, stream));
  // HANDLE_ERROR(cutensorPermute(handle, planDivision.plan, (void *)&alpha, radiance8_d, reflectance8_d, stream));

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUTENSOR," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;
  // ====== RUN REFLECTANCE ======

  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance1, reflectance1_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance2, reflectance2_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance3, reflectance3_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance4, reflectance4_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance5, reflectance5_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance6, reflectance6_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance7, reflectance7_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(reflectance8, reflectance8_d, sizeof(float) * MATRIX_HEIGTH * MATRIX_WIDTH, cudaMemcpyDeviceToHost));

  // Free device memory
  HANDLE_CUDA_ERROR(cudaFree(all1_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance1_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance2_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance3_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance4_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance5_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance6_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance7_d));
  HANDLE_CUDA_ERROR(cudaFree(reflectance8_d));
  std::cout << "Desaloca memoria ref" << std::endl;

  // Free cutensor resources
  HANDLE_ERROR(cutensorDestroy(handle));
  HANDLE_ERROR(cutensorDestroyPlan(planContraction.plan));
  HANDLE_ERROR(cutensorDestroyOperationDescriptor(descContraction));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
  HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
  TIFF *bands_resampled[8];

  // Sensor setup
  MTL mtl = MTL("./input/final_results/MTL.txt");
  Sensor sensor = Sensor(mtl.number_sensor, mtl.year);

  // TIFFs Setup
  std::string bands_paths[] = {
      "./input/final_results/B2.TIF",
      "./input/final_results/B3.TIF",
      "./input/final_results/B4.TIF",
      "./input/final_results/B5.TIF",
      "./input/final_results/B6.TIF",
      "./input/final_results/B10.TIF",
      "./input/final_results/B7.TIF",
      "./input/final_results/elevation.tif"};

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
    for (int line = 0; line < MATRIX_HEIGTH; line++)
    {
      TIFF *curr_band = bands_resampled[i];
      tdata_t band_line_buff = _TIFFmalloc(TIFFScanlineSize(curr_band));
      unsigned short curr_band_line_size = TIFFScanlineSize(curr_band) / width;
      TIFFReadScanline(curr_band, band_line_buff, line);

      for (int col = 0; col < MATRIX_WIDTH; col++)
      {
        float value = 0;
        memcpy(&value, static_cast<unsigned char *>(band_line_buff) + col * curr_band_line_size, curr_band_line_size);

        switch (i)
        {
        case 0:
          band1[line * MATRIX_WIDTH + col] = value;
          break;
        case 1:
          band2[line * MATRIX_WIDTH + col] = value;
          break;
        case 2:
          band3[line * MATRIX_WIDTH + col] = value;
          break;
        case 3:
          band4[line * MATRIX_WIDTH + col] = value;
          break;
        case 4:
          band5[line * MATRIX_WIDTH + col] = value;
          break;
        case 5:
          band6[line * MATRIX_WIDTH + col] = value;
          break;
        case 6:
          band7[line * MATRIX_WIDTH + col] = value;
          break;
        case 7:
          band8[line * MATRIX_WIDTH + col] = value;
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

    all1[i] = 1.0;
  }

  // ======== RUN =========
  // serial(sensor, mtl);
  cutensor(sensor, mtl);
  // ======== RUN =========

  // Print reflectance1
  // for (int i = 0; i < width; i++)
  // {
  //   for (int j = 0; j < height; j++)
  //   {
  //     std::cout << reflectance1[j * width + i] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  for (int i = 0; i < 8; i++)
  {
    TIFFClose(bands_resampled[i]);
  }

  return 0;
}