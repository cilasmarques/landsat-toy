#include <iostream>
#include <cutensor.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

using namespace std::chrono;

// Define the matrix size
#define MATRIX_HEIGTH 32768
#define MATRIX_WIDTH 32768

void cublasTensorExecution(cublasHandle_t cubaslHandle, float *d_matrixA, float *d_matrixB, float *d_matrixC)
{
  // Perform matrix sum: C = A / B
  float alpha = 1.0f;
  float beta = 1.0f;
  cublasStatus_t cublasStat = cublasSgeam(cubaslHandle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_HEIGTH, MATRIX_WIDTH, &alpha, d_matrixA, MATRIX_HEIGTH, &beta, d_matrixB, MATRIX_HEIGTH, d_matrixC, MATRIX_HEIGTH);
}

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

  // Allocate the memory on the device
  float *d_matrixA;
  float *d_matrixB;
  float *d_matrixC;
  cudaMalloc(&d_matrixA, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float));
  cudaMalloc(&d_matrixB, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float));
  cudaMalloc(&d_matrixC, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float));

  // Copy the data to the device
  cudaMemcpy(d_matrixA, matrixA, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrixB, matrixB, MATRIX_HEIGTH * MATRIX_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

  // ======== RUN CUBLAS TENSOR =========
  // Define the handle
  cublasHandle_t cubaslHandle;
  cublasStatus_t cublasStat = cublasCreate(&cubaslHandle);

  // Set the math mode to allow cuBLAS to use Tensor Cores:
  cublasStat = cublasSetMathMode(cubaslHandle, CUBLAS_TENSOR_OP_MATH);

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  cublasTensorExecution(cubaslHandle, d_matrixA, d_matrixB, d_matrixC);

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