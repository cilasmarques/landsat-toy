#include <iostream>
#include <cutensor.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

using namespace std::chrono;

// Define the matrix size
#define MATRIX_HEIGTH 4
#define MATRIX_WIDTH 4

__global__ void matrixMn(float *d_matrixA, float *d_matrixB, float *d_matrixC)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int row = idx / MATRIX_WIDTH;
  unsigned int col = idx % MATRIX_WIDTH;

  if (row < MATRIX_HEIGTH && col < MATRIX_WIDTH)
  {
    //// Produto Escalar
    // for (int i = 0; i < MATRIX_WIDTH; i++)
    // {
    //   d_matrixC[row * MATRIX_WIDTH + col] += d_matrixA[row * MATRIX_WIDTH + i] * d_matrixB[i * MATRIX_WIDTH + col];
    // }

    ///// Produto de hadamard
    unsigned int pos = row * MATRIX_WIDTH + col;
    d_matrixC[pos] = d_matrixA[pos] * d_matrixB[pos];
  }
}

void cudaCoreExecution(float *d_matrixA, float *d_matrixB, float *d_matrixC)
{
  // Perform matrix sum: C = A * B
  int num_threads = 4;
  int num_blocks = ceil(MATRIX_HEIGTH * MATRIX_WIDTH / num_threads);
  matrixMn<<<num_blocks, num_threads>>>(d_matrixA, d_matrixB, d_matrixC);
}

int main()
{
  system_clock::time_point begin, end;
  int64_t general_time;

  // Create a matrix
  float *matrixA = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
  float *matrixB = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

  // Initialize the matrix
  for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  {
    matrixA[i] = 4.0f;
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

  // ======== RUN CUDA CORE =========
  begin = system_clock::now();

  cudaCoreExecution(d_matrixA, d_matrixB, d_matrixC);
  cudaDeviceSynchronize();

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUDA CORE - TOTAL TIME (ns): " << general_time << std::endl;
  // ======== RUN CUDA CORE =========

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