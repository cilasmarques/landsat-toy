#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

using namespace std::chrono;

// Define the matrix size
#define MATRIX_HEIGTH 128*64
#define MATRIX_WIDTH 128*64

void multiplyPartial(float *matrixA, float *matrixB, float *matrixC, int startRow, int endRow)
{
  for (int row = startRow; row < endRow; ++row)
  {
    for (int col = 0; col < MATRIX_WIDTH; ++col)
    {
      for (int i = 0; i < MATRIX_WIDTH; ++i)
      {
        matrixC[row * MATRIX_WIDTH + col] += matrixA[row * MATRIX_WIDTH + i] * matrixB[i * MATRIX_WIDTH + col];
      }
    }
  }
}

int main()
{
  system_clock::time_point begin, end;
  int64_t general_time;

  // Create a matrix
  float *matrixA = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
  float *matrixB = new float[MATRIX_HEIGTH * MATRIX_WIDTH];
  float *matrixC = new float[MATRIX_HEIGTH * MATRIX_WIDTH];

  // Initialize the matrix
  for (int i = 0; i < MATRIX_HEIGTH * MATRIX_WIDTH; i++)
  {
    matrixA[i] = 4.0f;
    matrixB[i] = 2.0f;
  }

  // ======== RUN SERIAL =========
  begin = system_clock::now();

  // Make a scalar product
  for (int row = 0; row < MATRIX_HEIGTH; row++)
    for (int col = 0; col < MATRIX_WIDTH; col++)
      for (int i = 0; i < MATRIX_WIDTH; i++)
        matrixC[row * MATRIX_WIDTH + col] += matrixA[row * MATRIX_WIDTH + i] * matrixB[i * MATRIX_WIDTH + col];

  // std::thread threads[8];
  // int numThreads = 8;
  // int rowsPerThread = MATRIX_HEIGTH / numThreads;

  // for (int t = 0; t < numThreads; ++t)
  // {
  //   int startRow = t * rowsPerThread;
  //   int endRow = (t == numThreads - 1) ? MATRIX_HEIGTH : startRow + rowsPerThread;
  //   threads[t] = std::thread(multiplyPartial, matrixA, matrixB, matrixC, startRow, endRow);
  // }

  // for (int t = 0; t < numThreads; ++t)
  // {
  //   threads[t].join();
  // }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUDA CORE," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;
  // ======== RUN SERIAL =========

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

  return 0;
}