#include "matrix2D.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#define min(a,b) (((a)<(b))?(a):(b))

#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef WITH_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include "omp.h"
#endif

Matrix *createMatrix(const int rows, const int columns)
{
  // Check Positive
  if (rows <= 0 || columns <= 0)
  {
    printf("You should input positive rows and columns!");
    return NULL;
  }

  // Allocate memory for rows and columns
  Matrix *m = (Matrix *)malloc(sizeof(Matrix));

  // Allocate memory for entries
  // m->pData = malloc(rows*columns*sizeof(float *));
  //  to allocate memories continuously
  m->pData = malloc(rows * sizeof(float *));
  for (int i = 0; i < rows; i++)
  {
    m->pData[i] = malloc(columns * sizeof(float));
  }
  m->rows = rows;
  m->columns = columns;
  return m;
}

// Random the entries value from 0 to 1
Matrix *createRandomMatrix(const int rows, const int columns)
{
  // Allocate Memory
  Matrix *m = createMatrix(rows, columns);
  srand((unsigned int)time(NULL));
  float a = 1;
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      m->pData[i][j] = (float)rand()/(float)(RAND_MAX/a);
    }
  }
  return m;
}

bool deleteMatrix(Matrix **pmatrix)
{
  for (int i = 0; i < (*pmatrix)->rows; i++)
  {
    // Free the entries
    free((*pmatrix)->pData[i]);
  }
  free((*pmatrix)->pData);
  free(*pmatrix);
  pmatrix = NULL;
  return true;
}

bool matmul_plain(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result)
{
  // Check
  if (matrix_one->columns != matrix_two->rows || matrix_one->rows != result->rows || matrix_two->columns != result->columns)
    return false;

  // Mul
  for (int i = 0; i < result->rows; i++)
  {
    for (int j = 0; j < result->columns; j++)
    {
      result->pData[i][j] = 0;
      for (int k = 0; k < matrix_one->columns; k++)
      {
        result->pData[i][j] += matrix_one->pData[i][k] * matrix_two->pData[k][j];
      }
    }
  }
  return true;
}

bool matmul_unloop(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result)
{
  // Check
  if (matrix_one->columns != matrix_two->rows || matrix_one->rows != result->rows || matrix_two->columns != result->columns)
    return false;

  for (int i = 0; i < result->rows; i++)
  {
    for (int j = 0; j < result->columns; j++)
    {
      result->pData[i][j] = 0;
      int remainder = result->columns % 8;
      for (int k = 0; k < matrix_one->columns / 8; k += 8)
      {
        result->pData[i][j] += matrix_one->pData[i][k] * matrix_two->pData[k][j];
        result->pData[i][j] += matrix_one->pData[i][k + 1] * matrix_two->pData[k + 1][j];
        result->pData[i][j] += matrix_one->pData[i][k + 2] * matrix_two->pData[k + 2][j];
        result->pData[i][j] += matrix_one->pData[i][k + 3] * matrix_two->pData[k + 3][j];
        result->pData[i][j] += matrix_one->pData[i][k + 4] * matrix_two->pData[k + 4][j];
        result->pData[i][j] += matrix_one->pData[i][k + 5] * matrix_two->pData[k + 5][j];
        result->pData[i][j] += matrix_one->pData[i][k + 6] * matrix_two->pData[k + 6][j];
        result->pData[i][j] += matrix_one->pData[i][k + 7] * matrix_two->pData[k + 7][j];
      }
      for (int k = matrix_one->columns - remainder; k < matrix_one->columns; k++)
      {
        result->pData[i][j] += matrix_one->pData[i][k] * matrix_two->pData[k][j];
      }
    }
  }
  return true;
}

bool matmul_avx2(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result)
{
#ifdef WITH_AVX2

  return true;
#else
  printf("AVX2 is not supported\n");
  return false;
#endif
}

bool matmul_avx2_omp(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

bool matmul_neon(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

bool matmul_neon_omp(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

void printMatrix(const Matrix *matrix)
{
  for(int i = 0; i < matrix->rows; i++)
  {
    printf("\t");
    for(int j = 0; j < matrix->columns; j++)
    {
      printf("%f\t", matrix-> pData[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

bool matmul_improved(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result)
{
  unsigned int i, j;           // Indexes
  unsigned int block_size = 8; //
  // Check
  if (matrix_one->columns != matrix_two->rows || matrix_one->rows != result->rows || matrix_two->columns != result->columns)
    return false;

#pragma opm parallel for
  for (i = 0; i < matrix_one->rows; i += block_size)
  {
    float a1, a2, a3, a4, a5, a6, a7, a8;
    float b1, b2, b3, b4, b5, b6, b7, b8;

    float tempData[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    unsigned int jj, ii, stopii, stopjj, p;

    stopii = min(i + block_size, matrix_one->rows); // the smaller of the two
    for (j = 0; j < matrix_two->columns; j += block_size)
    {

      stopjj = min(j + block_size, matrix_two->columns);

      for (ii = i; ii < stopii; ii++)
      {
        for (jj = j; jj < stopjj; jj++)
        {

          float sum = 0;
          __m256 sum_temp = _mm256_set1_ps(0.0);

          float *ka = matrix_one->pData[ii];
          float *kb = matrix_two->pData[jj];

          while (ka < matrix_one->pData[ii] + matrix_one->columns)
          {
            if (ka + 7 >= matrix_one->pData[ii] + matrix_one->columns)
            {
              for (int k = 0; k <= 7 && ka + k < matrix_one->pData[ii] + matrix_one->columns; ++k)
              {
                float left = (*(ka + k)) * (*(kb + k));
                sum += left;
              }
            }
            else
            {
              a1 = *(ka);
              a2 = *(ka + 1);
              a3 = *(ka + 2);
              a4 = *(ka + 3);
              a5 = *(ka + 4);
              a6 = *(ka + 5);
              a7 = *(ka + 6);
              a8 = *(ka + 7);

              b1 = *(kb);
              b2 = *(kb + 1);
              b3 = *(kb + 2);
              b4 = *(kb + 3);
              b5 = *(kb + 4);
              b6 = *(kb + 5);
              b7 = *(kb + 6);
              b8 = *(kb + 7);

              __m256 a = _mm256_set_ps(a1, a2, a3, a4, a5, a6, a7, a8);
              __m256 b = _mm256_set_ps(b1, b2, b3, b4, b5, b6, b7, b8);

              __m256 product = _mm256_mul_ps(a, b);

              sum_temp = _mm256_add_ps(sum_temp, product);
            }

            ka += 8;
            kb += 8;
          }

          _mm256_storeu_ps(&tempData[0], sum_temp);

          for (p = 0; p < 8; p++)
          {
            sum += tempData[p];
          }

          result->pData[ii][jj] = sum;
        }
      }
    }
    
  }
  return true;
}