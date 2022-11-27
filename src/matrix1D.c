#include "matrix1D.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//#define min(a,b) (((a)<(b))?(a):(b))

#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef WITH_NEON
#include <arm_neon.h>
#endif

Matrix *createMatrix(const size_t rows, const size_t columns)
{
  Matrix *m = (Matrix *)malloc(sizeof(Matrix));

  // Allocate memory for entries
  m->pData = malloc(rows * columns * sizeof(float *));
  // to allocate memories continuously
  m->rows = rows;
  m->columns = columns;
  return m;
}

Matrix *createRandomMatrix(const size_t rows, const size_t columns, const size_t size)
{
  // Allocate Memory
  Matrix *m = createMatrix(rows, columns);

  // Random
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      m->pData[i * columns + j] = 0 + size * 1.0 * rand() / RAND_MAX * (1 - 0);
    }
  }
  return m;
}

bool deleteMatrix(Matrix **pmatrix)
{
  if (pmatrix == NULL)
  {
    fprintf(stderr, "File %s, Line %d : The pointer to the matrix pointer is NULL", __FILE__, __LINE__);
    return false;
  }
  else if (*pmatrix == NULL)
  {
    fprintf(stderr, "File %s, Line %d : The matrix pointer is null", __FILE__, __LINE__);
    return false;
  }
  else if ((*pmatrix)->pData == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The data pointer of the matrix is null", __FILE__, __LINE__);
    return false;
  }

  free((*pmatrix)->pData);
  free(*pmatrix);
  pmatrix = NULL;
  return true;
}

bool matmul_plain(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result)
{

  // Check
  if (matrix_one == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The first parameter is null", __FILE__, __LINE__);
    return false;
  }
  else if (matrix_one->pData == NULL)
  {
    fprintf(stderr, "File %s, Line %d The data of the first parameter is null", __FILE__, __LINE__);
    return false;
  }

  if (matrix_two == NULL)
  {
    fprintf(stderr, "File %s, Line %d The first parameter is null", __FILE__, __LINE__);
    return false;
  }
  else if (matrix_two->pData == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The data of the first parameter is null", __FILE__, __LINE__);
    return false;
  }

  if (result == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The first parameter is null", __FILE__, __LINE__);
    return false;
  }
  else if (result->pData == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The data of the first parameter is null", __FILE__, __LINE__);
    return false;
  }

  if (matrix_one->columns != matrix_two->rows || matrix_one->rows != result->rows || matrix_two->columns != result->columns)
  {
    fprintf(stderr, "File %s, Line %d : The matrices connot do the multiplication", __FILE__, __LINE__);
    return false;
  }

  // Mul

  for (int i = 0; i < result->rows; i++)
  {
    for (int j = 0; j < result->columns; j++)
    {
      result->pData[i * (result->columns) + j] = 0;
      for (int k = 0; k < matrix_one->columns; k++)
      {
        result->pData[i * (result->columns) + j] += matrix_one->pData[i * (result->columns) + k] * matrix_two->pData[k * (result->columns) + j];
      }
    }
  }
  return true;
}

void printMatrix(const Matrix *matrix)
{
  for (int i = 0; i < matrix->rows; i++)
  {
    printf("\t");
    for (int j = 0; j < matrix->columns; j++)
    {
      printf("%f\t", matrix->pData[i * matrix->columns + j]);
    }
    printf("\n");
  }
  printf("\n");
}

bool matmul_improved(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result)
{

  size_t i, j;           // Indexes
  size_t block_size = 8; //

  // Check
  if (matrix_one == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The first parameter is null", __FILE__, __LINE__);
    return false;
  }
  else if (matrix_one->pData == NULL)
  {
    fprintf(stderr, "File %s, Line %d The data of the first parameter is null", __FILE__, __LINE__);
    return false;
  }

  if (matrix_two == NULL)
  {
    fprintf(stderr, "File %s, Line %d The first parameter is null", __FILE__, __LINE__);
    return false;
  }
  else if (matrix_two->pData == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The data of the first parameter is null", __FILE__, __LINE__);
    return false;
  }

  if (result == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The first parameter is null", __FILE__, __LINE__);
    return false;
  }
  else if (result->pData == NULL)
  {
    fprintf(stderr, "File %s, Line %d: The data of the first parameter is null", __FILE__, __LINE__);
    return false;
  }

  if (matrix_one->columns != matrix_two->rows || matrix_one->rows != result->rows || matrix_two->columns != result->columns)
  {
    fprintf(stderr, "File %s, Line %d : The matrices connot do the multiplication", __FILE__, __LINE__);
    return false;
  }
#ifdef WITH_AVX2
  // #pragma omp parallel for
  //   for (i = 0; i < matrix_one->rows; i += block_size)
  //   {
  //     //temp variable
  //     float a1, a2, a3, a4, a5, a6, a7, a8;
  //     float b1, b2, b3, b4, b5, b6, b7, b8;
  //     //temp string
  //     float tempData[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  //     //temp index
  //     size_t jj, ii, stopii, stopjj, p;

  //     stopii = min(i + block_size, matrix_one->rows);
  //     for (j = 0; j < matrix_two->columns; j += block_size)
  //     {

  //       stopjj = min(j + block_size, matrix_two->columns);
  //       for (ii = i; ii < stopii; ii++)
  //       {
  //         for (jj = j; jj < stopjj; jj++)
  //         {

  //           float sum = 0;
  //           __m256 sum_temp = _mm256_set1_ps(0.0);

  //           float *ka = &(matrix_one->pData[ii*(matrix_two->columns)]);
  //           float *kb = &(matrix_two->pData[jj]);

  //           while (ka < &(matrix_one->pData[ii*(matrix_two->columns)]) + matrix_one->columns)
  //           {
  //             if (ka + 7 >= &(matrix_one->pData[ii*(matrix_two->columns)]) + matrix_one->columns)
  //             {
  //               for (int k = 0; k <= 7 && ka + k < ka + matrix_one->columns; ++k)
  //               {
  //                 float left = (*(ka + k)) * (*(kb + k*matrix_two->columns));
  //                 sum += left;
  //               }
  //             }
  //             else
  //             {
  //               a1 = *(ka);
  //               a2 = *(ka + 1);
  //               a3 = *(ka + 2);
  //               a4 = *(ka + 3);
  //               a5 = *(ka + 4);
  //               a6 = *(ka + 5);
  //               a7 = *(ka + 6);
  //               a8 = *(ka + 7);

  //               b1 = *(kb);
  //               b2 = *(kb + 1*matrix_two->columns);
  //               b3 = *(kb + 2*matrix_two->columns);
  //               b4 = *(kb + 3*matrix_two->columns);
  //               b5 = *(kb + 4*matrix_two->columns);
  //               b6 = *(kb + 5*matrix_two->columns);
  //               b7 = *(kb + 6*matrix_two->columns);
  //               b8 = *(kb + 7*matrix_two->columns);

  //               __m256 a = _mm256_set_ps(a1, a2, a3, a4, a5, a6, a7, a8);
  //               __m256 b = _mm256_set_ps(b1, b2, b3, b4, b5, b6, b7, b8);

  //               __m256 product = _mm256_mul_ps(a, b);

  //               sum_temp = _mm256_add_ps(sum_temp, product);
  //             }

  //             ka += 8;
  //             kb += 8;
  //           }

  //           _mm256_storeu_ps(tempData, sum_temp);

  //           for (p = 0; p < 8; p++)
  //           {
  //             sum += tempData[p];
  //           }

  //           result->pData[ii*matrix_two->columns+jj] = sum;
  //         }
  //       }
  //     }
  //   }
  size_t jj, ii, p;
#pragma omp parallel for
  for (ii = 0; ii < result->rows; ii++)
  {
    for (jj = 0; jj < result->columns; jj++)
    {
      // temp variable
      float a1, a2, a3, a4, a5, a6, a7, a8;
      float b1, b2, b3, b4, b5, b6, b7, b8;
      // temp string
      float tempData[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      // temp index
      float sum = 0;
      __m256 sum_temp = _mm256_setzero_ps();

      float *ka = &(matrix_one->pData[ii * (matrix_two->columns)]);
      float *kb = &(matrix_two->pData[jj]);

      while (ka < &(matrix_one->pData[ii * (matrix_two->columns)]) + matrix_one->columns)
      {
        if (ka + 7 >= &(matrix_one->pData[ii * (matrix_two->columns)]) + matrix_one->columns)
        {
          for (int k = 0; k <= 7 && k < matrix_one->columns; k++)
          {
            float left = (*(ka + k)) * (*(kb + k * matrix_two->columns));
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
          b2 = *(kb + 1 * matrix_two->columns);
          b3 = *(kb + 2 * matrix_two->columns);
          b4 = *(kb + 3 * matrix_two->columns);
          b5 = *(kb + 4 * matrix_two->columns);
          b6 = *(kb + 5 * matrix_two->columns);
          b7 = *(kb + 6 * matrix_two->columns);
          b8 = *(kb + 7 * matrix_two->columns);

          __m256 a = _mm256_set_ps(a1, a2, a3, a4, a5, a6, a7, a8);
          __m256 b = _mm256_set_ps(b1, b2, b3, b4, b5, b6, b7, b8);

          __m256 product = _mm256_mul_ps(a, b);

          sum_temp = _mm256_add_ps(sum_temp, product);
        }

        ka += 8;
        kb += 8;
      }

      _mm256_storeu_ps(tempData, sum_temp);

      for (p = 0; p < 8; p++)
      {
        sum += tempData[p];
      }

      result->pData[ii * matrix_two->columns + jj] = sum;
    }
  }
  return true;
#endif

#ifdef WITH_NEON
#pragma omp parallel for
  for (ii = 0; ii < result->rows; ii++)
  {
    for (jj = 0; jj < result->columns; jj++)
    {
      // temp variable
      float a1, a2, a3, a4;
      float b1, b2, b3, b4;
      // temp string
      float tempData[4] = {0, 0, 0, 0};
      // temp index
      float sum = 0;
      float32x4_t sum_temp = vdupq_n_f32(0);

      float *ka = &(matrix_one->pData[ii * (matrix_two->columns)]);
      float *kb = &(matrix_two->pData[jj]);

      while (ka < &(matrix_one->pData[ii * (matrix_two->columns)]) + matrix_one->columns)
      {
        if (ka + 3 >= &(matrix_one->pData[ii * (matrix_two->columns)]) + matrix_one->columns)
        {
          for (int k = 0; k <= 3 && k < matrix_one->columns; ++k)
          {
            float left = (*(ka + k)) * (*(kb + k * matrix_two->columns));
            sum += left;
          }
        }
        else
        {

          float32x4_t a = vld1q_f32_x4(ka);
          float32x4_t b = vld1q_f32_x4(kb);

          float32x4_t product = vmulq_f32(a, b);

          sum_temp = _mm256_add_ps(sum_temp, product);
        }

        ka += 4;
        kb += 4;
      }

      vst1q_f32(tempData, sum_temp);

      for (p = 0; p < 4; p++)
      {
        sum += tempData[p];
      }

      result->pData[ii * matrix_two->columns + jj] = sum;
    }
  }
  return true;
#endif
}