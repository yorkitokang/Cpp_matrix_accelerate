#include "matrix1D.h"
#include <stdio.h>
#include <time.h>
#include <sys/utsname.h>
#include <string.h>
#include <unistd.h>
#include <cblas.h>
#include <math.h>
#include <omp.h>
#define _SYS_NAMELEN 256
#define MATRIX_SIZE 8000
#define  MAGNIFY 1

int main()
{
  Matrix* m3_1 = createRandomMatrix(3,3,MAGNIFY);
  sleep(1);
  Matrix* m3_2 = createRandomMatrix(3,3,MAGNIFY);
  Matrix* m3_result_1 = createMatrix(3,3);
  Matrix* m3_result_2 = createMatrix(3,3);
  Matrix* m3_result_3 = createMatrix(3,3);
  matmul_plain(m3_1, m3_2, m3_result_1);
  matmul_improved(m3_1, m3_2, m3_result_2);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,3,3,3,MAGNIFY,m3_1->pData, 3, m3_2->pData, 3,0,m3_result_3->pData,3);
  printMatrix(m3_1);
  printMatrix(m3_2);
  printMatrix(m3_result_1);
  printMatrix(m3_result_2);
  printMatrix(m3_result_3);
  struct utsname uname_pointer;
  uname(&uname_pointer);
  printf("---------------SYSTEM INFORMATION-----------------\n");
  printf("System name - %s \n", uname_pointer.sysname);
  printf("Nodename    - %s \n", uname_pointer.nodename);
  printf("Release     - %s \n", uname_pointer.release);
  printf("Version     - %s \n", uname_pointer.version);
  printf("Machine     - %s \n", uname_pointer.machine);
  clock_t begin = clock();
  printf("---------------BENCHMARKING---------------\n");

  printf("Creating Matrices...\n");
  double begin_1 = omp_get_wtime();
  Matrix *m_1 = createRandomMatrix(MATRIX_SIZE,MATRIX_SIZE,MAGNIFY);
  sleep(1);
  Matrix *m_2 = createRandomMatrix(MATRIX_SIZE,MATRIX_SIZE,MAGNIFY);
  Matrix *m_result_1 = createMatrix(MATRIX_SIZE,MATRIX_SIZE);
  Matrix *m_result_2 = createMatrix(MATRIX_SIZE,MATRIX_SIZE);
  Matrix *m_result_3 = createMatrix(MATRIX_SIZE,MATRIX_SIZE);
  double end_1 = omp_get_wtime();
  double time_spent_1 = end_1 - begin_1;
  printf("Used %f seconds to generate size %d Matrices\n",time_spent_1, MATRIX_SIZE);

  // double begin_2 = omp_get_wtime();
  // matmul_plain(m_1,m_2,m_result_1);
  // double end_2 = omp_get_wtime();
  // time_spent_1 = end_2 - begin_2;
  // printf("Used %f seconds to plain mul size %d Matrices\n",time_spent_1, MATRIX_SIZE);

  // omp_set_num_threads(24);
  double begin_3 = omp_get_wtime();
  matmul_improved(m_1,m_2,m_result_2);
  double end_3 = omp_get_wtime();
  time_spent_1 = end_3 - begin_3;
  printf("Used %f seconds to optimized mul size %d Matrices\n",time_spent_1, MATRIX_SIZE);

  double begin_4 = omp_get_wtime();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,MATRIX_SIZE,MATRIX_SIZE,MATRIX_SIZE,MAGNIFY,m_1->pData, MATRIX_SIZE, m_2->pData, MATRIX_SIZE,0,m_result_3->pData,MATRIX_SIZE);
  double end_4 = omp_get_wtime();
  double time_spent_4 = end_4 - begin_4;
  printf("Used %f seconds to do open blas %d mul\n",time_spent_4, MATRIX_SIZE);

  float sum = 0;
  for(size_t i = 0; i < m_result_2->rows; i++)
  {
    for(size_t j = 0; j < m_result_3->columns; j++)
    {
      sum += fabs(m_result_3->pData[i*m_result_3->columns+j]-m_result_2->pData[i*m_result_3->columns+j]);
    }
  }
  printf("Error is %f", sum);
  return 0;
}