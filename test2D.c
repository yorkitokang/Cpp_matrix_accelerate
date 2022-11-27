#include "matrix1D.h"
#include <stdio.h>
#include <time.h>
#include <sys/utsname.h>
#include <string.h>
#include <cblas.h>
#define _SYS_NAMELEN    256

int main()
{
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

  //create Matrices
  printf("Creating Matrices...\n");
  clock_t begin_1 = clock();
  Matrix *m16_1 = createRandomMatrix(16,16);
  Matrix *m16_2 = createRandomMatrix(16,16);
  Matrix *m16_result1 = createMatrix(16,16);
  Matrix *m16_result2 = createMatrix(16,16);
  printf("Matrices 16 created\n");
  Matrix *m128_1 = createRandomMatrix(256,256);
  Matrix *m128_2 = createRandomMatrix(256,256);
  Matrix *m128_result = createMatrix(256,256);
  printf("Matrices 256 created\n");
  Matrix *m1k_1 = createRandomMatrix(1024,1024);
  Matrix1 *m1k_1_1D = createRandomMatrix1(1024,1024);
  Matrix *m1k_2 = createRandomMatrix(1024,1024);
  Matrix1 *m1k_2_1D = createRandomMatrix1(1024,1024);
  Matrix *m1k_result = createRandomMatrix(1024,1024);
  Matrix1 *m1k_result_1D = createRandomMatrix1(1024,1024);
  printf("Matrices 1024 created\n");
  Matrix *m8k_1 = createRandomMatrix(8096,8096);
  Matrix1 *m8k_1_1D = createRandomMatrix1(8096,8096);
  Matrix *m8k_2 = createRandomMatrix(8096,8096);
  Matrix1 *m8k_2_1D = createRandomMatrix1(8096,8096);
  Matrix *m8k_result = createMatrix(8096,8096);
  Matrix1 *m8k_result_1D = createRandomMatrix1(8096,8096);
  printf("Matrices 8096 created\n");
  // Matrix *m64k_1 = createRandomMatrix(65536,65536);
  // Matrix *m64k_2 = createRandomMatrix(65536,65536);
  // Matrix *m64k_result = createMatrix(65536,65536);
  // printf("Matrices 65536 created\n");
  clock_t end_1 = clock();
  double time_spent_1 = (double)(end_1 - begin_1) / CLOCKS_PER_SEC;
  printf("Used %f seconds to generate Matrices\n",time_spent_1);

  //Benchmarking
  if(!strcmp(uname_pointer.machine,"x86_64"))
  {
    printf("This is a %s machine \nUsing AVX2\n", "x86_64");

    /*
    3 check correctness
    */
    Matrix* m3_1 = createRandomMatrix(3,3);
    Matrix* m3_2 = createRandomMatrix(3,3);
    Matrix* m3_result_1 = createMatrix(3,3);
    Matrix* m3_result_2 = createMatrix(3,3);
    matmul_plain(m3_1, m3_2, m3_result_1);
    matmul_improved(m3_1, m3_2, m3_result_2);
    printMatrix(m3_1);
    printMatrix(m3_2);
    printMatrix(m3_result_1);
    printMatrix(m3_result_2);

    /*
    16 check correctness
    */
    printf("Check the correctness of the algorithms\n");
    matmul_plain(m16_1,m16_2,m16_result1);
    matmul_improved(m16_1,m16_2,m16_result2);
    printMatrix(m16_result1);
    printMatrix(m16_result2);

    /*
      1k benchmark
    */
    clock_t begin_1k_plain = clock();
    printf("Doing 2D 1k plain multiplication\n");
    matmul_plain(m1k_1, m1k_2, m1k_result);
    clock_t end_1k_plain = clock();
    printMatrix(m1k_result);
    double time_spent_m1k_plain = (double)(end_1k_plain - begin_1k_plain) / CLOCKS_PER_SEC;
    printf("Done 2D 1k plain mul, Using %f seconds\n", time_spent_m1k_plain);

    clock_t begin_1k_improved = clock();
    printf("Doing 2D 1k improved multiplication\n");
    matmul_improved(m1k_1, m1k_2, m1k_result);
    clock_t end_1k_improved = clock();
    printMatrix(m1k_result);
    double time_spent_m1k_improved = (double)(end_1k_improved - begin_1k_improved) / CLOCKS_PER_SEC;
    printf("Done 2D 1k improved mul, Using %f seconds\n", time_spent_m1k_improved);

    clock_t begin_1k_openblas = clock();
    printf("Doing 1D 1k openblas multiplication\n");
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,1024,1024,1024,1,m1k_1_1D->pData, 1024, m1k_2_1D, 1024,1,m1k_result_1D,1024);
    clock_t end_1k_openblas = clock();
    double time_spent_m1k_openblas = (double)(end_1k_openblas - begin_1k_openblas) / CLOCKS_PER_SEC;
    printf("Done 1D 1k openblas mul, Using %f seconds\n", time_spent_m1k_openblas);
    
    //Delete matrix
    deleteMatrix(&m1k_1);
    deleteMatrix(&m1k_2);
    deleteMatrix(&m1k_result);
    /*
      8k benchmark
    */
    // clock_t begin_8k_plain = clock();
    // printf("Doing 2D 8k plain multiplication\n");
    // matmul_plain(m8k_1, m8k_2, m8k_result);
    // clock_t end_8k_plain = clock();
    // double time_spent_m8k_plain = (double)(end_8k_plain - begin_8k_plain) / CLOCKS_PER_SEC;
    // printf("Done 2D 8k plain mul, Using %f seconds\n", time_spent_m8k_plain);

    clock_t begin_8k_improved = clock();
    printf("Doing 2D 8k improved multiplication\n");
    matmul_improved(m8k_1, m8k_2, m8k_result);
    printMatrix(m8k_result);
    clock_t end_8k_improved = clock();
    double time_spent_m8k_improved = (double)(end_8k_improved - begin_8k_improved) / CLOCKS_PER_SEC;
    printf("Done 2D 8k improved mul, Using %f seconds\n", time_spent_m8k_improved);

    clock_t begin_8k_openblas = clock();
    printf("Doing 1D 8k openblas multiplication\n");
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,8096,8096,8096,1,m8k_1_1D->pData, 8096, 8096, 8096,1,m8k_result_1D,8096);
    clock_t end_8k_openblas = clock();
    double time_spent_m8k_openblas = (double)(end_8k_openblas - begin_8k_openblas) / CLOCKS_PER_SEC;
    printf("Done 1D 8k openblas mul, Using %f seconds\n", time_spent_m8k_openblas);


    // clock_t begin_8k_plain = clock();
    // printf("Doing 8k plain multiplication\n");
    // matmul_plain(m8k_1, m8k_2, m8k_result);
    // clock_t end_8k_plain = clock();
    // double time_spent_m8k_plain = (double)(end_8k_plain - begin_8k_plain) / CLOCKS_PER_SEC;
    // printf("Done 8k plain mul, Using %f seconds", time_spent_m8k_plain);

    // clock_t begin_8k_unloop = clock();
    // printf("Doing 8k unloop multiplication\n");
    // matmul_plain(m8k_1, m8k_2, m8k_result);
    // clock_t end_8k_unloop = clock();
    // double time_spent_m8k_plain = (double)(end_8k_plain - begin_8k_plain) / CLOCKS_PER_SEC;
    // printf("Done 8k plain mul, Using %f seconds", time_spent_m8k_plain);
  }
  
}