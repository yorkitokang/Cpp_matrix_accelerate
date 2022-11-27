#ifndef MATRIX_2D_H
#define MATRIX_2D_H
#include <stddef.h>
#include <stdbool.h>

typedef struct
{
    int rows;
    int columns;
    float **pData;
}Matrix;

Matrix *createMatrix(const int rows,const int columns);

Matrix *createRandomMatrix(const int rows,const int columns);

bool deleteMatrix(Matrix **pmatrix);

bool matmul_plain(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

bool matmul_unloop(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

bool matmul_avx2(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

bool matmul_avx2_omp(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

bool matmul_neon(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

bool matmul_neon_omp(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

void printMatrix(const Matrix *matrix);

bool matmul_improved(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);

#endif