#ifndef MATRIX_H
#define MATRIX_H
#include <stddef.h>
#include <stdbool.h>

typedef struct
{
  size_t rows;
  size_t columns;
  float *pData;
}Matrix;

Matrix *createMatrix(const size_t rows,const size_t columns);
Matrix *createRandomMatrix(const size_t rows,const size_t columns,const size_t size);

bool deleteMatrix(Matrix **pmatrix);
bool matmul_plain(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);
bool matmul_improved(const Matrix *matrix_one, const Matrix *matrix_two, Matrix *result);
void printMatrix(const Matrix *matrix);

#endif