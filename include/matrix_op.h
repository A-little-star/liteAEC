#ifndef MATRIX_OP_H
#define MATRIX_OP_H
// 矩阵运算

// 逐元素加法
void elementwise_add(float *a, float *b, float *out, int size);
// 逐元素乘法
void elementwise_mul(float *a, float *b, float *out, int size);
// 矩阵向量乘法
void matvec_mul(float *mat, float *vec, float *out, int rows, int cols);

#endif