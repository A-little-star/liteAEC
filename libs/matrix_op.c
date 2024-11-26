#include "../include/matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 逐元素加法
void elementwise_add(float *a, float *b, float *out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

// 逐元素乘法
void elementwise_mul(float *a, float *b, float *out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

// 矩阵向量乘法
void matvec_mul(float *mat, float *vec, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = 0;
        for (int j = 0; j < cols; j++) {
            out[i] += mat[i * cols + j] * vec[j];
        }
    }
}