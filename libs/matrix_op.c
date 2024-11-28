#include "../include/matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 创建一个新的张量
Tensor create_tensor(int c, int t, int f) {
    Tensor tensor;
    tensor.C = c;
    tensor.T = t;
    tensor.F = f;
    tensor.data = (float *)malloc(c * t * f * sizeof(float));
    if (tensor.data == NULL) {
        fprintf(stderr, "Failed to allocate memory for output\n");
        exit(EXIT_FAILURE); // 确保程序不会继续运行
    }
    return tensor;
}

// 释放张量所占的内存空间
void delete_tensor(Tensor tensor) {
    free(tensor.data);
    tensor.data = NULL; // 避免悬挂指针
}

// 初始化张量数据
void init_tensor(Tensor* tensor, float* values) {
    if (tensor == NULL || tensor->data == NULL || values == NULL) {
        fprintf(stderr, "Invalid tensor or values\n");
        return;
    }
    int total_elements = tensor->C * tensor->T * tensor->F;
    for (int i = 0; i < total_elements; i++) {
        tensor->data[i] = values[i];
    }
}

// 打印张量数据
void print_tensor(const Tensor* tensor) {
    if (tensor == NULL || tensor->data == NULL) {
        fprintf(stderr, "Invalid tensor\n");
        return;
    }
    for (int c = 0; c < tensor->C; c++) {
        printf("Channel %d:\n", c + 1);
        for (int t = 0; t < tensor->T; t++) {
            for (int f = 0; f < tensor->F; f++) {
                printf("%.2f ", tensor->data[c * tensor->T * tensor->F + t * tensor->F + f]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

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