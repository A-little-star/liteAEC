#include "../include/matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 创建一个新的张量
Tensor* create_tensor(int* shape, int ndim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));

    // 复制形状并计算步长
    tensor->size = 1;
    for (int i = 0; i < ndim; ++i) {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    // 分配内存
    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    return tensor;
}
// Tensor create_tensor(int c, int t, int f) {
//     Tensor tensor;
//     tensor.C = c;
//     tensor.T = t;
//     tensor.F = f;
//     tensor.data = (float *)malloc(c * t * f * sizeof(float));
//     if (tensor.data == NULL) {
//         fprintf(stderr, "Failed to allocate memory for output\n");
//         exit(EXIT_FAILURE); // 确保程序不会继续运行
//     }
//     return tensor;
// }

// 释放张量所占的内存空间
void delete_tensor(Tensor *tensor) {
    free(tensor->data);
    free(tensor->shape);
    free(tensor);
}

// 初始化张量数据
void init_tensor(Tensor* tensor, float* values) {
    if (tensor == NULL || tensor->data == NULL || values == NULL) {
        fprintf(stderr, "Invalid tensor or values\n");
        return;
    }
    int total_elements = tensor->size;
    for (int i = 0; i < total_elements; i++) {
        tensor->data[i] = values[i];
    }
}

// 打印张量数据
// 打印Tensor递归函数
void print_tensor_recursive(const Tensor* tensor, int depth, int offset) {
    if (depth == tensor->ndim) {
        // 到达最内层，打印单个元素
        printf("%f", tensor->data[offset]);
        return;
    }

    // 打印当前维度
    printf("[");
    for (int i = 0; i < tensor->shape[depth]; ++i) {
        if (i > 0) {
            printf(", ");
        }
        // 递归打印下一维度
        print_tensor_recursive(tensor, depth + 1, offset + i * (tensor->size / tensor->shape[depth]));
    }
    printf("]");
}

// 打印Tensor的主函数
void print_tensor(const Tensor* tensor) {
    if (!tensor || !tensor->data || !tensor->shape) {
        printf("Error: Invalid Tensor.\n");
        return;
    }

    print_tensor_recursive(tensor, 0, 0);
    printf("\n");
}
// void print_tensor(const Tensor* tensor) {
//     if (tensor == NULL || tensor->data == NULL) {
//         fprintf(stderr, "Invalid tensor\n");
//         return;
//     }
//     for (int c = 0; c < tensor->C; c++) {
//         printf("Channel %d:\n", c + 1);
//         for (int t = 0; t < tensor->T; t++) {
//             for (int f = 0; f < tensor->F; f++) {
//                 printf("%.2f ", tensor->data[c * tensor->T * tensor->F + t * tensor->F + f]);
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }
// }

// 计算多维索引对应的一维偏移
int compute_offset(const Tensor* tensor, const int* indices) {
    int offset = 0;
    int stride = tensor->size;

    for (int i = 0; i < tensor->ndim; ++i) {
        stride /= tensor->shape[i];  // 计算当前维度的步长
        if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
            printf("Error: Index out of bounds at dimension %d.\n", i);
            return -1;  // 返回无效偏移
        }
        offset += indices[i] * stride;
    }
    return offset;
}

// 修改Tensor某位置的值
void tensor_set(Tensor* tensor, const int* indices, float value) {
    int offset = compute_offset(tensor, indices);
    if (offset == -1) {
        return;  // 出错直接返回
    }
    tensor->data[offset] = value;
}

// 获取Tensor某位置的值
float tensor_get(const Tensor* tensor, const int* indices) {
    int offset = compute_offset(tensor, indices);
    if (offset == -1) {
        return 0.0f;  // 出错返回默认值
    }
    return tensor->data[offset];
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