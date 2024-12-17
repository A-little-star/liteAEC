#include "../include/tensor.h"
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

// 打印 Tensor 的形状
void print_tensor_shape(Tensor* tensor) {
    printf("Tensor shape: (");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf(")\n");
}

// 打印张量数据
// 打印Tensor递归函数
void print_tensor_recursive(const Tensor* tensor, int depth, int offset) {
    if (depth == tensor->ndim) {
        // 到达最内层，打印单个元素
        printf("%f", tensor->data[offset]);
        return;
    }

    int depth_size = 1;
    for (int i = 0; i <= depth; i ++ )
        depth_size *= tensor->shape[i];

    // 打印当前维度
    printf("[");
    for (int i = 0; i < tensor->shape[depth]; ++i) {
        if (i > 0) {
            printf(", ");
        }
        // 递归打印下一维度
        print_tensor_recursive(tensor, depth + 1, offset + i * (tensor->size / depth_size));
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

// 获取切片后的Tensor
Tensor* tensor_slice(Tensor* tensor, int* start_indices, int* end_indices) {
    // 创建一个新的Tensor来存储切片结果
    Tensor* slice = (Tensor*)malloc(sizeof(Tensor));
    slice->ndim = tensor->ndim;
    slice->shape = (int*)malloc(slice->ndim * sizeof(int));
    
    int size = 1;
    for (int i = 0; i < slice->ndim; i++) {
        slice->shape[i] = end_indices[i] - start_indices[i];
        size *= slice->shape[i];
    }
    
    slice->size = size;
    slice->data = (float*)malloc(size * sizeof(float));
    
    // 复制切片数据
    int* indices = (int*)malloc(slice->ndim * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        // 计算当前索引
        int temp_offset = i;
        for (int j = slice->ndim - 1; j >= 0; j--) {
            indices[j] = temp_offset % slice->shape[j] + start_indices[j];
            temp_offset /= slice->shape[j];
        }
        offset = compute_offset(tensor, indices);
        slice->data[i] = tensor->data[offset];
    }

    free(indices);
    return slice;
}

// 获取Tensor的元素总数
int get_tensor_size(Tensor* tensor) {
    int size = 1;
    for (int i = 0; i < tensor->ndim; i++) {
        size *= tensor->shape[i];
    }
    return size;
}

// Tensor的squeeze操作，只移除指定dim维度的大小为1的维度
Tensor* tensor_squeeze(Tensor* tensor, int dim) {
    // 检查dim维度是否为1
    if (tensor->shape[dim] != 1) {
        printf("Error: The dimension %d is not of size 1!\n", dim);
        return tensor;  // 如果该维度不是1，直接返回原Tensor
    }
    // 创建一个新的Tensor来存储squeeze后的结果
    Tensor* squeezed_tensor = (Tensor*)malloc(sizeof(Tensor));
    // 新的Tensor形状数组
    squeezed_tensor->ndim = tensor->ndim - 1;
    squeezed_tensor->shape = (int*)malloc(squeezed_tensor->ndim * sizeof(int));
    // 复制新的维度形状，去掉dim维度
    int j = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        if (i != dim) {
            squeezed_tensor->shape[j++] = tensor->shape[i];
        }
    }
    // 计算新的Tensor大小
    squeezed_tensor->size = get_tensor_size(squeezed_tensor);
    // 分配新的内存空间
    squeezed_tensor->data = (float*)malloc(squeezed_tensor->size * sizeof(float));
    // 使用memcpy进行数据拷贝
    memcpy(squeezed_tensor->data, tensor->data, squeezed_tensor->size * sizeof(float));
    return squeezed_tensor;
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