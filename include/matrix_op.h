#ifndef MATRIX_OP_H
#define MATRIX_OP_H

// 张量的结构体定义
typedef struct {
    float* data;
    int C; // 通道数
    int T; // 时间维度
    int F; // 特征维度
} Tensor;

// 创建张量
Tensor create_tensor(int c, int t, int f);
// 删除张量
void delete_tensor(Tensor m);
// 初始化张量数据
void init_tensor(Tensor* tensor, float* values);
// 打印张量数据
void print_tensor(const Tensor* tensor);

// 矩阵运算

// 逐元素加法
void elementwise_add(float *a, float *b, float *out, int size);
// 逐元素乘法
void elementwise_mul(float *a, float *b, float *out, int size);
// 矩阵向量乘法
void matvec_mul(float *mat, float *vec, float *out, int rows, int cols);

#endif