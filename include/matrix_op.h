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
void delete_tensor(Tensor *tensor);
// 初始化张量数据
void init_tensor(Tensor* tensor, float* values);
// 打印张量数据
void print_tensor(const Tensor* tensor);
// 将索引为[c_idx, t_idx, f_idx]的元素的值设为value
void set_value(Tensor *tensor, int c_idx, int t_idx, int f_idx, float value);
// 获取索引为[c_idx, t_idx, f_idx]的元素的值
float get_value(Tensor *tensor, int c_idx, int t_idx, int f_idx);

// 矩阵运算

// 逐元素加法
void elementwise_add(float *a, float *b, float *out, int size);
// 逐元素乘法
void elementwise_mul(float *a, float *b, float *out, int size);
// 矩阵向量乘法
void matvec_mul(float *mat, float *vec, float *out, int rows, int cols);

#endif