#ifndef MATRIX_OP_H
#define MATRIX_OP_H

// 张量的结构体定义
// 目前只支持float32类型
typedef struct {
    float* data;            // 存储数据的指针
    int* shape;             // 形状数组
    int ndim;               // Tensor的维度
    int size;               // Tensor的总元素个数
} Tensor;

// 创建张量
// Tensor create_tensor(int c, int t, int f);
Tensor* create_tensor(int* shape, int ndim);
// 删除张量
void delete_tensor(Tensor *tensor);
// 初始化张量数据
void init_tensor(Tensor* tensor, float* values);
// 打印张量的形状
void print_tensor_shape(Tensor* tensor);
// 打印张量数据
void print_tensor(const Tensor* tensor);
// 修改Tensor某位置的值
void tensor_set(Tensor* tensor, const int* indices, float value);
// 获取Tensor某位置的值
float tensor_get(const Tensor* tensor, const int* indices);
// 矩阵切片
Tensor* tensor_slice(Tensor* tensor, int* start_indices, int* end_indices);
// squeeze()操作，将第dim维度折叠
Tensor* tensor_squeeze(Tensor* tensor, int dim);
// concat拼接操作
Tensor* concatenate(Tensor* tensor1, Tensor* tensor2, int dim);

// 矩阵运算

// 逐元素加法
void elementwise_add(float *a, float *b, float *out, int size);
// 逐元素乘法
void elementwise_mul(float *a, float *b, float *out, int size);
// 矩阵向量乘法
void matvec_mul(float *mat, float *vec, float *out, int rows, int cols);

#endif