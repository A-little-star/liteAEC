#ifndef LINEAR_H
#define LINEAR_H
#include "tensor.h"
#include "parser.h"

// 定义线性层的结构
typedef struct {
    int input_size;
    int output_size;
    float *weight;  // 权重矩阵 (output_size x input_size)
    float *bias;    // 偏置向量 (output_size)
} LinearLayer;

LinearLayer* create_linear_layer(int input_size, int output_size);
void free_linear_layer(LinearLayer* layer);
Parameter* linear_load_params(LinearLayer* layer, Parameter* params);
Tensor* linear_forward(LinearLayer* layer, Tensor* input);

#endif