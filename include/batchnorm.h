#ifndef BATCHNORM_H
#define BATCHNORM_H
#include "tensor.h"
#include "parser.h"

// BatchNorm 层定义
typedef struct {
    int num_features;   // 输入通道数
    float* gamma;       // 缩放参数
    float* beta;        // 平移参数
    float* running_mean; // 训练阶段累积的均值
    float* running_var;  // 训练阶段累积的方差
    float eps;      // 防止除零的小值
} BatchNormLayer;

BatchNormLayer* create_batchnorm_layer(int num_features, float epsilon);
void free_batchnorm_layer(BatchNormLayer* layer);
Parameter* batchnorm_load_params(BatchNormLayer *layer, Parameter *params);
Tensor* batchnorm_forward(BatchNormLayer* layer, Tensor* input);

#endif