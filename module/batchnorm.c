#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../include/tensor.h"
#include "../include/batchnorm.h"

// 创建 BatchNorm 层
BatchNormLayer* create_batchnorm_layer(int num_features, float eps) {
    BatchNormLayer* layer = (BatchNormLayer*)malloc(sizeof(BatchNormLayer));
    layer->num_features = num_features;
    layer->gamma = (float*)malloc(num_features * sizeof(float));
    layer->beta = (float*)malloc(num_features * sizeof(float));
    layer->running_mean = (float*)malloc(num_features * sizeof(float));
    layer->running_var = (float*)malloc(num_features * sizeof(float));
    layer->eps = eps;

    // 初始化 gamma 为 1，beta 和均值为 0，方差为 1
    for (int i = 0; i < num_features; i++) {
        layer->gamma[i] = 1.0f;
        layer->beta[i] = 0.0f;
        layer->running_mean[i] = 0.0f;
        layer->running_var[i] = 1.0f;
    }

    return layer;
}

// 释放 BatchNorm 层的内存
void free_batchnorm_layer(BatchNormLayer* layer) {
    free(layer->gamma);
    free(layer->beta);
    free(layer->running_mean);
    free(layer->running_var);
    free(layer);
}

// BatchNorm 前向推理
Tensor* batchnorm_forward(BatchNormLayer* layer, Tensor* input) {
    assert(input->ndim >= 1); // 确保输入至少有 1 个维度（C, ...）
    
    int channels = input->shape[0]; // 输入通道数
    int height = input->shape[1];   // 高度
    int width = input->shape[2];    // 宽度

    if (channels != layer->num_features) {
        fprintf(stderr, "BatchNorm layer's number of features doesn't match the input tensor's channels.\n");
        return NULL;
    }

    // 创建输出 Tensor
    Tensor *output = create_tensor(input->shape, input->ndim);

    // 遍历每个通道，应用 BatchNorm
    for (int c = 0; c < channels; c++) {
        float gamma = layer->gamma[c];
        float beta = layer->beta[c];
        float mean = layer->running_mean[c];
        float var = layer->running_var[c];
        float inv_std = 1.0f / sqrtf(var + layer->eps);

        // 遍历该通道的每个元素
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int index = (c * height + h) * width + w;
                float normalized = (input->data[index] - mean) * inv_std; // 归一化
                output->data[index] = normalized * gamma + beta;         // 应用缩放和平移
            }
        }
    }

    return output;
}
