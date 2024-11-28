#ifndef CONV2D_H
#define CONV2D_H
#include "matrix_op.h"

// 定义卷积层的结构体（包括权重、偏置等参数）
typedef struct {
    int in_channels;   // 输入通道数
    int out_channels;  // 输出通道数
    int kernel_h, kernel_w; // 卷积核尺寸
    int stride_h, stride_w; // 步幅
    int padding_h, padding_w; // 填充
    int group; // 分组

    // 权重和偏置
    float *weights; // 权重: [out_channels, in_channels, kernel_h, kernel_w]
    float *bias;    // 偏置: [out_channels]
} Conv2DLayer;

// 创建卷积层
Conv2DLayer create_conv2d_layer(int in_channels, int out_channels,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w,
                                int group);

// 执行卷积操作
Tensor conv2d_forward(Conv2DLayer *layer, Tensor input);

// 释放内存
void free_conv2d_layer(Conv2DLayer *layer);

#endif // CONV2D_H
