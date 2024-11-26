#ifndef DEPTHWISECONV2D_H
#define DEPTHWISECONV2D_H

#include "conv2d.h"

typedef struct {
    int in_channels;   // 输入通道数
    int out_channels;  // 输出通道数
    int kernel_h, kernel_w; // 卷积核尺寸
    int stride_h, stride_w; // 步幅
    int padding_h, padding_w; // 填充

    Conv2DLayer depth_conv;
    Conv2DLayer point_conv;
} DepthwiseConv2DLayer;

DepthwiseConv2DLayer create_depthwise_conv2d_layer(int in_channels, int out_channels,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w);

void depthwise_conv2d_forward(Conv2DLayer *layer, float *input, int in_h, int in_w,
                    float *output, int out_h, int out_w);

#endif