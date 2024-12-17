#ifndef DEPTHWISECONV2D_H
#define DEPTHWISECONV2D_H

#include "conv2d.h"

typedef struct {
    int in_channels;   // 输入通道数
    int out_channels;  // 输出通道数
    int kernel_h, kernel_w; // 卷积核尺寸
    int stride_h, stride_w; // 步幅
    int padding_h, padding_w; // 填充

    Conv2DLayer* depth_conv;
    Conv2DLayer* point_conv;
} DepthwiseConv2DLayer;

DepthwiseConv2DLayer* create_depthwise_conv2d_layer(int in_channels, int out_channels,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w);

Parameter* depthwise_conv2d_load_params(DepthwiseConv2DLayer *layer, Parameter *params);

void free_depthwise_conv2d_layer(DepthwiseConv2DLayer *layer);

Tensor* depthwise_conv2d_forward(DepthwiseConv2DLayer *layer, Tensor *input);

#endif