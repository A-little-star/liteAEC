#include "../include/conv2d.h"
#include "../include/depthwise_conv2d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DepthwiseConv2DLayer create_depthwise_conv2d_layer(int in_channels, int out_channels,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w) {
    DepthwiseConv2DLayer layer;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_h = kernel_h;
    layer.kernel_w = kernel_w;
    layer.stride_h = stride_h;
    layer.stride_w = stride_w;
    layer.padding_h = padding_h;
    layer.padding_w = padding_w;

    layer.depth_conv = create_conv2d_layer(
        in_channels, in_channels, 
        kernel_h, kernel_w, 
        stride_h, stride_w, 
        padding_h, padding_w, 
        in_channels
    );
    layer.point_conv = create_conv2d_layer(
        in_channels, out_channels,
        1, 1,
        1, 1, 
        0, 0,
        1
    );
    return layer;
};

void free_depthwise_conv2d_layer(DepthwiseConv2DLayer *layer) {
    free_conv2d_layer(&layer->depth_conv);
    free_conv2d_layer(&layer->point_conv);
}

Tensor depthwise_conv2d_forward(DepthwiseConv2DLayer *layer, Tensor input) {
    Conv2DLayer depth_conv_layer = layer->depth_conv;
    Conv2DLayer point_conv_layer = layer->point_conv;
    Tensor mid = conv2d_forward(&depth_conv_layer, input);
    Tensor out = conv2d_forward(&point_conv_layer, mid);
    // 销毁中间变量
    delete_tensor(&mid);
    return out;
}