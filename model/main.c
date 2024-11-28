#include "../include/matrix_op.h"
#include "../include/gru.h"
#include "../include/conv2d.h"
#include "../include/elu.h"
#include "../include/linear.h"
#include "../include/depthwise_conv2d.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    // 输入参数
    int in_channels = 3, in_h = 4, in_w = 4;
    float input_data[3 * 4 * 4] = {
        1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,  13, 14, 15, 16, // 通道 1
        1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,    1, 1, 1, 1, // 通道 2
        0, 1, 0, 1,   0, 1, 0, 1,   0, 1, 0, 1,    0, 1, 0, 1  // 通道 3
    };
    Tensor input = create_tensor(3, 4, 4);
    init_tensor(&input, input_data);

    // 卷积层参数
    int out_channels = 2, kernel_h = 3, kernel_w = 3, stride_h = 1, stride_w = 1;
    int padding_h = 1, padding_w = 1; // 使用 0 填充
    int group = 1;
    Conv2DLayer conv_layer = create_conv2d_layer(in_channels, out_channels,
                                                 kernel_h, kernel_w,
                                                 stride_h, stride_w,
                                                 padding_h, padding_w,
                                                 group);
    DepthwiseConv2DLayer depthwise_conv2d_layer = create_depthwise_conv2d_layer(in_channels, out_channels, 
                                                                                kernel_h, kernel_w,
                                                                                stride_h, stride_w, padding_h, padding_w);

    // 执行卷积
    Tensor output = conv2d_forward(&conv_layer, input);
    Tensor output_dwconv = depthwise_conv2d_forward(&depthwise_conv2d_layer, input);

    // 打印输出结果
    printf("Output conv2d:\n");
    print_tensor(&output);
    printf("Output dwconv2d:\n");
    print_tensor(&output_dwconv);

    // 释放内存
    free_conv2d_layer(&conv_layer);
    free_depthwise_conv2d_layer(&depthwise_conv2d_layer);

    return 0;
}