#include "../include/matrix_op.h"
#include "../include/gru.h"
#include "../include/conv2d.h"
#include "../include/elu.h"
#include "../include/linear.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    // 输入参数
    int in_channels = 3, in_h = 4, in_w = 4;
    float input[3 * 4 * 4] = {
        1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,  13, 14, 15, 16, // 通道 1
        1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,    1, 1, 1, 1, // 通道 2
        0, 1, 0, 1,   0, 1, 0, 1,   0, 1, 0, 1,    0, 1, 0, 1  // 通道 3
    };

    // 卷积层参数
    int out_channels = 2, kernel_h = 3, kernel_w = 3, stride_h = 1, stride_w = 1;
    int padding_h = 1, padding_w = 1; // 使用 0 填充
    int group = 1;
    Conv2DLayer conv_layer = create_conv2d_layer(in_channels, out_channels,
                                                 kernel_h, kernel_w,
                                                 stride_h, stride_w,
                                                 padding_h, padding_w,
                                                 group);

    // 输出参数
    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    float output[2 * out_h * out_w]; // 2 是 out_channels

    // 执行卷积
    conv2d_forward(&conv_layer, input, in_h, in_w, output, out_h, out_w);

    // 打印输出结果
    printf("Output:\n");
    for (int oc = 0; oc < out_channels; oc++) {
        printf("Channel %d:\n", oc + 1);
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                printf("%f ", output[(oc * out_h + oh) * out_w + ow]);
            }
            printf("\n");
        }
    }

    // 释放内存
    free_conv2d_layer(&conv_layer);

    return 0;
}