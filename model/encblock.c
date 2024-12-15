#include "../include/depthwise_conv2d.h"

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;

    DepthwiseConv2DLayer *conv2d;

} EncoderBlock;