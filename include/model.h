#ifndef MODEL_H
#define MODEL_H
#include "depthwise_conv2d.h"
#include "batchnorm.h"
#include "elu.h"

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;

    DepthwiseConv2DLayer *conv2d;
    BatchNormLayer *bn;
    ELULayer *act;
} EncoderBlock;
EncoderBlock *create_encoder_block(int in_channels, int out_channels);
void free_encoder_block(EncoderBlock *block);
Parameter* encoderblock_load_params(EncoderBlock *block, Parameter *params);
Tensor *encoderblock_forward(EncoderBlock *block, Tensor *input);

#endif