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

typedef struct {
    EncoderBlock *mic_enc[4];
    EncoderBlock *ref_enc;
} RNNVQE;

EncoderBlock *create_encoder_block(int in_channels, int out_channels);
void free_encoder_block(EncoderBlock *block);
Parameter* encoderblock_load_params(EncoderBlock *block, Parameter *params);
Tensor *encoderblock_forward(EncoderBlock *block, Tensor *input);

RNNVQE* create_rnnvqe();
Parameter* rnnvqe_load_params(RNNVQE *model, ModelStateDict *sd);
void free_rnnvqe(RNNVQE *model);
Tensor *rnnvqe_forward(RNNVQE *model, Tensor *mic, Tensor *ref);

#endif