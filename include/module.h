#ifndef MODULE_H
#define MODULE_H
#include "conv2d.h"
#include "batchnorm.h"
#include "elu.h"

typedef struct {
    int hidden_channels;
    Conv2DLayer *conv;
} SkipBlock;

typedef struct {
    int in_channels;
    int out_channels;
    Conv2DLayer *conv;
} SubPixelConv;

typedef struct {
    int hidden_channels;
    Conv2DLayer* conv2d;
    BatchNormLayer* bn;
    ELULayer* act;
} ResidualBlock;

SkipBlock* create_skipblock(int hidden_channels);
void free_skipblock(SkipBlock *block);
Parameter* skipblock_load_params(SkipBlock* block, Parameter* params);
Tensor* skipblock_forward(SkipBlock* block, Tensor* encoder_feats, Tensor* decoder_feats);

SubPixelConv* create_subpixelconv(int in_channels, int out_channels);
void free_subpixelconv(SubPixelConv *block);
Parameter* subpixelconv_load_params(SubPixelConv* block, Parameter* params);
Tensor* subpixelconv_forward(SubPixelConv* block, Tensor* input);

ResidualBlock* create_residualblock(int hidden_channels);
void free_residualblock(ResidualBlock* block);
Parameter* residualblock_load_params(ResidualBlock* block, Parameter* params);
Tensor* residualblock_forward(ResidualBlock* block, Tensor* input);

#endif