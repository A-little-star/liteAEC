#ifndef SIGMOID_H
#define SIGMOID_H

#include <math.h>
#include <stdlib.h>
#include "tensor.h"
#include "act_func.h"

// Sigmoid 层定义
typedef struct {
    // Sigmoid 层不需要额外参数
} SigmoidLayer;

SigmoidLayer* create_sigmoid_layer();

void free_sigmoid_layer(SigmoidLayer* layer);

// Sigmoid 前向传播
Tensor* sigmoid_forward(SigmoidLayer* layer, Tensor* input);

#endif