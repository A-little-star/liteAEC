#ifndef ELU_H
#define ELU_H

// ELU 层定义
typedef struct {
    float alpha;  // 控制负值部分的参数
} ELULayer;

ELULayer* create_elu_layer(float alpha);

void free_elu_layer(ELULayer *layer);

Tensor* elu_forward(ELULayer* layer, Tensor* input);

#endif