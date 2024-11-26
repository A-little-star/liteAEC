#ifndef ELU_H
#define ELU_H

typedef struct {
    float alpha;  // ELU 的 alpha 参数，通常是 1
} ELULayer;

ELULayer create_elu_layer(float alpha);

void free_elu_layer(ELULayer *layer);

void elu_forward(ELULayer *layer, float *input, int size, float *output);

#endif