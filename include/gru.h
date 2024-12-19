#ifndef GRU_H
#define GRU_H
#include "tensor.h"
#include "parser.h"

// GRU层定义
typedef struct {
    int input_size;  // 输入特征维度
    int hidden_size; // 隐藏层维度

    // 权重矩阵和偏置向量
    float *W_ir, *W_iz, *W_in; // 输入到各门的权重
    float *W_hr, *W_hz, *W_hn; // 隐藏状态到各门的权重
    float *b_ir, *b_iz, *b_in; // 输入偏置
    float *b_hr, *b_hz, *b_hn; // 隐藏状态偏置
} GRULayer;

GRULayer* create_gru_layer(int input_size, int hidden_size);
void free_gru_layer(GRULayer* layer);
Parameter* gru_load_params(GRULayer* layer, Parameter* params);
Tensor* gru_forward(GRULayer* layer, Tensor* input, Tensor* hidden_state);

#endif