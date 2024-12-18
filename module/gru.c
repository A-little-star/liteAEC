#include "../include/tensor.h"
#include "../include/gru.h"
#include "../include/act_func.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 初始化 GRU 层
GRULayer create_gru_layer(int input_size, int hidden_size) {
    GRULayer layer;
    layer.input_size = input_size;
    layer.hidden_size = hidden_size;

    // 分配权重和偏置的内存
    int weight_size = input_size * hidden_size;
    int hidden_weight_size = hidden_size * hidden_size;

    layer.W_z = (float *)malloc(weight_size * sizeof(float));
    layer.U_z = (float *)malloc(hidden_weight_size * sizeof(float));
    layer.b_z = (float *)malloc(hidden_size * sizeof(float));

    layer.W_r = (float *)malloc(weight_size * sizeof(float));
    layer.U_r = (float *)malloc(hidden_weight_size * sizeof(float));
    layer.b_r = (float *)malloc(hidden_size * sizeof(float));

    layer.W_h = (float *)malloc(weight_size * sizeof(float));
    layer.U_h = (float *)malloc(hidden_weight_size * sizeof(float));
    layer.b_h = (float *)malloc(hidden_size * sizeof(float));

    // 随机初始化权重和偏置
    for (int i = 0; i < weight_size; i++) {
        layer.W_z[i] = (float)rand() / RAND_MAX;
        layer.W_r[i] = (float)rand() / RAND_MAX;
        layer.W_h[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < hidden_weight_size; i++) {
        layer.U_z[i] = (float)rand() / RAND_MAX;
        layer.U_r[i] = (float)rand() / RAND_MAX;
        layer.U_h[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < hidden_size; i++) {
        layer.b_z[i] = 0.1f;
        layer.b_r[i] = 0.1f;
        layer.b_h[i] = 0.1f;
    }

    return layer;
}

// 释放 GRU 层内存
void free_gru_layer(GRULayer *layer) {
    free(layer->W_z);
    free(layer->U_z);
    free(layer->b_z);
    free(layer->W_r);
    free(layer->U_r);
    free(layer->b_r);
    free(layer->W_h);
    free(layer->U_h);
    free(layer->b_h);
}

// GRU 前向传播
void gru_forward(GRULayer *layer, float *x_t, float *h_prev, float *h_t) {
    int hidden_size = layer->hidden_size;

    float z_t[hidden_size];
    float r_t[hidden_size];
    float h_candidate[hidden_size];
    float temp[hidden_size];

    // 计算更新门 z_t
    matvec_mul(layer->W_z, x_t, z_t, hidden_size, layer->input_size);
    matvec_mul(layer->U_z, h_prev, temp, hidden_size, hidden_size);
    elementwise_add(z_t, temp, z_t, hidden_size);
    elementwise_add(z_t, layer->b_z, z_t, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        z_t[i] = sigmoid(z_t[i]);
    }

    // 计算重置门 r_t
    matvec_mul(layer->W_r, x_t, r_t, hidden_size, layer->input_size);
    matvec_mul(layer->U_r, h_prev, temp, hidden_size, hidden_size);
    elementwise_add(r_t, temp, r_t, hidden_size);
    elementwise_add(r_t, layer->b_r, r_t, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        r_t[i] = sigmoid(r_t[i]);
    }

    // 计算候选隐藏状态 h_candidate
    matvec_mul(layer->W_h, x_t, h_candidate, hidden_size, layer->input_size);
    elementwise_mul(r_t, h_prev, temp, hidden_size);
    matvec_mul(layer->U_h, temp, temp, hidden_size, hidden_size);
    elementwise_add(h_candidate, temp, h_candidate, hidden_size);
    elementwise_add(h_candidate, layer->b_h, h_candidate, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        h_candidate[i] = tanh_activation(h_candidate[i]);
    }

    // 计算当前隐藏状态 h_t
    for (int i = 0; i < hidden_size; i++) {
        h_t[i] = (1 - z_t[i]) * h_prev[i] + z_t[i] * h_candidate[i];
    }
}

// int main() {
//     int input_size = 3;
//     int hidden_size = 4;

//     // 创建 GRU 层
//     GRULayer gru = create_gru_layer(input_size, hidden_size);

//     // 定义输入和前一隐藏状态
//     float x_t[3] = {1.0, 2.0, 3.0};
//     float h_prev[4] = {0.0, 0.0, 0.0, 0.0};

//     // 定义当前隐藏状态
//     float h_t[4];

//     // 前向传播
//     gru_forward(&gru, x_t, h_prev, h_t);

//     // 打印输出
//     printf("Hidden State:\n");
//     for (int i = 0; i < hidden_size; i++) {
//         printf("%f ", h_t[i]);
//     }
//     printf("\n");

//     // 释放内存
//     free_gru_layer(&gru);

//     return 0;
// }
