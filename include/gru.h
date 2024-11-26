#ifndef GRU_H
#define GRU_H

typedef struct {
    int input_size;
    int hidden_size;

    // 权重矩阵和偏置向量
    float *W_z, *U_z, *b_z;
    float *W_r, *U_r, *b_r;
    float *W_h, *U_h, *b_h;
} GRULayer;

GRULayer create_gru_layer(int input_size, int hidden_size);

void free_gru_layer(GRULayer *layer);

void gru_forward(GRULayer *layer, float *x_t, float *h_prev, float *h_t);

#endif