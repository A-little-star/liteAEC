#include <stdlib.h>
#include "../include/depthwise_conv2d.h"
#include "../include/batchnorm.h"
#include "../include/elu.h"
#include "../include/tensor.h"
#include "../include/model.h"
#include "../include/linear.h"

BottleNeck* create_bottleneck(int hidden_dim) {
    BottleNeck *btnk = (BottleNeck*)malloc(sizeof(BottleNeck));
    btnk->hidden_dim = hidden_dim;
    btnk->rnn = create_gru_layer(hidden_dim, hidden_dim);
    btnk->fc = create_linear_layer(hidden_dim, hidden_dim);
    return btnk;
}

void free_bottleneck(BottleNeck *btnk) {
    if (btnk) {
        free_gru_layer(btnk->rnn);
        free_linear_layer(btnk->fc);
        free(btnk);
    }
}

Parameter* bottleneck_load_params(BottleNeck *btnk, Parameter *params) {
    Parameter *p1 = gru_load_params(btnk->rnn, params);
    Parameter *p2 = linear_load_params(btnk->fc, p1);
    return p2;
}

Tensor *bottleneck_forward(BottleNeck *btnk, Tensor *input, Tensor *hidden_state) {
    Tensor *mid = gru_forward(btnk->rnn, input, hidden_state);
    Tensor *out = linear_forward(btnk->fc, mid);
    delete_tensor(mid);
    return out;
}