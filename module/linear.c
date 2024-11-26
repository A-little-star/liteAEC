#include <stdio.h>
#include <stdlib.h>

// 定义线性层的结构
typedef struct {
    int input_size;
    int output_size;
    float *weights; // 权重矩阵 (input_size x output_size)
    float *bias;    // 偏置向量 (output_size)
} LinearLayer;

// 初始化线性层
LinearLayer create_linear_layer(int input_size, int output_size) {
    LinearLayer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;

    // 分配权重和偏置的内存
    layer.weights = (float *)malloc(input_size * output_size * sizeof(float));
    layer.bias = (float *)malloc(output_size * sizeof(float));

    // 初始化权重和偏置为随机值（这里使用固定值示例）
    for (int i = 0; i < input_size * output_size; i++) {
        layer.weights[i] = (float)rand() / RAND_MAX; // 随机初始化
    }
    for (int i = 0; i < output_size; i++) {
        layer.bias[i] = 0.1f; // 偏置初始化为 0.1
    }

    return layer;
}

// 释放线性层的内存
void free_linear_layer(LinearLayer *layer) {
    free(layer->weights);
    free(layer->bias);
}

// 线性层的前向传播
void linear_forward(LinearLayer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->bias[i]; // 初始化为偏置
        for (int j = 0; j < layer->input_size; j++) {
            output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }
    }
}

// int main() {
//     // 定义输入向量
//     float input[3] = {1.0, 2.0, 3.0};

//     // 创建隐藏层和输出层
//     LinearLayer hidden_layer = create_linear_layer(3, 4);
//     LinearLayer output_layer = create_linear_layer(4, 2);

//     // 定义隐藏层和输出层的输出向量
//     float hidden_output[4];
//     float final_output[2];

//     // 前向传播
//     linear_forward(&hidden_layer, input, hidden_output);
//     linear_forward(&output_layer, hidden_output, final_output);

//     // 打印输出
//     printf("Final Output:\n");
//     for (int i = 0; i < 2; i++) {
//         printf("%f ", final_output[i]);
//     }
//     printf("\n");

//     // 释放内存
//     free_linear_layer(&hidden_layer);
//     free_linear_layer(&output_layer);

//     return 0;
// }
