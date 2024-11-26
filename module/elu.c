#include "../include/elu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 初始化 ELU 层
ELULayer create_elu_layer(float alpha) {
    ELULayer layer;
    layer.alpha = alpha;
    return layer;
}

// 释放 ELU 层的内存（如果有需要的话，虽然本例中没有动态分配内存）
void free_elu_layer(ELULayer *layer) {
    // 这里暂时不需要释放内存
}

// ELU 激活函数的前向传播
void elu_forward(ELULayer *layer, float *input, int size, float *output) {
    float alpha = layer->alpha;

    // 遍历输入数据，进行 ELU 激活
    for (int i = 0; i < size; i++) {
        if (input[i] >= 0) {
            output[i] = input[i];  // 对于 x >= 0，f(x) = x
        } else {
            output[i] = alpha * (exp(input[i]) - 1);  // 对于 x < 0，f(x) = alpha * (exp(x) - 1)
        }
    }
}

// int main() {
//     // 输入数据（例如，一个包含 5 个元素的数组）
//     float input[] = {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f};
//     int size = sizeof(input) / sizeof(input[0]);
//     float output[size];

//     // 创建 ELU 层，alpha 设为 1.0
//     ELULayer elu_layer = create_elu_layer(1.0f);

//     // 执行前向传播
//     elu_forward(&elu_layer, input, size, output);

//     // 打印输出结果
//     printf("ELU Activation Output:\n");
//     for (int i = 0; i < size; i++) {
//         printf("%f ", output[i]);
//     }
//     printf("\n");

//     // 释放 ELU 层
//     free_elu_layer(&elu_layer);

//     return 0;
// }
