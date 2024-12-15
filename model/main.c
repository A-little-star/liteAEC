#include "../include/tensor.h"
#include "../include/gru.h"
#include "../include/conv2d.h"
#include "../include/elu.h"
#include "../include/linear.h"
#include "../include/depthwise_conv2d.h"
#include "../include/wavreader.h"
#include "../include/stft.h"
#include "../include/typedef.h"
#include "../include/pfdkf.h"
#include "../include/parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 测试wavreader.c
// int main() {
//     const char *filename = "example.wav";    // 替换为实际文件名
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *audio_data = read_wav_file(filename, &num_samples, &sample_rate);
//     // wav_norm(audio_data, num_samples);

//     if (write_wav_file("output.wav", audio_data, num_samples, sample_rate, 1, 16) == 0) {
//         printf("WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }

//     free(audio_data);
//     return 0;
// }

// 测试线性滤波
// int main() {
//     // Example usage of PFDKF
//     const char *filename_mic = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
//     const char *filename_ref = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav";
//     const char *filename_e = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/e.wav";
//     const char *filename_y = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/y.wav";
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *mic = read_wav_file(filename_mic, &num_samples, &sample_rate);
//     float *ref = read_wav_file(filename_ref, &num_samples, &sample_rate);

//     float *e = (float*)calloc(num_samples, sizeof(float));
//     float *y = (float*)calloc(num_samples, sizeof(float));
//     pfdkf(ref, mic, e, y, num_samples);

//     if (write_wav_file(filename_e, e, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }
//     if (write_wav_file(filename_y, y, num_samples, sample_rate, 1, 16) == 0) {
//         printf("y WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }

//     return 0;
// }

// 测试加载参数
// int main() {
//     const char *filename = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/test_dict.txt";
//     Parameter *params = (Parameter*)malloc(MAX_PARAMS * sizeof(Parameter));
//     parse_ckpt(filename, params);
//     return EXIT_SUCCESS;
// }

// 测试整个模型
int main() {
    // *** stage 1: 读取文件 ***
    const char *filename_mic = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
    const char *filename_ref = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav";
    const char *filename_e = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/e.wav";
    const char *filename_y = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/hello.wav";
    int num_samples;                        // 样本数
    int sample_rate;                        // 采样率
    float *mic_ = read_wav_file(filename_mic, &num_samples, &sample_rate);
    float *ref_ = read_wav_file(filename_ref, &num_samples, &sample_rate);

    float *e_ = (float*)calloc(num_samples, sizeof(float));
    float *y_ = (float*)calloc(num_samples, sizeof(float));

    printf("stage 1 executed successfully.\n");

    // *** stage 2: 线性滤波与特征提取 ***
    // pfdkf(ref_, mic_, e_, y_, num_samples);

    wav_norm(mic_, num_samples);
    wav_norm(ref_, num_samples);
    wav_norm(e_, num_samples);
    wav_norm(y_, num_samples);

    Tensor *mic = create_tensor((int[]){num_samples}, 1);
    Tensor *y = create_tensor((int[]){num_samples}, 1);
    init_tensor(mic, mic_);
    init_tensor(y, y_);

    int length = num_samples / FRAME_SIZE;
    Tensor* cspecs_mic = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
    Tensor* cspecs_y = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
    Tensor* features_mic = create_tensor((int[]){1, length, NB_FEATURES}, 3);
    Tensor* features_y = create_tensor((int[]){1, length, NB_FEATURES}, 3);
    feature_extract(mic, cspecs_mic, features_mic);
    feature_extract(y, cspecs_y, features_y);

    printf("stage 2 executed successfully.\n");

    // *** stage 3: 读取模型参数并解析 ***
    const char *cpt = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/test_dict.txt";
    ModelStateDict *sd = create_model_state_dict();
    parse_ckpt(cpt, sd);

    printf("stage 3 executed successfully.\n");

    // *** stage 4: 构建模型并加载模型参数 ***
    DepthwiseConv2DLayer *conv2d = create_depthwise_conv2d_layer(1, 8, 4, 3, 1, 2, 3, 0);
    Parameter *params = sd->params;
    conv2d_load_params(conv2d->depth_conv, params[0].values, params[1].values);
    // conv2d_load_params(conv2d->point_conv, params[2].values, params[3].values);

    free_model_state_dict(sd);

    printf("stage 4 executed successfully.\n");

    // *** stage 5: 模型推理 ***
    Tensor *outputs = depthwise_conv2d_forward(conv2d, features_mic);
    print_tensor_shape(outputs);

    // const char *output_file_mic = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/bfcc_mic.txt";
    // const char *output_file_y = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/bfcc_y.txt";
    // FILE *file = fopen(output_file_mic, "w");
    // FILE *file_y = fopen(output_file_y, "w");
    // if (!file) {
    //     perror("Error opening output file");
    //     return 1;
    // }

    // int T = features_mic->shape[1], F = features_mic->shape[2];
    // for (int t = 0; t < T; t ++ ) {
    //     for (int f = 0; f < F; f ++ ) {
    //         float tfbin = tensor_get(features_mic, (int[]){0, t, f});
    //         float tfbin_y = tensor_get(features_y, (int[]){0, t, f});
    //         // float tfbin = get_value(&features, 0, t, f);
    //         fprintf(file, "%f ", tfbin);
    //         fprintf(file_y, "%f ", tfbin_y);
    //     }
    //     fprintf(file, "\n");
    //     fprintf(file_y, "\n");
    // }

    // fclose(file); // 关闭文件
    // fclose(file_y);

    return 0;
}

// 测试卷积模块
// int main() {
//     // 输入参数
//     int in_channels = 3, in_h = 4, in_w = 4;
//     float input_data[3 * 4 * 4] = {
//         1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,  13, 14, 15, 16, // 通道 1
//         1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,    1, 1, 1, 1, // 通道 2
//         0, 1, 0, 1,   0, 1, 0, 1,   0, 1, 0, 1,    0, 1, 0, 1  // 通道 3
//     };
//     Tensor input = create_tensor(3, 4, 4);
//     init_tensor(&input, input_data);

//     // 卷积层参数
//     int out_channels = 2, kernel_h = 3, kernel_w = 3, stride_h = 1, stride_w = 1;
//     int padding_h = 1, padding_w = 1; // 使用 0 填充
//     int group = 1;
//     Conv2DLayer conv_layer = create_conv2d_layer(in_channels, out_channels,
//                                                  kernel_h, kernel_w,
//                                                  stride_h, stride_w,
//                                                  padding_h, padding_w,
//                                                  group);
//     DepthwiseConv2DLayer depthwise_conv2d_layer = create_depthwise_conv2d_layer(in_channels, out_channels, 
//                                                                                 kernel_h, kernel_w,
//                                                                                 stride_h, stride_w, padding_h, padding_w);

//     // 执行卷积
//     Tensor output = conv2d_forward(&conv_layer, input);
//     Tensor output_dwconv = depthwise_conv2d_forward(&depthwise_conv2d_layer, input);

//     // 打印输出结果
//     printf("Output conv2d:\n");
//     print_tensor(&output);
//     printf("Output dwconv2d:\n");
//     print_tensor(&output_dwconv);

//     // 释放内存
//     free_conv2d_layer(&conv_layer);
//     free_depthwise_conv2d_layer(&depthwise_conv2d_layer);

//     return 0;
// }