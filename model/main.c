#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#include "../include/batchnorm.h"
#include "../include/elu.h"
#include "../include/model.h"

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

// 测试矩阵切片
// int main() {
//     // 输入参数
//     int in_channels = 3, in_h = 4, in_w = 4;
//     float input_data[3 * 4 * 4] = {
//         1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,  13, 14, 15, 16, // 通道 1
//         1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,    1, 1, 1, 1, // 通道 2
//         0, 1, 0, 1,   0, 1, 0, 1,   0, 1, 0, 1,    0, 1, 0, 1  // 通道 3
//     };
//     // Tensor *input = create_tensor((int[]){3, 4, 4}, 3);
//     Tensor *input = create_tensor((int[]){3, 4, 4}, 3);
//     init_tensor(input, input_data);
//     print_tensor(input);
//     printf("\n");

//     Tensor *output = tensor_squeeze(input, 0);
//     print_tensor(output);

//     // Tensor *output = tensor_slice(input, (int[]){0, 0, 0}, (int[]){2, 4, 4});
//     // print_tensor(output);

//     return 0;
// }

// 测试concatenate函数
// int main() {
//     // 输入参数
//     int in_channels = 3, in_h = 4, in_w = 4;
//     float input_data1[3 * 2 * 4] = {
//         1, 2, 3, 4,   5, 6, 7, 8,  // 通道 1
//         1, 1, 1, 1,   1, 1, 1, 1,  // 通道 2
//         0, 1, 0, 1,   0, 1, 0, 1,  // 通道 3
//     };
//     float input_data2[1 * 2 * 4] = {
//         9, 10, 11, 12,  13, 14, 15, 16, // 通道 1
//     };
//     // Tensor *input = create_tensor((int[]){3, 4, 4}, 3);
//     Tensor *input1 = create_tensor((int[]){3, 2, 4}, 3);
//     init_tensor(input1, input_data1);
//     Tensor *input2 = create_tensor((int[]){1, 2, 4}, 3);
//     init_tensor(input2, input_data2);
//     print_tensor(input1);
//     printf("\n");
//     print_tensor(input2);
//     printf("\n");

//     Tensor *output = concatenate(input1, input2, 0);
//     print_tensor(output);
//     printf("\n");
//     return 0;
// }

// 测试GRU
// int main() {
//     Tensor *input = create_tensor((int[]){2, 2}, 2);
//     GRULayer *gru = create_gru_layer(2, 2);
//     Tensor *hidden_state = create_tensor((int[]){2}, 1);
//     Tensor *output = gru_forward(gru, input, hidden_state);
//     print_tensor_shape(output);
//     return 0;
// }

// 测试pad
// int main() {
//     int shape_in[2] = {3, 4};
//     Tensor *t = create_tensor(shape_in, 2);
//     print_tensor_shape(t);
//     printf("%d\n", t->size);
//     float data[] = {
//         1, 2, 3, 4,
//         4, 5, 6, 7,
//         8, 9, 10, 11
//     };
//     init_tensor(t, data);
//     print_tensor(t);
//     printf("\n");
//     int pad[4] = {1, 0, 1, 2};
//     Tensor* o = tensor_pad(t, pad);
//     print_tensor(o);
//     return 0;
// }

// 测试permute
// int main() {
//     int shape_in[2] = {3, 4};
//     Tensor *t = create_tensor(shape_in, 2);
//     print_tensor_shape(t);
//     printf("%d\n", t->size);
//     float data[] = {
//         1, 2, 3, 4,
//         4, 5, 6, 7,
//         8, 9, 10, 11
//     };
//     init_tensor(t, data);
//     print_tensor(t);

//     Tensor* o = permute(t, (int[]){1, 0});
//     print_tensor(o);
//     return 0;
// }

// 测试decoder
// int main() {
//     Tensor *input = create_tensor((int[]){8, 70, 55}, 3);
//     Tensor *mid = create_tensor((int[]){8, 70, 54}, 3);
//     DecoderBlock* dec1 = create_decoder_block(8, 1, 1, 1);
//     Tensor* output = decoderblock_forward(dec1, input, mid);
//     print_tensor_shape(output);
//     return 0;
// }

// 测试正逆变换
// int main() {
//     const char *filename = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *wav_ = read_wav_file(filename, &num_samples, &sample_rate);
//     wav_norm(wav_, num_samples);
//     Tensor *wav = create_tensor((int[]){num_samples}, 1);
//     init_tensor(wav, wav_);
//     int length = num_samples / FRAME_SIZE - 1;
//     Tensor* cspecs = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
//     Tensor* features = create_tensor((int[]){1, length, NB_FEATURES}, 3);
//     feature_extract(wav, cspecs, features);

//     const char *filename_out = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/test.wav";
//     float* out_wav = istft(cspecs, num_samples);
//     wav_invnorm(out_wav, num_samples);
//     if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }

//     return 0;
// }

// 测试整个模型
int main() {
    // *** stage 1: 读取文件 ***
    const char *filename_mic = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
    const char *filename_ref = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav";
    const char *filename_e = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/e.wav";
    const char *filename_y = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/y.wav";
    int num_samples;                        // 样本数
    int sample_rate;                        // 采样率
    float *mic_ = read_wav_file(filename_mic, &num_samples, &sample_rate);
    float *ref_ = read_wav_file(filename_ref, &num_samples, &sample_rate);

    float *e_ = (float*)calloc(num_samples, sizeof(float));
    float *y_ = (float*)calloc(num_samples, sizeof(float));

    printf("stage 1 executed successfully.\n");

    // *** stage 2: 线性滤波与特征提取 ***
    // num_samples = 18000;  // 目前只处理18000个采样点
    pfdkf(ref_, mic_, e_, y_, num_samples);

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
    // const char *cpt = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/test_dict.txt";
    const char *cpt = "/home/node25_tmpdata/xcli/percepnet/c_aec/model_weights.json";
    // ModelStateDict *sd = create_model_state_dict();
    // parse_ckpt(cpt, sd);
    ModelStateDict *sd = parse_json_to_parameters(cpt);

    printf("stage 3 executed successfully.\n");

    // *** stage 4: 构建模型并加载模型参数 ***

    // EncoderBlock *enc1 = create_encoder_block(1, 8);
    RNNVQE *model = create_rnnvqe();

    Parameter *params = sd->params;
    // encoderblock_load_params(enc1, params);
    rnnvqe_load_params(model, sd);

    free_model_state_dict(sd);

    printf("stage 4 executed successfully.\n");

    // *** stage 5: 模型推理 ***
    printf("input shape:\n");
    print_tensor_shape(features_mic);
    // Tensor *mid1 = depthwise_conv2d_forward(conv2d, features_mic);
    // Tensor *mid2 = batchnorm_forward(bn, mid1);
    // Tensor *outputs = elu_forward(elu, mid2);
    // Tensor *outputs = encoderblock_forward(enc1, features_mic);
    Tensor *gains = rnnvqe_forward(model, features_mic, features_y);
    printf("gains shape:\n");
    print_tensor_shape(gains);

    float* out_wav = (float*)malloc(num_samples*sizeof(float));
    post_process(cspecs_mic, gains, out_wav);

    wav_invnorm(out_wav, num_samples);

    const char *filename_out = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/test_last.wav";
    if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
        printf("e WAV file written successfully.\n");
    } else {
        printf("Failed to write WAV file.\n");
    }

    // Tensor *feats_in1 = permute(output, (int[]){1, 0, 2}); // [T, C, F]
    // int T = feats_in1->shape[0], C = feats_in1->shape[1], F = feats_in1->shape[2];
    // Tensor *outputs = reshape(feats_in1, (int[]){T, C*F}, 2); // [T, C*F]
    // Tensor *o1 = tensor_slice(output, (int[]){0, 0, 0}, (int[]){1, output->shape[1], output->shape[2]});
    // Tensor *o2 = tensor_squeeze(o1, 0);

    // const char *output_file = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/output.txt";
    // FILE *file = fopen(output_file, "w");
    // if (!file) {
    //     perror("Error opening output file");
    //     return 1;
    // }

    // int h = o2->shape[0], w = o2->shape[1];
    // for (int t = 0; t < h; t ++ ) {
    //     for (int f = 0; f < w; f ++ ) {
    //         float tfbin = tensor_get(o2, (int[]){t, f});
    //         fprintf(file, "%f ", tfbin);
    //     }
    //     fprintf(file, "\n");
    // }

    // fclose(file); // 关闭文件

    return 0;
}