#ifndef STFT_H
#define STFT_H
#include "tensor.h"

void feature_extract(Tensor* input, Tensor* cspecs, Tensor* features);
float* istft(Tensor* cspecs, int num_samples);
void post_process(Tensor* cspecs, Tensor* gains, float* outputs_ptr);

#endif