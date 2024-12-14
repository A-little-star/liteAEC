#ifndef STFT_H
#define STFT_H
#include "matrix_op.h"

void feature_extract(Tensor* input, Tensor* cspecs, Tensor* features);

#endif