#include <assert.h>
#include "../include/depthwise_conv2d.h"
#include "../include/batchnorm.h"
#include "../include/elu.h"
#include "../include/tensor.h"
#include "../include/model.h"

RNNVQE* create_rnnvqe() {
    int encoder_mic_channels[] = {1, 8, 16, 24, 32};
    int encoder_ref_channels[] = {1, 8};
    RNNVQE *model = (RNNVQE*)malloc(sizeof(RNNVQE));
    model->mic_enc[0] = create_encoder_block(encoder_mic_channels[0], encoder_mic_channels[1]);
    model->ref_enc = create_encoder_block(encoder_ref_channels[0], encoder_ref_channels[1]);
    model->mic_enc[1] = create_encoder_block(encoder_mic_channels[1] + encoder_ref_channels[1], encoder_mic_channels[2]);
    model->mic_enc[2] = create_encoder_block(encoder_mic_channels[2], encoder_mic_channels[3]);
    model->mic_enc[3] = create_encoder_block(encoder_mic_channels[3], encoder_mic_channels[4]);
    
    return model;
}

Parameter* rnnvqe_load_params(RNNVQE *model, ModelStateDict *sd) {
    Parameter *params = sd->params;
    Parameter *p1 = encoderblock_load_params(model->mic_enc[0], params);
    Parameter *p2 = encoderblock_load_params(model->mic_enc[1], p1);
    Parameter *p3 = encoderblock_load_params(model->mic_enc[2], p2);
    Parameter *p4 = encoderblock_load_params(model->mic_enc[3], p3);
    Parameter *p5 = encoderblock_load_params(model->ref_enc, p4);

    assert(p5 == sd->params + sd->size);
    return p5;
}

void free_rnnvqe(RNNVQE *model) {
    for (int i = 0; i < 4; i ++ ) {
        free_encoder_block(model->mic_enc[i]);
    }
    free_encoder_block(model->ref_enc);

    free(model);
}

Tensor *rnnvqe_forward(RNNVQE *model, Tensor *mic, Tensor *ref) {
    Tensor *mic_enc1_out = encoderblock_forward(model->mic_enc[0], mic);
    Tensor *mic_ref1_out = encoderblock_forward(model->ref_enc, ref);
    Tensor *enc1_out_cat = concatenate(mic_ref1_out, mic_enc1_out, 0);
    // delete_tensor(mid_mic_1);
    // delete_tensor(mid_ref_1);
    Tensor *enc2_out = encoderblock_forward(model->mic_enc[1], enc1_out_cat);
    Tensor *enc3_out = encoderblock_forward(model->mic_enc[2], enc2_out);
    Tensor *enc4_out = encoderblock_forward(model->mic_enc[3], enc3_out);
    // delete_tensor(mid_mic_cat);

    return enc3_out;
}