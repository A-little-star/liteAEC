#include "../include/depthwise_conv2d.h"
#include "../include/batchnorm.h"
#include "../include/elu.h"
#include "../include/tensor.h"
#include "../include/model.h"

typedef struct {
    EncoderBlock *mic_enc[2];
    EncoderBlock *ref_enc;
} RNNVQE;

RNNVQE* create_rnnvqe() {
    int encoder_mic_channels[] = {1, 8, 16, 24, 32};
    int encoder_ref_channels[] = {1, 8};
    RNNVQE *model = (RNNVQE*)malloc(sizeof(RNNVQE));
    model->mic_enc[0] = create_encoder_block(encoder_mic_channels[0], encoder_mic_channels[1]);
    model->ref_enc = create_encoder_block(encoder_ref_channels[0], encoder_ref_channels[1]);
    model->mic_enc[1] = create_encoder_block(encoder_mic_channels[1], encoder_mic_channels[2]);
    // model->mic_enc[2] = create_encoder_block(encoder_mic_channels[2], encoder_mic_channels[3]);
    // model->mic_enc[3] = create_encoder_block(encoder_mic_channels[3], encoder_mic_channels[4]);
    
    return model;
}

Parameter* rnnvqe_load_params(RNNVQE *model, ModelStateDict *sd) {
    Parameter *params = sd->params;
    Parameter *p1 = encoderblock_load_params(model->mic_enc[0], params);
    Parameter *p2 = encoderblock_load_params(model->mic_enc[1], p1);
    Parameter *p3 = encoderblock_load_params(model->ref_enc, p2);

    assert(p3 == sd->params + sd->size);
    return p3;
}

Tensor *rnnvqe_forward(RNNVQE *model, Tensor *mic, Tensor *ref) {
    Tensor *mid1 = encoderblock_forward(model->mic_enc[0], mic);
    Tensor *mid2 = encoderblock_forward(model->ref_enc, ref);
}