#ifndef RNNOISE_H
#define RNNOISE_H
#include "typedef.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
  float analysis_mem[WINDOW_SIZE-FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[WINDOW_SIZE-FRAME_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
} DenoiseState;

int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

int rnnoise_init(DenoiseState *st) {
  memset(st, 0, sizeof(*st));

  return 0;
}

DenoiseState *rnnoise_create() {
  DenoiseState *st;
  st = (DenoiseState*)malloc(rnnoise_get_size());
  rnnoise_init(st);
  return st;
}

#endif