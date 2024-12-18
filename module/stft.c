#include "../include/_kiss_fft_guts.h"
#include "../include/typedef.h"
#include "../include/kiss_fft.h"
#include "../include/tensor.h"
#include "../include/rnnoise.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>

typedef struct {
  int init;
  kiss_fft_state *kfft;
  float window[WINDOW_SIZE];
} CommonState;

CommonState common;

static void check_init() {
  int i;
  if (common.init) return;
  // common.kfft = opus_fft_alloc_twiddles(WINDOW_SIZE, NULL, NULL, NULL, 0);
  common.kfft = opus_fft_alloc_twiddles(FFT_LEN, NULL, NULL, NULL, 0);

#ifdef PROCESSUNIT10MS
  for(i=0; i<(FRAME_SIZE);i++)
  {
    common.window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  common.window[FFT_LEN -1 -i] = common.window[i];
  }
#else 
  for (i=0;i<(WINDOW_SIZE);i++)
    common.window[i] = kBlocks320w512[i];
 #endif 
 
  common.init = 1;
}

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[FFT_LEN];
  kiss_fft_cpx y[FFT_LEN];
  check_init();
  for (i=0;i<FFT_LEN;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[FFT_LEN];
  kiss_fft_cpx y[FFT_LEN];
  check_init();
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<FFT_LEN;i++) {
    x[i].r = x[FFT_LEN - i].r;
    x[i].i = -x[FFT_LEN - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = FFT_LEN*y[0].r;
  for (i=1;i<FFT_LEN;i++) {
    out[i] = FFT_LEN*y[FFT_LEN - i].r;
  }
}

//Vorbis windows
static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i] *= common.window[i];
  
  }
}

void feature_extract(Tensor* input, Tensor* cspecs, Tensor* features) {
  assert(input->ndim == 1);
  int length = input->size;

  int win_len = WINDOW_SIZE;
  int hop_len = FRAME_SIZE;

  DenoiseState *noisy;
  noisy = rnnoise_create();

  for (int s = 0, num_frame = 0; s + win_len < length; s += hop_len, num_frame ++ ) {
    float x[FFT_LEN];
    kiss_fft_cpx X[FREQ_SIZE];
    float Ex[NB_BANDS];
    float feature[NB_FEATURES];

    for (int i = 0; i < WINDOW_SIZE; i ++ )
      if (s + i >= 0)
        x[i] = input->data[s + i];
      else
        x[i] = 0;
    for (int i = WINDOW_SIZE; i < FFT_LEN; i ++ )
      x[i] = 0;
    
    apply_window(x);
    
    forward_transform(X, x);

    compute_frame_features(noisy, X, Ex, feature);

    // 这里使用memcpy会更快一些，为了方便调试，先使用赋值的方法
    for (int i = 0; i < FREQ_SIZE; i ++ ) {

      tensor_set(cspecs, (int[]){0, num_frame, i}, X[i].r);
      tensor_set(cspecs, (int[]){1, num_frame, i}, X[i].i);
      // set_value(cspecs, 0, num_frame, i, (float)X[i].r);
      // set_value(cspecs, 1, num_frame, i, X[i].i);
    }
    for (int i = 0; i < NB_FEATURES; i ++ ) {
      tensor_set(features, (int[]){0, num_frame, i}, feature[i]);
      // set_value(features, 0, num_frame, i, feature[i]);
    }
  }
  return;
}

void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex) {
  compute_band_energy(Ex, X);
}

int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, 
                                  float *Ex, float *features) {
  int i;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2;
  float spec_variability = 0;
  float Ly[NB_BANDS];
  float *(pre[1]);
  float follow, logMax;

  //calc energy stored in Ex, in triangle mode in bark or opus spectrum
  frame_analysis(st, X, Ex);

  logMax = -2;
  follow = -2;

  //xuefeifang:slow trace when Ly is decrease ??
  //Ly[i] will increase instantly; and Ly[i] will decrease for a certain time
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    // Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    // logMax = MAX16(logMax, Ly[i]);
    // follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }

  for(int i = 0; i < NB_BANDS; i++) {features[i] = Ly[i];}
    
  // mean value
  // xuefei: what's the mean of these features
  // delta cepstral ceps_0 is the current(time T) cep coef & ceps_1 is the T-1 cep coef & ceps_2 is the T-2 cep coef
  features[0] -= 12;
  features[1] -= 4;
  ceps_0 = st->cepstral_mem[st->memid];
  ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];
  ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];
  for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
  st->memid++;
  for (i=0;i<NB_DELTA_CEPS;i++) {
    // features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
    features[NB_BANDS+i] = ceps_0[i] - ceps_2[i];
    features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
  }
  /* Spectral variability features. */
  if (st->memid == CEPS_MEM) st->memid = 0;
  for (i=0;i<CEPS_MEM;i++)
  {
    int j;
    float mindist = 1e15f;
    for (j=0;j<CEPS_MEM;j++)
    {
      int k;
      float dist=0;
      for (k=0;k<NB_BANDS;k++)
      {
        float tmp;
        tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
        dist += tmp*tmp;
      }
      if (j!=i)
        mindist = MIN32(mindist, dist);
    }
    spec_variability += mindist;
  }

  return E < 0.1;
}