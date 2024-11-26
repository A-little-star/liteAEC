#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "./include/kiss_fft.h"
#include "./include/_kiss_fft_guts.h"
#include "./include/typedef.h"
#include <math.h>

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