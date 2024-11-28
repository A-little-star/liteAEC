#include "../include/_kiss_fft_guts.h"
#include "../include/typedef.h"
#include "../include/kiss_fft.h"
#include "../include/matrix_op.h"
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

Tensor stft(Tensor input) {
  int channel = input.C, length = input.T, dim = input.F;
  assert((channel == 1) && (dim == 1));

  int win_len = WINDOW_SIZE;
  int hop_len = FRAME_SIZE;
  int fft_len = FFT_LEN;

  int num_frame = length / hop_len;
  Tensor cspecs = create_tensor(2, num_frame, FREQ_SIZE);  // 复数谱 [2, T, F]

  for (int s = 0, num_frame = 0; s + win_len < length; s += hop_len, num_frame ++ ) {
    float x[FFT_LEN];
    kiss_fft_cpx X[FREQ_SIZE];

    for (int i = 0; i < WINDOW_SIZE; i ++ )
      x[i] = input.data[s + i];
    for (int i = WINDOW_SIZE; i < FFT_LEN; i ++ )
      x[i] = 0;
    
    apply_window(x);
    if (num_frame == 300) {
      printf("wav: \n");
      for (int i = 0; i < FFT_LEN; i ++ )
        printf("%f ", x[i]);
      printf("\n");
    }
    
    forward_transform(X, x);

    for (int i = 0; i < FREQ_SIZE; i ++ ) {
      set_value(&cspecs, 0, num_frame, i, (float)X[i].r);
      set_value(&cspecs, 1, num_frame, i, X[i].i);
    }
  }
  return cspecs;
}