#ifndef WAVREADER_H
#define WAVREADER_H

float* read_wav_file(const char *filename, int *num_samples, int *sample_rate);
void wav_norm(float* audio_data, int num_samples);

#endif