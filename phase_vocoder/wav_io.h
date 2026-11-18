/*
 * Minimal mono WAV reader/writer (PCM 16/24/32-bit and IEEE float32).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef WAV_IO_H
#define WAV_IO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int sample_rate;
    int channels;
    long num_frames;       /* frames per channel */
    float *data;           /* interleaved float samples in [-1, 1] */
} wav_file;

/* Reads a mono-compatible WAV; multi-channel files are downmixed.
   Returns 0 on success, -1 on failure. Free with wav_free(). */
int wav_read(const char *path, wav_file *w);

/* Writes a mono 16-bit PCM WAV. Returns 0 on success, -1 on failure. */
int wav_write_mono_pcm16(const char *path, const float *data,
                         long num_samples, int sample_rate);

void wav_free(wav_file *w);

#ifdef __cplusplus
}
#endif

#endif /* WAV_IO_H */
