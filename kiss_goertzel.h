/*
 *  Real-time keypress tone detection on top of KISS FFT.
 *
 *  Three-stage pipeline:
 *    1. framing      - input stream is split into overlapping frames
 *                      (configurable frame length + hop)
 *    2. multi-Goertzel - each frame is evaluated only at the configured
 *                      target frequencies (no full FFT needed)
 *    3. peak decision  - a tone is reported when its spectral power
 *                      fraction (confidence) crosses a configurable
 *                      threshold; silence and broadband noise frames
 *                      are rejected
 *
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_GOERTZEL_H
#define KISS_GOERTZEL_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of target frequencies / detections per frame. */
#define KISS_GOERTZEL_MAX_DET 16

typedef struct kiss_goertzel_state *kiss_goertzel_cfg;

typedef struct {
    int frame_index;     /* 0-based index of this frame in the stream      */
    int sample_offset;   /* offset (in samples) of the frame start         */
    int is_silence;      /* 1 if the frame was rejected as silence         */
    int n_det;           /* number of detected target tones (0..MAX_DET)   */
    int det_id[KISS_GOERTZEL_MAX_DET];   /* index into the cfg freq list  */
    float det_freq[KISS_GOERTZEL_MAX_DET]; /* detected frequency, Hz      */
    float det_conf[KISS_GOERTZEL_MAX_DET]; /* confidence, 0..1            */
} kiss_goertzel_frame_result;

/*
 * Allocate a detector.
 *
 *   frame_len    samples per frame (>= 8)
 *   hop          samples between consecutive frame starts (>= 1)
 *   sample_rate  sample rate in Hz (> 0)
 *   freqs        target frequencies in Hz, each in (0, sample_rate/2)
 *   nfreqs       number of target frequencies (1..KISS_GOERTZEL_MAX_DET)
 *   threshold    detection threshold: minimum confidence, (0, 1]
 *   silence_db   frames with RMS below 10^(silence_db/20) of full scale
 *                are rejected as silence; must be in [-400, 0)
 *
 * Returns NULL if any parameter is invalid.
 */
kiss_goertzel_cfg KISS_FFT_API kiss_goertzel_alloc(
    int frame_len, int hop, float sample_rate,
    const float *freqs, int nfreqs,
    float threshold, float silence_db);

/*
 * Run the detector over a stream of samples.
 *
 * Frames are taken at offsets 0, hop, 2*hop, ... while a full frame
 * fits inside [0, nsamples). At most max_results frames are written
 * to results[].
 *
 * Returns the number of frames processed (>= 0),
 * or a negative value on invalid arguments.
 */
int KISS_FFT_API kiss_goertzel_process(
    kiss_goertzel_cfg cfg,
    const kiss_fft_scalar *samples, int nsamples,
    kiss_goertzel_frame_result *results, int max_results);

/*
 * Raw Goertzel power |X(f)|^2 of one target tone on one frame
 * (frame must point to frame_len samples). Analysis/debug aid.
 * Returns -1 on invalid arguments.
 */
double KISS_FFT_API kiss_goertzel_tone_power(
    kiss_goertzel_cfg cfg,
    const kiss_fft_scalar *frame, int tone_index);

/*
 * Upper bound on the difference between the Goertzel coefficient
 * computed here and the corresponding bin of a full kiss_fft/kiss_fftr
 * transform of the same frame:
 *
 *   |X_goertzel(f) - X_fft(k)| <= bound,   k = round(f*N/fs)
 *
 *   bound = (2*N*u_goertzel + 6*ceil(log2(N))*u_fft) * sum_abs_x
 *
 * where sum_abs_x = sum |x[n]| over the frame, u_goertzel is the
 * double-precision unit roundoff (the recurrence is evaluated in
 * double) and u_fft is the unit roundoff of kiss_fft_scalar.
 * The corresponding power-spectrum bound is
 *   |P_goertzel - P_fft| <= bound * (2*|X_fft| + bound).
 */
double KISS_FFT_API kiss_goertzel_error_bound(
    kiss_goertzel_cfg cfg, double sum_abs_x);

void KISS_FFT_API kiss_goertzel_free(kiss_goertzel_cfg cfg);

#ifdef __cplusplus
}
#endif
#endif
