/*
 *  Copyright (c) 2003-2026, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_TUNER_H
#define KISS_TUNER_H

#include "kiss_fftr.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Frame-by-frame fundamental frequency (f0) estimator built on kiss_fftr.
 *
 * Per-frame pipeline:
 *   1. spectral peak pick (Hann-windowed magnitude spectrum)
 *   2. parabolic interpolation around the peak bin (sub-bin accuracy)
 *   3. median smoothing over a configurable window of recent estimates
 *   4. octave correction (sub-harmonic spectral check)
 *
 * The kiss_fft / kiss_fftr cores are used read-only; their outputs are
 * bit-for-bit identical to using them directly.
 */

typedef enum {
    KISS_TUNER_SILENCE = 0, /* frame energy below silence threshold, f0 invalid */
    KISS_TUNER_OK = 1       /* voiced frame, *f0_hz holds the estimate */
} kiss_tuner_status;

typedef struct kiss_tuner_state *kiss_tuner_cfg;

/*
 * nfft:          frame size (must be even, as required by kiss_fftr)
 * sample_rate:   samples per second
 * median_window: number of recent raw estimates used for median smoothing;
 *                values < 1 are clamped to 1, even values are rounded up
 *                to the next odd number, values > 63 are clamped to 63
 *
 * If you don't care to allocate space, use mem = lenmem = NULL.
 * Returns NULL on invalid arguments (e.g. odd nfft, sample_rate <= 0).
 */
kiss_tuner_cfg KISS_FFT_API kiss_tuner_alloc(int nfft, int sample_rate, int median_window, void *mem, size_t *lenmem);

/*
 * Process one frame of nfft samples.
 * Returns KISS_TUNER_SILENCE for silent frames (*f0_hz set to 0),
 * otherwise KISS_TUNER_OK and *f0_hz holds the estimated fundamental in Hz.
 */
kiss_tuner_status KISS_FFT_API kiss_tuner_process(kiss_tuner_cfg cfg, const kiss_fft_scalar *frame, float *f0_hz);

/* Override the silence threshold (RMS of the input frame). Default 1e-4. */
void KISS_FFT_API kiss_tuner_set_silence_threshold(kiss_tuner_cfg cfg, float rms_threshold);

void KISS_FFT_API kiss_tuner_free(kiss_tuner_cfg cfg);

#ifdef __cplusplus
}
#endif

#endif /* KISS_TUNER_H */
