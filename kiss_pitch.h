/*
 *  Copyright (c) 2003-2026, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_PITCH_H
#define KISS_PITCH_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Spectral peak detection and harmonic tracking on top of kiss_fftr.
 *
 * Per frame the analysis:
 *   1. windows the frame with a periodic Hann window and takes a real FFT
 *   2. picks the local maxima of the magnitude spectrum inside [fmin,fmax]
 *   3. refines every peak position with parabolic interpolation, which for a
 *      Hann-windowed sinusoid is accurate to well below a cent
 *   4. links peaks into trajectories ("tracks") that persist across frames
 *   5. takes each peak frequency divided by 1..max_harmonics as a candidate
 *      fundamental, scores candidates by how many partials they explain,
 *      and refines the winner by weighted least squares
 *   6. reports f0, a 0..1 confidence, and every peak with its harmonic number
 *      and track id
 *
 * Once a fundamental has been established, an octave jump has to win on score
 * for several consecutive frames before the analysis follows it. That keeps a
 * steady tone from flipping between octaves on noisy or phase-shifted frames,
 * while still allowing a real note change to be picked up.
 *
 * The analysis reads individual sample values, so it is only available when
 * kiss_fft_scalar is a real scalar type (float, double, int16_t, int32_t).
 * With USE_SIMD the scalar is a packed vector; kiss_pitch_alloc() then
 * returns NULL and kiss_pitch_available() returns 0.
 */

typedef struct kiss_pitch_state *kiss_pitch_cfg;

/* 1 when this build can analyse frames, 0 with USE_SIMD */
int KISS_FFT_API kiss_pitch_available(void);

typedef struct {
    double freq_hz;   /* interpolated peak frequency                        */
    double magnitude; /* peak magnitude relative to the strongest peak, 0..1 */
    int bin;          /* FFT bin holding the magnitude maximum              */
    int harmonic;     /* 1..max_harmonics when explained by f0, else 0      */
    int track_id;     /* >= 0, stable while the peak keeps being tracked    */
    int track_age;    /* 1 for a new track, +1 for every further frame      */
} kiss_pitch_peak;

typedef struct {
    int voiced;        /* 1 when the frame has a confident fundamental      */
    double f0_hz;      /* estimated fundamental, 0 when unvoiced            */
    double confidence; /* 0..1, share of in-band power on the harmonic comb */
    double rms;        /* RMS level of the input frame                      */
    int n_peaks;       /* number of detected peaks (peaks array may be cut) */
} kiss_pitch_frame;

/*
 * nfft:        frame size, must be even (kiss_fftr requirement)
 * sample_rate: samples per second, must be positive
 *
 * Defaults: 50..2000 Hz search range (clamped to what nfft can resolve), up
 * to 16 harmonics, peaks more than 60 dB below the strongest in-band peak
 * ignored, confidence threshold 0.5.
 *
 * If you don't care to allocate space, use mem = lenmem = NULL.
 * Returns NULL on invalid arguments or when the analysis is unavailable
 * (USE_SIMD build, nfft too small for the search range).
 */
kiss_pitch_cfg KISS_FFT_API kiss_pitch_alloc(int nfft, double sample_rate,
                                             void *mem, size_t *lenmem);

/*
 * Analyse one frame of nfft samples.
 *
 * On return *out describes the frame. Up to max_peaks detected peaks, sorted
 * by increasing frequency, are written to peaks (peaks may be NULL when
 * max_peaks is 0 or when the peak list is not needed) and out->n_peaks holds
 * the number of peaks actually written.
 *
 * Returns 1 when the frame is voiced, 0 otherwise.
 */
int KISS_FFT_API kiss_pitch_process(kiss_pitch_cfg cfg,
                                    const kiss_fft_scalar *frame,
                                    kiss_pitch_frame *out,
                                    kiss_pitch_peak *peaks,
                                    int max_peaks);

/* Narrow or widen the fundamental search range. Returns 1 on success. */
int KISS_FFT_API kiss_pitch_set_range(kiss_pitch_cfg cfg, double fmin, double fmax);

/* Set the maximum harmonic number considered, 1..64. Returns 1 on success. */
int KISS_FFT_API kiss_pitch_set_max_harmonics(kiss_pitch_cfg cfg, int max_harmonics);

/*
 * floor_db:        peak detection floor relative to the strongest in-band
 *                  peak, in dB (values > 0 are ignored; -60 keeps weak
 *                  partials)
 * min_confidence:  frames whose confidence is below this are unvoiced;
 *                  values outside 0..1 are ignored
 */
void KISS_FFT_API kiss_pitch_set_detection(kiss_pitch_cfg cfg,
                                           double floor_db,
                                           double min_confidence);

/* Forget tracking history, the established f0 and all tracks. */
void KISS_FFT_API kiss_pitch_reset(kiss_pitch_cfg cfg);

#define kiss_pitch_free KISS_FFT_FREE

#ifdef __cplusplus
}
#endif

#endif
