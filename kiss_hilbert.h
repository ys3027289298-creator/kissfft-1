/*
 *  Copyright (c) 2026, Kiss FFT contributors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Hilbert transform, analytic signal and instantaneous-frequency tracking.
 *
 * The analytic signal z[n] of a real frame x[n] is built in the frequency
 * domain: the FFT spectrum is zeroed below DC, doubled at the positive
 * frequencies and left untouched at DC / Nyquist, then transformed back.
 * The instantaneous frequency follows from the phase difference of two
 * adjacent analytic samples:
 *
 *     f_inst[n] = arg(z[n+1] * conj(z[n])) / (2*pi)   [cycles / sample]
 *
 * atan2 already returns the wrapped phase difference, so no explicit phase
 * unwrapping is needed and the result is correct for every true frequency
 * strictly below Nyquist (|f_inst| < 0.5 cycles/sample).
 *
 * Edge handling
 * -------------
 * Every finite frame is windowed with a Hann window and zero padded before
 * the transform. The window makes both frame edges C1-continuous with the
 * surrounding zeros, which removes the spectral leakage a hard truncation
 * causes. Mirror (reflection) extension is deliberately NOT used: reflecting
 * an oscillating signal reverses the time axis at the fold, so the analytic
 * phase at the fold is a local frequency reversal that reads as a bogus
 * instantaneous frequency (measured 40%-200% error, strongly dependent on
 * the signal phase at the edge). Zero padding has no such artifact.
 *
 * The price of windowing is that the few samples right at either edge have a
 * vanishing signal amplitude, so their phase (and hence IF) is numerically
 * unreliable. The valid region depends on the signal: roughly one signal
 * period has to be discarded at each edge. kiss_hilbert_track therefore
 * reports a per-frame valid margin derived from the lowest instantaneous
 * frequency actually present in the frame (the low-frequency end of a sweep
 * is the limiting case). Entries [margin, nfft-1-margin) of f_inst meet the
 * 0.5% accuracy envelope. For continuous tracking, feed overlapping frames
 * with a hop of nfft - 2*margin so every output sample stays in a trusted
 * region.
 *
 * Floating-point builds only: the phase tracking uses atan2 and the
 * frequency-domain Hilbert filter has no fixed-point equivalent here.
 */

#ifndef KISS_HILBERT_H
#define KISS_HILBERT_H

#if defined(FIXED_POINT)
#error "kiss_hilbert: instantaneous-frequency tracking requires a floating-point build (float or double)"
#endif

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiss_hilbert_state *kiss_hilbert_cfg;

/*
 * Allocate a Hilbert / analytic-signal state for frames of nfft real samples.
 *
 * Each frame is Hann windowed and zero padded to the next fast FFT size at
 * or least nfft + 2*zero_pad. zero_pad may be 0; it only interpolates the
 * spectrum and does not change the edge behavior, which the Hann window
 * already controls.
 *
 * The usual kissfft mem/lenmem protocol applies: mem == NULL allocates a
 * single contiguous block (release it with kiss_hilbert_free), otherwise the
 * required size is reported through lenmem.
 */
kiss_hilbert_cfg KISS_FFT_API kiss_hilbert_alloc(int nfft, int zero_pad,
                                                 void *mem, size_t *lenmem);

/*
 * Hann-window the real frame x (nfft samples), zero pad it and compute its
 * analytic signal. On return z holds nfft complex samples; z[n].r equals the
 * windowed x[n].
 */
void KISS_FFT_API kiss_hilbert_analytic(kiss_hilbert_cfg st,
                                        const kiss_fft_scalar *x,
                                        kiss_fft_cpx *z);

/*
 * Instantaneous frequency from analytic-signal phase differences, in cycles
 * per sample (multiply by the sampling rate for Hz). f_inst has nfft-1
 * entries: f_inst[n] tracks the frequency between z[n] and z[n+1], i.e.
 * sample position n+0.5. Safe for any input, including a zero-magnitude z
 * (a zero sample yields 0 instead of an atan2 NaN).
 */
void KISS_FFT_API kiss_hilbert_instfreq(kiss_hilbert_cfg st,
                                        const kiss_fft_cpx *z,
                                        kiss_fft_scalar *f_inst);

/*
 * One-shot convenience: window + zero pad + analytic signal + phase
 * difference, combining the two calls above. z may be NULL when only the
 * instantaneous frequency is needed (an internal buffer is used).
 *
 * On return *valid_margin is the number of untrusted samples at each edge of
 * f_inst for this frame (entries [valid_margin, nfft-1-valid_margin) are in
 * the 0.5% accuracy envelope). valid_margin may be NULL if the caller does
 * not need it.
 */
void KISS_FFT_API kiss_hilbert_track(kiss_hilbert_cfg st,
                                     const kiss_fft_scalar *x,
                                     kiss_fft_cpx *z,
                                     kiss_fft_scalar *f_inst,
                                     int *valid_margin);

/*
 * Valid margin for a frame from its already-computed instantaneous frequency,
 * using the same estimator as kiss_hilbert_track. The margin is the number of
 * edge samples to discard at each end of f_inst (which has nfft-1 entries).
 */
int KISS_FFT_API kiss_hilbert_valid_margin(kiss_hilbert_cfg st,
                                           const kiss_fft_scalar *f_inst);

#define kiss_hilbert_free KISS_FFT_FREE

#ifdef __cplusplus
}
#endif

#endif /* KISS_HILBERT_H */
