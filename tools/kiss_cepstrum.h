/*
 *  Cepstral pitch detection built on the KISS FFT real transforms.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_CEPSTRUM_H
#define KISS_CEPSTRUM_H

#include "kiss_fft.h"

/* The cepstrum needs log() and therefore a scalar floating-point build. */
#if defined(FIXED_POINT) || defined(USE_SIMD)
#error "kiss_cepstrum requires a scalar floating-point build (float or double)"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiss_cepstrum_state *kiss_cepstrum_cfg;

/*
 * Return codes for kiss_cepstrum_detect().
 */
#define KISS_CEPSTRUM_VOICED    1   /* periodic frame, a pitch was found */
#define KISS_CEPSTRUM_UNVOICED  0   /* audible frame but no periodic peak */
#define KISS_CEPSTRUM_SILENCE  (-1) /* frame energy below the silence floor */

typedef struct {
    int nfft;             /* FFT size, must be even and positive          */
    int sample_rate;      /* sampling rate in Hz                          */
    int min_pitch_hz;     /* lowest pitch searched for                    */
    int max_pitch_hz;     /* highest pitch searched for                   */
    int lifter_order;     /* high-time lifter cutoff in quefrency samples:
                             cepstrum bins [0, lifter_order) are zeroed,
                             as is their mirror. 0 disables liftering.    */
    kiss_fft_scalar silence_rms;       /* RMS below this means silence    */
    kiss_fft_scalar voicing_threshold; /* cepstral peak below this means
                                          an unvoiced frame               */
} kiss_cepstrum_config;

typedef struct {
    int voiced;             /* 1 when voiced, 0 for silence/unvoiced      */
    int quefrency_bin;      /* peak position in samples, 0 when unvoiced  */
    kiss_fft_scalar pitch_hz; /* sample_rate / quefrency_bin, 0 unvoiced  */
    kiss_fft_scalar rms;    /* RMS of the analyzed frame                  */
    kiss_fft_scalar peak;   /* liftered cepstrum value at the peak        */
} kiss_cepstrum_pitch;

/*
 * Fill cfg with working defaults for the given FFT size and sample rate:
 *   min_pitch_hz     = 60
 *   max_pitch_hz     = 1000
 *   lifter_order     = 8 (half a millisecond at 16 kHz)
 *   silence_rms      = 1e-4
 *   voicing_threshold= 0.05
 */
void KISS_FFT_API kiss_cepstrum_default_config(kiss_cepstrum_config *cfg,
                                               int nfft, int sample_rate);

/*
 * Allocate a detector. Returns NULL when the configuration is invalid
 * (bad sizes, empty pitch search range, ...). All storage is obtained
 * through KISS_FFT_MALLOC and released with kiss_cepstrum_free().
 */
kiss_cepstrum_cfg KISS_FFT_API kiss_cepstrum_alloc(const kiss_cepstrum_config *cfg);

#define kiss_cepstrum_free KISS_FFT_FREE

/*
 * Change the high-time lifter order on an existing detector.
 * order == 0 disables liftering; values past the first searched quefrency
 * are clamped. Returns the order actually applied, or -1 on error.
 */
int KISS_FFT_API kiss_cepstrum_set_lifter_order(kiss_cepstrum_cfg st, int order);

/*
 * Full cepstrum pipeline for one nfft-sample frame:
 * Hann window -> forward real FFT -> log magnitude spectrum ->
 * inverse real FFT -> high-time liftering.
 * The frame RMS is returned and nfft cepstrum values are written to
 * cepstrum (which must hold nfft scalars).
 */
kiss_fft_scalar KISS_FFT_API
kiss_cepstrum_compute(kiss_cepstrum_cfg st,
                      const kiss_fft_scalar *frame,
                      kiss_fft_scalar *cepstrum);

/*
 * Locate the strongest cepstral peak inside the configured pitch range.
 * Returns the quefrency bin and stores the peak value in *peak when
 * non-NULL. The pitch in Hz is sample_rate / bin.
 */
int KISS_FFT_API kiss_cepstrum_find_peak(kiss_cepstrum_cfg st,
                                         const kiss_fft_scalar *cepstrum,
                                         kiss_fft_scalar *peak);

/*
 * One-call pitch detection. Silence is handled explicitly: frames whose
 * RMS is below silence_rms skip the transforms and come back as
 * KISS_CEPSTRUM_SILENCE with a zeroed result. Otherwise the cepstrum is
 * computed and the peak is compared against the voicing threshold.
 */
int KISS_FFT_API kiss_cepstrum_detect(kiss_cepstrum_cfg st,
                                      const kiss_fft_scalar *frame,
                                      kiss_cepstrum_pitch *result);

#ifdef __cplusplus
}
#endif
#endif /* KISS_CEPSTRUM_H */
