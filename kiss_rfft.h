/*
 *  Copyright (c) 2026, The kissfft authors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Canonical real FFT layer.
 *
 * Unlike kiss_fftr, which hands back the packed nfft/2+1 half spectrum,
 * this layer works with the *full* nfft-point complex spectrum and enforces
 * the Hermitian symmetry that a real time signal implies:
 *
 *     X[nfft-k] = conj(X[k])      for k = 1 .. nfft-1
 *     imag(X[0]) = 0
 *     imag(X[nfft/2]) = 0         (even nfft only)
 *
 * Conventions (numpy-style):
 *   - the forward transform is unnormalized,
 *   - the inverse transform divides by nfft, so rfft followed by irfft
 *     reproduces the original time samples.
 *
 * Only floating-point builds and 2,3,5-smooth transform sizes (see
 * kiss_fft_next_fast_size) are supported.
 */

#ifndef KISS_RFFT_H
#define KISS_RFFT_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiss_rfft_state *kiss_rfft_cfg;

/* Status codes returned by the canonical real FFT entry points. */
typedef enum {
    KISS_RFFT_OK = 0,
    KISS_RFFT_EALLOC,     /* configuration is NULL or an allocation failed */
    KISS_RFFT_ESIZE,      /* size is not a positive 2,3,5-smooth number */
    KISS_RFFT_EARG,       /* a required input/output pointer is NULL */
    KISS_RFFT_ESYMMETRY,  /* spectrum violates Hermitian symmetry */
    KISS_RFFT_EVALUE      /* non-finite value (NaN/Inf) encountered */
} kiss_rfft_status;

/* Up to this many violating index pairs are recorded per failed check. */
#define KISS_RFFT_MAX_VIOLATIONS 16

typedef struct {
    kiss_rfft_status code;
    /* Human-readable description, always NUL terminated. */
    char message[160];
    /* Number of symmetry violations actually recorded. */
    int n_violations;
    /*
     * For each violation:
     *   index[k] is the lower half index (0 .. nfft/2),
     *   mirror[k] is the index it must agree with (nfft - index[k]),
     *   abs_err[k] is the magnitude of the symmetry residual.
     * index == mirror marks a self-conjugate bin (DC / Nyquist).
     */
    int index[KISS_RFFT_MAX_VIOLATIONS];
    int mirror[KISS_RFFT_MAX_VIOLATIONS];
    double abs_err[KISS_RFFT_MAX_VIOLATIONS];
} kiss_rfft_error;

/*
 * Allocate a configuration usable for both the forward and inverse real
 * transforms of nfft real points.
 *
 * nfft must be positive and have no prime factor other than 2, 3 and 5.
 * The mem/lenmem arguments follow the same convention as kiss_fft_alloc:
 * pass NULL, NULL for a malloc'd configuration, or query the required size
 * with mem == NULL and lenmem != NULL.
 *
 * Returns NULL on failure; when err is non-NULL it describes the reason.
 */
kiss_rfft_cfg KISS_FFT_API kiss_rfft_alloc(int nfft, void *mem, size_t *lenmem,
                                           kiss_rfft_error *err);

#define kiss_rfft_free KISS_FFT_FREE

/*
 * Forward real FFT.
 *
 * in  : nfft real time samples
 * out : nfft complex bins, filled so that Hermitian symmetry holds exactly
 *       (the upper half is written as the conjugate mirror of the lower half)
 */
kiss_rfft_status KISS_FFT_API kiss_rfft(kiss_rfft_cfg cfg,
                                        const kiss_fft_scalar *in,
                                        kiss_fft_cpx *out,
                                        kiss_rfft_error *err);

/*
 * Inverse real FFT.
 *
 * in  : nfft complex bins of a spectrum that must be Hermitian symmetric;
 *       every violating index pair is recorded in err and
 *       KISS_RFFT_ESYMMETRY is returned, without touching out
 * out : nfft real time samples, scaled by 1/nfft
 *
 * tol gives the accepted symmetry residual per component:
 *   |a - b| <= tol + tol_rel * max(|a|, |b|)
 * Pass a negative value for a build-dependent default.
 */
kiss_rfft_status KISS_FFT_API kiss_irfft(kiss_rfft_cfg cfg,
                                         const kiss_fft_cpx *in,
                                         kiss_fft_scalar *out,
                                         double tol, double tol_rel,
                                         kiss_rfft_error *err);

/*
 * Validate the Hermitian symmetry of an nfft-point spectrum independently of
 * any transform. Returns KISS_RFFT_OK when every pair satisfies
 *
 *     |in[nfft-k] - conj(in[k])| <= tol + tol_rel * component magnitudes
 *
 * and KISS_RFFT_ESYMMETRY otherwise, with the offending index pairs recorded
 * in err. Pass tol < 0 for the build-dependent default tolerance.
 */
kiss_rfft_status KISS_FFT_API kiss_rfft_check_hermitian(int nfft,
                                                        const kiss_fft_cpx *in,
                                                        double tol, double tol_rel,
                                                        kiss_rfft_error *err);

#ifdef __cplusplus
}
#endif
#endif /* KISS_RFFT_H */
