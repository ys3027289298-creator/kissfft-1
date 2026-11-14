/*
  Copyright (c) 2026, Mark Borgerding. All rights reserved.
  This file is part of KISS FFT - https://github.com/mborgerding/kissfft

  SPDX-License-Identifier: BSD-3-Clause
  See COPYING file for more information.
*/

#ifndef KISS_MDCT_H
#define KISS_MDCT_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
  MDCT / IMDCT built on top of the unmodified kiss_fft complex transform.

  Floating-point only (default float, or -Dkiss_fft_scalar=double). The
  fixed-point and SIMD kiss_fft configurations use different internal scaling
  and are not supported by this module.

  For a block of N real samples (N must be even, N >= 2), the forward MDCT
  produces N/2 real coefficients:

      X[k] = sum_{n=0}^{N-1} x[n] *
             cos( (2*pi/N) * (n + N/4 + 1/2) * (k + 1/2) ),  k = 0 .. N/2-1

  The IMDCT is the transpose of that analysis, normalized so that a 50%
  overlap-add of consecutive blocks with a Princen-Bradley window (for
  example the sine window) gives perfect reconstruction:

      y[n] = (2/(N/2)) * sum_k X[k] *
             cos( (2*pi/N) * (n + N/4 + 1/2) * (k + 1/2) )

  The implementation uses a single length-2N complex FFT per transform and
  works for every even N, including sizes with arbitrary (non power-of-two)
  prime factors such as N = 1000.
*/

typedef struct kiss_mdct_state *kiss_mdct_cfg;

/*
  Allocate an MDCT/IMDCT configuration for block length n.

  n must be even and >= 2; otherwise the function logs an error and returns
  NULL. The same configuration serves both directions: call kiss_mdct() for
  the analysis transform and kiss_imdct() for the synthesis transform.

  The mem/lenmem arguments follow the same convention as kiss_fft_alloc():
  when both are NULL a buffer is allocated internally and must be released
  with kiss_mdct_free(); otherwise the configuration is placed in the caller
  supplied buffer and the required size is written to *lenmem.
*/
kiss_mdct_cfg KISS_FFT_API kiss_mdct_alloc(int n, void *mem, size_t *lenmem);

/*
  Forward MDCT.

  input  : n real samples (kiss_fft_scalar).
  output : n/2 real coefficients, stored in the .r members of out[0..n/2-1]
           (the .i members are set to zero).
*/
void KISS_FFT_API kiss_mdct(kiss_mdct_cfg cfg, const kiss_fft_scalar *input,
                            kiss_fft_scalar *output);

/*
  Inverse MDCT.

  input  : n/2 real coefficients.
  output : n real samples. Overlap-add consecutive outputs (shifted by n/2)
           together with a matching Princen-Bradley window for perfect
           reconstruction.
*/
void KISS_FFT_API kiss_imdct(kiss_mdct_cfg cfg, const kiss_fft_scalar *coeffs,
                             kiss_fft_scalar *output);

/* Sine (a.k.a. MDCT analysis/synthesis) window of length n:
 * w[i] = sin(pi * (i + 1/2) / n). It satisfies w[i]^2 + w[i+n/2]^2 = 1. */
void KISS_FFT_API kiss_mdct_sine_window(int n, kiss_fft_scalar *window);

#define kiss_mdct_free KISS_FFT_FREE

#ifdef __cplusplus
}
#endif

#endif /* KISS_MDCT_H */
