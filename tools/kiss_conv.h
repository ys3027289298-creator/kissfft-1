/*
 *  FFT-based linear convolution and autocorrelation built on KISS FFT.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_CONV_H
#define KISS_CONV_H

#include "kiss_fft.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 Linear convolution and autocorrelation for complex sequences of arbitrary
 length.  Internally the inputs are zero-padded to
 kiss_fft_next_fast_size(nx+nh-1) and transformed with kiss_fft, so any
 length >= 1 works (not just powers of two).

 Round-off error bound (floating point builds):
     max_k |y_fft[k] - y_exact[k]|
         <= eps * (3*log2(nfft) + 5) * ||x||2 * ||h||2
 where eps is the machine epsilon of kiss_fft_scalar (FLT_EPSILON for
 float, DBL_EPSILON for double), nfft = kiss_fft_next_fast_size(nx+nh-1),
 and ||.||2 is the Euclidean norm of the (real or complex) input.
 Rationale: each of the three FFTs contributes a relative error of about
 eps*log2(nfft) and the pointwise product contributes one more eps; the
 constants above are conservative.

 Fixed-point (FIXED_POINT) builds are not supported; all functions return
 KISS_CONV_ERR_UNSUPPORTED there.
*/

typedef enum {
    KISS_CONV_OK = 0,
    KISS_CONV_ERR_NULL,        /* a NULL pointer was passed */
    KISS_CONV_ERR_LENGTH,      /* zero length, or required size overflows / exceeds INT_MAX */
    KISS_CONV_ERR_ALLOC,       /* memory allocation failed */
    KISS_CONV_ERR_UNSUPPORTED  /* fixed-point build */
} kiss_conv_status;

/* Output length of kiss_conv_linear: nx+nh-1, or 0 if it overflows. */
size_t kiss_conv_linear_len(size_t nx, size_t nh);

/* Output length of kiss_conv_autocorr: 2*nx-1, or 0 if it overflows. */
size_t kiss_conv_autocorr_len(size_t nx);

/*
 Linear convolution: y[k] = sum_i x[i] * h[k-i], k = 0 .. nx+nh-2.
 y must have room for kiss_conv_linear_len(nx, nh) samples.
*/
kiss_conv_status kiss_conv_linear(const kiss_fft_cpx *x, size_t nx,
                                  const kiss_fft_cpx *h, size_t nh,
                                  kiss_fft_cpx *y);

/*
 Autocorrelation: r[k + nx - 1] = sum_i x[i+k] * conj(x[i]),
 for lags k = -(nx-1) .. (nx-1).  Zero lag sits at r[nx-1].
 r must have room for kiss_conv_autocorr_len(nx) samples.
*/
kiss_conv_status kiss_conv_autocorr(const kiss_fft_cpx *x, size_t nx,
                                    kiss_fft_cpx *r);

#ifdef __cplusplus
}
#endif

#endif /* KISS_CONV_H */
