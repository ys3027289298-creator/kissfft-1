/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 *  kiss_fft_f16.h  --  binary16 (IEEE-754 half precision) input/output layer
 *                      for kissfft.
 *
 *  This is an additive codec layer on top of the existing float32 engine.
 *  Nothing in kiss_fft.h, kiss_fft.c or the other engine sources changes; the
 *  layer stores complex samples as raw binary16 bit patterns
 *  (kiss_f16_cpx_t, 4 bytes per complex sample), widens them to float32,
 *  calls the unmodified kiss_fft_stride() engine out of place, and narrows the
 *  spectrum back to binary16.
 *
 *  The conversion is pure integer bit manipulation: no <fp16.h>, no _Float16,
 *  no F16C and no other dependency.  Every binary16 pattern, including
 *  subnormals, signed zero, infinities and NaNs, is handled per IEEE-754
 *  round-to-nearest-even.
 *
 *  Sample usage:
 *      kiss_fft_f16_cfg cfg = kiss_fft_f16_alloc(1024, 0);
 *      kiss_f16_cpx_t in[1024], out[1024];
 *      ... fill in[] with kiss_fft_f32_to_f16() ...
 *      int err = kiss_fft_f16(cfg, in, out);
 *      kiss_fft_f16_free(cfg);
 */

#ifndef KISS_FFT_F16_H
#define KISS_FFT_F16_H

#include <stdint.h>
#include <stddef.h>
#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One complex binary16 sample: r/i are IEEE-754 binary16 bit patterns
 * (the value a uint16_t holds, not a C arithmetic type).  4 bytes total. */
typedef struct {
    uint16_t r;
    uint16_t i;
} kiss_f16_cpx_t;

/* Opaque configuration object (wrapper state + float32 engine + scratch). */
typedef struct kiss_fft_f16_state *kiss_fft_f16_cfg;

/* Error codes returned by the transform entry points. */
#define KISS_FFT_F16_OK      0
#define KISS_FFT_F16_EINVAL  1 /* bad cfg/pointer/stride argument */
#define KISS_FFT_F16_ERANGE  2 /* at least one output component overflowed
                                  binary16 and was clamped to +/-inf */

/*
 * Largest nfft accepted.  Keeps every size_t multiplication performed by
 * kiss_fft_f16_mem_bytes()/kiss_fft_f16_alloc() well below 2^32, so an
 * unreasonable length is rejected instead of silently wrapping.
 * (2^28-1 == 268435455)
 */
#define KISS_FFT_F16_MAX_NFFT ( (int)( (1u<<28) - 1u ) )

/*
 * Convert one binary16 bit pattern to float32.  Exact for every input.
 * A signaling NaN is widened to a quiet NaN; the payload is preserved in the
 * top bits of the float32 mantissa.
 */
float KISS_FFT_API kiss_fft_f16_to_f32(uint16_t h);

/*
 * Convert one float32 to a binary16 bit pattern, IEEE-754 round-to-nearest,
 * ties-to-even, covering subnormals.
 *
 * If overflow is not NULL it receives 1 when the result is +/-infinity
 * produced by rounding a finite input beyond the binary16 range (the
 * threshold is 65520 = 65504 + half a half-ULP), and 0 otherwise.
 * Infinities and NaNs in the input are encoded as infinities and NaNs and
 * never set the overflow flag.  A finite output near the top of the range
 * that rounds up into +infinity (the 65504 -> 65536 carry) does set it, since
 * the represented value is no longer finite.
 */
uint16_t KISS_FFT_API kiss_fft_f32_to_f16(float f, int *overflow);

/*
 * Bytes that kiss_fft_f16_alloc() obtains for a given length, including the
 * wrapper state, the float32 engine state and both float32 scratch buffers.
 * Returns 0 for nfft outside [1, KISS_FFT_F16_MAX_NFFT].
 * Closed form on a 64-bit platform: 24*nfft + 296.
 */
size_t KISS_FFT_API kiss_fft_f16_mem_bytes(int nfft);

/*
 * Allocate a half-precision transform of nfft points (forward if
 * inverse_fft == 0, inverse otherwise).
 * Returns NULL for nfft outside [1, KISS_FFT_F16_MAX_NFFT] or on allocation
 * failure; no large allocation is attempted for a rejected length.
 * The returned object is freed with kiss_fft_f16_free() (plain free()).
 */
kiss_fft_f16_cfg KISS_FFT_API kiss_fft_f16_alloc(int nfft, int inverse_fft);

/* Release a config from kiss_fft_f16_alloc(); NULL is accepted.
 * The object is one malloc'd block, so plain free() is also sufficient. */
void KISS_FFT_API kiss_fft_f16_free(kiss_fft_f16_cfg cfg);

/*
 * Transform nfft binary16 complex samples from fin to fout.
 *
 * fin reads every fin_stride-th sample (1 = contiguous).  fout is always
 * written contiguously.  Entry points are out of place: fin may alias fout
 * only when fin_stride == 1 (the wrapper then uses its own scratch).
 *
 * Each output component is narrowed to binary16 after the float32 transform;
 * if any component does not fit, the component is stored as +/-infinity and
 * KISS_FFT_F16_ERANGE is returned (the rest of the spectrum is still valid).
 *
 * Return value: KISS_FFT_F16_OK, KISS_FFT_F16_EINVAL (NULL cfg, NULL fin or
 * fout, fin_stride <= 0, or a stride whose (nfft-1)*stride byte offset would
 * overflow size_t) or KISS_FFT_F16_ERANGE.
 * The transform allocates no heap memory.
 */
int KISS_FFT_API kiss_fft_f16_strided(kiss_fft_f16_cfg cfg, const kiss_f16_cpx_t *fin, int fin_stride, kiss_f16_cpx_t *fout);

/* As kiss_fft_f16_strided() with fin_stride == 1. */
int KISS_FFT_API kiss_fft_f16(kiss_fft_f16_cfg cfg, const kiss_f16_cpx_t *fin, kiss_f16_cpx_t *fout);

/*
 * Absolute error bound of the binary16 output relative to the exact DFT of
 * the widened input, with xmax = max_j |x[j]|:
 *
 *   |F16[k] - DFT(x)[k]| <= nfft * xmax * (gamma_s + 2^-11)
 *   s      = ceil(log2(nfft))
 *   gamma_s= 3*s*2^-24 / (1 - 3*s*2^-24)
 *
 * The first term is the 1-norm forward error of the float32 Cooley-Tukey
 * engine (classical 3s-stage FFT rounding bound, sum_j |x[j]| <= nfft*xmax).
 * The second term is the half half-ULP (2^-11 relative) added by narrowing to
 * binary16, applied to |X[k]| <= nfft*xmax.  Widening binary16 -> float32 is
 * exact.  The bound is conservative (measured error is typically 10-30x
 * smaller); it assumes the engine's stages do not overflow float32.
 *
 * Returns 0 for nfft outside [1, KISS_FFT_F16_MAX_NFFT] or xmax < 0,
 * +infinity if xmax is +infinity or NaN.
 */
double KISS_FFT_API kiss_fft_f16_error_bound(int nfft, double xmax);

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFT_F16_H */
