/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 *  binary16 codec + wrapper around the unmodified float32 kissfft engine.
 *
 *  Nothing here touches the engine sources.  The transform path is:
 *
 *      f16 wire samples --widen--> float32 scratch --kiss_fft_stride()-->
 *      float32 scratch --narrow--> f16 wire samples
 *
 *  Both scratch buffers are allocated once by kiss_fft_f16_alloc(), so a
 *  transform performs no heap allocation of its own for lengths whose prime
 *  factors are all <= 5.  For other lengths the unmodified engine allocates
 *  and frees a temporary of at most MAXFACTORS (32) complex samples per call,
 *  so the steady-state footprint does not grow either.
 */

#include <string.h>
#include <math.h>
#include <stddef.h>
#include <limits.h>
#include "kiss_fft_f16.h"

#if defined(FIXED_POINT) || defined(USE_SIMD)
# error "kiss_fft_f16 requires the plain floating-point scalar build"
#endif
/* compile-time assertion that the engine scalar is 32-bit float */
typedef char kiss_fft_f16_requires_float[(sizeof(kiss_fft_scalar) == sizeof(float)) ? 1 : -1];

/* ------------------------------------------------------------------------- */
/* binary16 <-> float32, pure integer bit manipulation                       */
/* ------------------------------------------------------------------------- */

/* Layout:
 *   binary16: 1 sign | 5 exponent (bias 15)  | 10 mantissa
 *   float32 : 1 sign | 8 exponent (bias 127) | 23 mantissa
 *
 * A normal half widens exactly: rebase the exponent (e32 = e16 - 15 + 127) and
 * shift the mantissa left by the 13-bit width difference.
 * A subnormal half (exponent 0, mantissa m != 0) represents m * 2^-24; it is
 * normalised by shifting m left until bit 10 is set, which costs `shift`
 * halvings: the float32 exponent is then (10 - shift) - 24 + 127 = 113 - shift.
 */

float kiss_fft_f16_to_f32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16; /* bit 31 */
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    float f;

    if (exp == 0x1Fu) {
        /* infinity or NaN: keep sign, all-ones exponent and payload.  Any
         * nonzero payload is forced quiet, so a signaling NaN is quieted. */
        bits = sign | 0x7F800000u | (mant << 13);
        if (mant != 0)
            bits |= 0x00400000u; /* quiet bit */
        memcpy(&f, &bits, sizeof f);
        return f;
    }

    if (exp == 0) {
        if (mant == 0) {
            /* +/- zero, sign preserved */
            bits = sign;
            memcpy(&f, &bits, sizeof f);
            return f;
        }
        {
            int shift = 0;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++shift;
            }
            bits = sign | (((uint32_t)(113 - shift)) << 23) | ((m & 0x3FFu) << 13);
            memcpy(&f, &bits, sizeof f);
            return f;
        }
    }

    bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    memcpy(&f, &bits, sizeof f);
    return f;
}

/*
 * Narrow float32 -> binary16, round-to-nearest-even.
 *
 * For a finite input the value is (2^23 + mant) * 2^(exp - 150).
 *  - exp in [113, 141]: a normal half; the 10-bit half mantissa is the top ten
 *    bits of the float mantissa, rounded on the low 13 bits.
 *  - exp == 142: the overflow boundary.  Values >= 65520 (mant >= 0x7FF000)
 *    round to infinity; lower values round to 65504 (0x7BFF).
 *  - exp >= 143: always infinity.
 *  - exp <= 112: a subnormal half (or zero); round the value scaled by 2^24.
 */
uint16_t kiss_fft_f32_to_f16(float f, int *overflow)
{
    uint32_t bits;
    uint32_t sign, absbits;
    int32_t exp;
    uint32_t mant;
    uint16_t sign16;

    memcpy(&bits, &f, sizeof bits);
    sign    = bits >> 31;
    absbits = bits & 0x7FFFFFFFu;
    exp     = (int32_t)(absbits >> 23);
    mant    = absbits & 0x7FFFFFu;
    sign16  = (uint16_t)(sign << 15);

    if (overflow != NULL)
        *overflow = 0;

    if (exp == 0xFF) {
        if (mant == 0)
            return (uint16_t)(sign16 | 0x7C00u); /* +/- infinity */
        /* NaN: keep the top 10 payload bits, force the quiet bit */
        return (uint16_t)(sign16 | 0x7E00u | (mant >> 13));
    }

    if (absbits == 0)
        return sign16; /* +/- zero */

    if (exp >= 143) {
        /* >= 65536: always infinity */
        if (overflow != NULL)
            *overflow = 1;
        return (uint16_t)(sign16 | 0x7C00u);
    }

    if (exp >= 113) {
        /* normal half, exponent rebased by 112 so e16 == exp-112 (e16 == 30
         * for exp == 142 is still a valid normal half). */
        uint32_t e16 = (uint32_t)exp - 112u; /* 1..30 */
        uint32_t low = mant & 0x1FFFu;
        uint32_t m10 = mant >> 13;

        if (low > 0x1000u || (low == 0x1000u && (m10 & 1u) != 0))
            ++m10;
        if (m10 == 0x400u) {
            m10 = 0;
            ++e16;
            if (e16 >= 31u) {
                /* rounded past 65504: e.g. 65520 (the exact halfway point
                 * between 65504 and 65536) ties to the even mantissa 0x400,
                 * i.e. to infinity. */
                if (overflow != NULL)
                    *overflow = 1;
                return (uint16_t)(sign16 | 0x7C00u);
            }
        }
        return (uint16_t)(sign16 | (e16 << 10) | m10);
    }

    /* exp <= 112: subnormal half or zero.
     * half mantissa = round( (2^23 + mant) * 2^(exp - 126) ). */
    {
        int shift = 126 - exp; /* >= 14 */
        uint32_t m, keep, rem, halfway;

        if (shift >= 32)
            return sign16; /* below half of the smallest subnormal -> 0 */

        m = 0x800000u | mant;
        keep = m >> shift;
        rem = m & (((uint32_t)1 << shift) - 1u);
        halfway = (uint32_t)1 << (shift - 1);

        if (rem > halfway || (rem == halfway && (keep & 1u) != 0))
            ++keep;

        /* keep == 0x400 is the smallest normal half (2^-14), correctly
         * encoded as exponent 1 with mantissa 0. */
        return (uint16_t)(sign16 | (uint16_t)keep);
    }
}

/* ------------------------------------------------------------------------- */
/* wrapper state                                                             */
/* ------------------------------------------------------------------------- */

struct kiss_fft_f16_state {
    int nfft;
    int inverse;
    kiss_fft_cfg engine;      /* unmodified float32 engine */
    kiss_fft_cpx *scratch_in; /* nfft float32 complex samples */
    kiss_fft_cpx *scratch_out;
};

size_t kiss_fft_f16_mem_bytes(int nfft)
{
    size_t engine_bytes = 0;
    size_t scratch_bytes;

    if (nfft < 1 || nfft > KISS_FFT_F16_MAX_NFFT)
        return 0;

    /* ask the engine for its exact footprint (this is a pure size query and
     * does not allocate or compute twiddles) */
    (void)kiss_fft_alloc(nfft, 0, NULL, &engine_bytes);
    scratch_bytes = 2u * sizeof(kiss_fft_cpx) * (size_t)nfft;

    /* the MAX_NFFT ceiling keeps every product far below SIZE_MAX */
    return sizeof(struct kiss_fft_f16_state) + engine_bytes + scratch_bytes;
}

kiss_fft_f16_cfg kiss_fft_f16_alloc(int nfft, int inverse_fft)
{
    kiss_fft_f16_cfg cfg;

    if (nfft < 1 || nfft > KISS_FFT_F16_MAX_NFFT)
        return NULL;

    cfg = (kiss_fft_f16_cfg)KISS_FFT_MALLOC(sizeof(struct kiss_fft_f16_state));
    if (cfg == NULL)
        return NULL;

    cfg->nfft = nfft;
    cfg->inverse = inverse_fft ? 1 : 0;
    cfg->engine = kiss_fft_alloc(nfft, cfg->inverse, NULL, NULL);
    cfg->scratch_in  = (kiss_fft_cpx *)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx) * (size_t)nfft);
    cfg->scratch_out = (kiss_fft_cpx *)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx) * (size_t)nfft);

    if (cfg->engine == NULL || cfg->scratch_in == NULL || cfg->scratch_out == NULL) {
        KISS_FFT_FREE(cfg->scratch_out);
        KISS_FFT_FREE(cfg->scratch_in);
        KISS_FFT_FREE(cfg->engine);
        KISS_FFT_FREE(cfg);
        return NULL;
    }
    return cfg;
}

void kiss_fft_f16_free(kiss_fft_f16_cfg cfg)
{
    if (cfg == NULL)
        return;
    KISS_FFT_FREE(cfg->scratch_out);
    KISS_FFT_FREE(cfg->scratch_in);
    KISS_FFT_FREE(cfg->engine);
    KISS_FFT_FREE(cfg);
}

/* ------------------------------------------------------------------------- */
/* transform                                                                 */
/* ------------------------------------------------------------------------- */

int kiss_fft_f16_strided(kiss_fft_f16_cfg cfg, const kiss_f16_cpx_t *fin,
                         int fin_stride, kiss_f16_cpx_t *fout)
{
    int i;
    int overflowed = 0;

    if (cfg == NULL || fin == NULL || fout == NULL || fin_stride <= 0)
        return KISS_FFT_F16_EINVAL;
    /* the last sample read is fin + (nfft-1)*fin_stride; reject a stride whose
     * byte offset cannot be represented in size_t */
    if ((size_t)(cfg->nfft - 1) > SIZE_MAX / (size_t)fin_stride / sizeof(kiss_f16_cpx_t))
        return KISS_FFT_F16_EINVAL;

    /* widen f16 -> f32.  The widened copy lives in scratch_in, so fin may
     * alias fout when fin_stride == 1. */
    for (i = 0; i < cfg->nfft; ++i) {
        const kiss_f16_cpx_t *in = fin + (size_t)i * (size_t)fin_stride;
        cfg->scratch_in[i].r = kiss_fft_f16_to_f32(in->r);
        cfg->scratch_in[i].i = kiss_fft_f16_to_f32(in->i);
    }

    /* unmodified float32 engine, always out of place */
    kiss_fft_stride(cfg->engine, cfg->scratch_in, cfg->scratch_out, 1);

    /* narrow f32 -> f16, accumulating the overflow flag over all components */
    for (i = 0; i < cfg->nfft; ++i) {
        int ov_r = 0, ov_i = 0;
        fout[i].r = kiss_fft_f32_to_f16(cfg->scratch_out[i].r, &ov_r);
        fout[i].i = kiss_fft_f32_to_f16(cfg->scratch_out[i].i, &ov_i);
        overflowed |= (ov_r | ov_i);
    }

    return overflowed ? KISS_FFT_F16_ERANGE : KISS_FFT_F16_OK;
}

int kiss_fft_f16(kiss_fft_f16_cfg cfg, const kiss_f16_cpx_t *fin, kiss_f16_cpx_t *fout)
{
    return kiss_fft_f16_strided(cfg, fin, 1, fout);
}

/* ------------------------------------------------------------------------- */
/* error bound                                                               */
/* ------------------------------------------------------------------------- */

double kiss_fft_f16_error_bound(int nfft, double xmax)
{
    int s;
    double u32 = 1.0 / 16777216.0; /* 2^-24 */
    double gamma;
    uint64_t xbits;

    if (nfft < 1 || nfft > KISS_FFT_F16_MAX_NFFT)
        return 0.0;

    /* Inspect the IEEE-754 encoding rather than comparing against INFINITY /
     * testing x != x: this keeps the classification correct even when the
     * caller's translation unit is compiled with -ffast-math. */
    memcpy(&xbits, &xmax, sizeof xbits);
    if ((xbits & 0x8000000000000000ull) != 0ull)
        return 0.0;                       /* negative (including -0) */
    if ((xbits & 0x7FF0000000000000ull) == 0x7FF0000000000000ull)
        return INFINITY;                  /* inf or NaN */

    s = 0;
    while ((1 << s) < nfft) /* ceil(log2(nfft)); nfft <= 2^28-1 */
        ++s;

    /* Forward error of the float32 Cooley-Tukey engine.  Each of the s stages
     * performs a bounded number of fl(+,*); the classical 1-norm FFT bound
     * (Higham, "Accuracy and Stability of Numerical Algorithms", ch. 23.3) is
     *   |F32[k] - DFT(x)[k]| <= 3*s*u32*sum_j |x[j]| / (1 - 3*s*u32),
     * and sum_j |x[j]| <= nfft*max_j |x[j]| = nfft*xmax.
     * The factor 3 covers the handful of rounding operations per butterfly.
     * Narrowing to binary16 adds at most half a half-ULP (2^-11) relative to
     * each output magnitude, and |X[k]| <= nfft*xmax.
     * Widening binary16 -> float32 is exact.
     */
    gamma = 3.0 * (double)s * u32;
    if (gamma >= 1.0)
        return INFINITY;
    gamma /= (1.0 - gamma);

    return (double)nfft * xmax * (gamma + 1.0 / 2048.0); /* + 2^-11 */
}
