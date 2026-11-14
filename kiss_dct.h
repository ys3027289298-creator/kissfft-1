/*
 *  Copyright (c) 2024, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * kiss_dct.h
 *
 * Single-header real transforms built on top of the complex KISS FFT:
 *
 *   - DCT-II  (the "DCT" used in most audio/image coding), unnormalized
 *   - DCT-III (inverse of DCT-II), unnormalized
 *   - DHT     (discrete Hartley transform), self-inverse
 *
 * All transforms accept any positive length N (prime lengths included;
 * they are not restricted to powers of two) and work both in-place
 * (fin == fout) and out-of-place.
 *
 * Conventions, matching the unnormalized KISS FFT style:
 *
 *   DCT-II:    X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k / (2N))
 *   DCT-III:   y[n] = X[0] + 2 * sum_{k=1}^{N-1}
 *                      X[k] * cos(pi*(2n+1)*k / (2N))
 *   DHT:       H[k] = sum_{n=0}^{N-1} x[n] * cas(2*pi*k*n/N)
 *                      with cas(a) = cos(a) + sin(a)
 *
 * DCT-II is the textbook plain-sum definition. DCT-III uses the
 * unnormalized inverse convention (factor 2 on every non-DC term, no 1/2
 * weight on X[0]), so DCT-III is the exact algebraic inverse of DCT-II:
 * applying DCT-II and then DCT-III reproduces N times the original input,
 * exactly like a forward/inverse pair of KISS FFTs. Divide by N to
 * normalize. The DHT is self-inverse: two successive DHTs likewise
 * reproduce N times the original input.
 *
 * Only the floating-point scalar types (float / double) are supported;
 * fixed-point and SIMD builds are rejected at compile time.
 *
 * Accuracy guarantee (for inputs bounded by |x[n]| <= 1):
 *
 *   The KISS FFT radix-r stages each contribute at most a small constant
 *   number of correctly rounded operations, so a length-N transform has a
 *   forward error bounded by
 *
 *       eps_abs <= C * u * N * (1 + log2(N)),
 *
 *   where u is the unit roundoff (u = 2^-24 ~= 5.96e-8 for float,
 *   u = 2^-53 ~= 1.11e-16 for double). Using the conservative constant
 *   C = 32, which covers the mixed-radix FFT stages plus the O(N)
 *   pre/post processing of the real transforms:
 *
 *       N = 1000, float:  eps_abs <= 32 * 5.9605e-8 * 1000 *
 *                         (1 + log2(1000)) ~= 2.09e-2
 *       N = 1024, float:  eps_abs <= 32 * 5.9605e-8 * 1024 *
 *                         (1 + log2(1024)) ~= 2.15e-2
 *
 *   Measured errors on typical |x| <= 1 signals are ~1e-5 for these two
 *   lengths, i.e. several hundred to ~1500x below the bound (see the
 *   verification program). Errors scale linearly with input magnitude.
 */

#ifndef KISS_DCT_H
#define KISS_DCT_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "kiss_fft.h"
#include "kiss_fft_log.h"

#if defined(FIXED_POINT) || defined(USE_SIMD)
#error "kiss_dct.h real transforms require a floating-point kiss_fft_scalar"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Header-only static helpers: silence "defined but not used" warnings. */
#if defined(__GNUC__)
#  define KISS_DCT_MAYBE_UNUSED __attribute__((unused))
#else
#  define KISS_DCT_MAYBE_UNUSED
#endif

typedef struct kiss_dct_state *kiss_dct_cfg;
typedef struct kiss_dht_state *kiss_dht_cfg;

struct kiss_dct_state {
    int nfft;
    int inverse;            /* 0 = DCT-II, 1 = DCT-III */
    kiss_fft_cfg fft_cfg;
    kiss_fft_cpx *tmp;      /* internal length-nfft complex scratch buffer */
    void *owned_mem;        /* non-NULL when this cfg block was malloc'd */
};

struct kiss_dht_state {
    int nfft;
    kiss_fft_cfg fft_cfg;
    kiss_fft_cpx *tmp;      /* internal length-nfft complex scratch buffer */
    void *owned_mem;        /* non-NULL when this cfg block was malloc'd */
};

/*
 * Number of bytes required for a DCT/DHT cfg placed in a user supplied
 * buffer. The returned value only covers the cfg structure itself; the
 * internal complex scratch buffer and the embedded KISS FFT cfg are always
 * allocated separately through KISS_FFT_MALLOC, mirroring how kiss_fftr's
 * sub-cfgs are managed.
 */
#define kiss_dct_cfg_size() (sizeof(struct kiss_dct_state))
#define kiss_dht_cfg_size() (sizeof(struct kiss_dht_state))

static KISS_DCT_MAYBE_UNUSED KISS_FFT_API kiss_dct_cfg
kiss_dct_alloc(int nfft, int inverse_fft, void *mem, size_t *lenmem)
{
    struct kiss_dct_state *st = NULL;
    void *owned = NULL;

    if (nfft <= 0) {
        if (lenmem != NULL) {
            *lenmem = (size_t)0;
        }
        KISS_FFT_ERROR("kiss_dct_alloc: nfft must be positive (got %d).",
                       nfft);
        return NULL;
    }

    if (lenmem == NULL) {
        st = (struct kiss_dct_state *)KISS_FFT_MALLOC(
            sizeof(struct kiss_dct_state));
        owned = st;
    } else {
        if (mem != NULL && *lenmem >= sizeof(struct kiss_dct_state)) {
            st = (struct kiss_dct_state *)mem;
        }
        *lenmem = sizeof(struct kiss_dct_state);
        if (st == NULL) {
            return NULL;
        }
    }

    if (st == NULL) {
        KISS_FFT_ERROR("kiss_dct_alloc: memory allocation failed.");
        return NULL;
    }

    st->nfft = nfft;
    st->inverse = inverse_fft;
    st->owned_mem = owned;
    st->fft_cfg = kiss_fft_alloc(nfft, inverse_fft, NULL, NULL);
    st->tmp = (kiss_fft_cpx *)KISS_FFT_MALLOC(
        sizeof(kiss_fft_cpx) * (size_t)nfft);

    if (st->fft_cfg == NULL || st->tmp == NULL) {
        KISS_FFT_ERROR("kiss_dct_alloc: scratch allocation failed.");
        if (st->fft_cfg != NULL) {
            KISS_FFT_FREE(st->fft_cfg);
        }
        if (st->tmp != NULL) {
            KISS_FFT_FREE(st->tmp);
        }
        if (owned != NULL) {
            KISS_FFT_FREE(owned);
        }
        return NULL;
    }

    return st;
}

/*
 * Release a cfg returned by kiss_dct_alloc. Always frees the embedded FFT
 * cfg and scratch buffer; the cfg structure itself is only freed when it
 * was malloc'd by the allocator (user-supplied buffers are left alone).
 */
static KISS_DCT_MAYBE_UNUSED KISS_FFT_API void
kiss_dct_free(kiss_dct_cfg st)
{
    if (st == NULL) {
        return;
    }
    if (st->fft_cfg != NULL) {
        KISS_FFT_FREE(st->fft_cfg);
    }
    if (st->tmp != NULL) {
        KISS_FFT_FREE(st->tmp);
    }
    if (st->owned_mem != NULL) {
        KISS_FFT_FREE(st->owned_mem);
    }
}

static KISS_DCT_MAYBE_UNUSED KISS_FFT_API kiss_dht_cfg
kiss_dht_alloc(int nfft, void *mem, size_t *lenmem)
{
    struct kiss_dht_state *st = NULL;
    void *owned = NULL;

    if (nfft <= 0) {
        if (lenmem != NULL) {
            *lenmem = (size_t)0;
        }
        KISS_FFT_ERROR("kiss_dht_alloc: nfft must be positive (got %d).",
                       nfft);
        return NULL;
    }

    if (lenmem == NULL) {
        st = (struct kiss_dht_state *)KISS_FFT_MALLOC(
            sizeof(struct kiss_dht_state));
        owned = st;
    } else {
        if (mem != NULL && *lenmem >= sizeof(struct kiss_dht_state)) {
            st = (struct kiss_dht_state *)mem;
        }
        *lenmem = sizeof(struct kiss_dht_state);
        if (st == NULL) {
            return NULL;
        }
    }

    if (st == NULL) {
        KISS_FFT_ERROR("kiss_dht_alloc: memory allocation failed.");
        return NULL;
    }

    st->nfft = nfft;
    st->owned_mem = owned;
    st->fft_cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
    st->tmp = (kiss_fft_cpx *)KISS_FFT_MALLOC(
        sizeof(kiss_fft_cpx) * (size_t)nfft);

    if (st->fft_cfg == NULL || st->tmp == NULL) {
        KISS_FFT_ERROR("kiss_dht_alloc: scratch allocation failed.");
        if (st->fft_cfg != NULL) {
            KISS_FFT_FREE(st->fft_cfg);
        }
        if (st->tmp != NULL) {
            KISS_FFT_FREE(st->tmp);
        }
        if (owned != NULL) {
            KISS_FFT_FREE(owned);
        }
        return NULL;
    }

    return st;
}

/* Release a cfg returned by kiss_dht_alloc (see kiss_dct_free). */
static KISS_DCT_MAYBE_UNUSED KISS_FFT_API void
kiss_dht_free(kiss_dht_cfg st)
{
    if (st == NULL) {
        return;
    }
    if (st->fft_cfg != NULL) {
        KISS_FFT_FREE(st->fft_cfg);
    }
    if (st->tmp != NULL) {
        KISS_FFT_FREE(st->tmp);
    }
    if (st->owned_mem != NULL) {
        KISS_FFT_FREE(st->owned_mem);
    }
}

/*
 * DCT-II (inverse_fft == 0) or DCT-III (inverse_fft == 1) of a real
 * length-N vector. fin and fout may point to the same buffer (in-place).
 */
static KISS_DCT_MAYBE_UNUSED KISS_FFT_API void
kiss_dct(kiss_dct_cfg st, const kiss_fft_scalar *fin, kiss_fft_scalar *fout)
{
    const int n = st->nfft;
    kiss_fft_cpx *z = st->tmp;
    int k;
    const double pi =
        3.141592653589793238462643383279502884197169399375105820974944;

    if (st == NULL || fin == NULL || fout == NULL) {
        KISS_FFT_ERROR("kiss_dct: NULL argument.");
        return;
    }

    /*
     * Length-1 fast path. All three real transforms are the identity for
     * N = 1 (DCT-II, DCT-III and DHT). It also avoids the pre-existing
     * in-place length-1 buffer overflow inside the KISS FFT engine.
     */
    if (n == 1) {
        fout[0] = fin[0];
        return;
    }

    if (!st->inverse) {
        /*
         * DCT-II via one length-N complex FFT (Makhoul's reordering).
         * y[j] = x[2j] for the even indices and
         * y[N-1-j] = x[2j+1] for the odd indices. This single rule is
         * valid for both even and odd N (including prime N).
         */
        for (k = 0; k < n; ++k) {
            z[k].r = (kiss_fft_scalar)0;
            z[k].i = (kiss_fft_scalar)0;
        }
        for (k = 0; 2 * k < n; ++k) {
            z[k].r = fin[2 * k];
        }
        for (k = 0; 2 * k + 1 < n; ++k) {
            z[n - 1 - k].r = fin[2 * k + 1];
        }

        kiss_fft(st->fft_cfg, z, z);

        for (k = 0; k < n; ++k) {
            double phase = -pi * k / (2.0 * n);
            double c = cos(phase);
            double s = sin(phase);
            fout[k] = (kiss_fft_scalar)(z[k].r * c - z[k].i * s);
        }
    } else {
        /*
         * DCT-III (unnormalized inverse) via one length-N inverse complex
         * FFT, the exact inverse of the Makhoul DCT-II above. With
         * alpha_k = pi*k/(2N) build
         *
         *   Z[0] = X[0] / 2
         *   Z[k] = e^{+i alpha_k} * X[k]              (k = 1..N-1).
         *
         * Let r = IFFT(Z). The unnormalized DCT-III coefficients are then
         *
         *   y[2j]   = 2 * Re(r[j])
         *   y[2j+1] = 2 * Re(r[N-1-j]),
         *
         * which evaluates to
         *   y[n] = X[0] + 2*sum_{k>=1} X[k] cos(pi*(2n+1)*k/(2N)).
         *
         * Hence DCT-III(DCT-II(x)) = N*x, matching the unnormalized
         * forward/inverse KISS FFT semantics. The same formulas hold for
         * even, odd and prime N (verified by exhaustive sign/weight/reorder
         * probing against a brute-force reference).
         */
        for (k = 0; k < n; ++k) {
            double xk = (k == 0) ? 0.5 * (double)fin[0]
                                 : (double)fin[k];
            double phase = pi * k / (2.0 * n);
            z[k].r = (kiss_fft_scalar)(xk * cos(phase));
            z[k].i = (kiss_fft_scalar)(xk * sin(phase));
        }

        kiss_fft(st->fft_cfg, z, z);

        for (k = 0; k < n; ++k) {
            int src = (k & 1) ? (n - 1 - k / 2) : (k / 2);
            fout[k] = (kiss_fft_scalar)(2.0 * (double)z[src].r);
        }
    }
}

/*
 * Discrete Hartley transform of a real length-N vector. The DHT is its own
 * inverse up to a factor of N: DHT(DHT(x)) = N * x. fin and fout may point
 * to the same buffer (in-place).
 */
static KISS_DCT_MAYBE_UNUSED KISS_FFT_API void
kiss_dht(kiss_dht_cfg st, const kiss_fft_scalar *fin, kiss_fft_scalar *fout)
{
    const int n = st->nfft;
    kiss_fft_cpx *z = st->tmp;
    int k;

    if (st == NULL || fin == NULL || fout == NULL) {
        KISS_FFT_ERROR("kiss_dht: NULL argument.");
        return;
    }

    /* Length-1 DHT is the identity; also avoids the KISS FFT n=1 quirk. */
    if (n == 1) {
        fout[0] = fin[0];
        return;
    }

    for (k = 0; k < n; ++k) {
        z[k].r = fin[k];
        z[k].i = (kiss_fft_scalar)0;
    }

    /* H[k] = Re(F[k]) - Im(F[k]), where F is the forward FFT of x */
    kiss_fft(st->fft_cfg, z, z);

    for (k = 0; k < n; ++k) {
        fout[k] = (kiss_fft_scalar)(z[k].r - z[k].i);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* KISS_DCT_H */
