/*
 *  Copyright (c) 2026, The kissfft authors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Canonical real FFT layer: full-spectrum (Hermitian) interface on top of the
 * mixed-radix (2,3,5) complex FFT.
 *
 * Two execution paths:
 *
 *  - nfft divisible by 4: the classic two-reals-in-one-complex split. The
 *    even/odd samples of the length-nfft real signal are packed as the real
 *    and imaginary parts of an m = nfft/2-point complex sequence (m even),
 *    transformed once, then separated with twiddle post-processing.
 *
 *  - every other size (odd nfft, or nfft == 2 mod 4 where m would be odd and
 *    the split packing does not work): an ordinary nfft-point complex FFT of
 *    the real sequence with a zero imaginary part.
 *
 * Conventions: the forward transform is unnormalized; the inverse divides by
 * nfft, so rfft followed by irfft reproduces the original samples.
 */

#include "kiss_rfft.h"
#include "_kiss_fft_guts.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

#if defined(FIXED_POINT) || defined(USE_SIMD)
#error "kiss_rfft only supports floating-point builds (kiss_fft_scalar float or double)"
#endif

/* machine epsilon of the active scalar type (float or double) */
#define KISS_RFFT_EPSILON ((sizeof(kiss_fft_scalar) == sizeof(float)) \
                               ? (double)FLT_EPSILON : (double)DBL_EPSILON)

struct kiss_rfft_state {
    int nfft;
    int use_split;     /* 1 when nfft % 4 == 0, 0 for the complex fallback */
    int m;             /* sub-FFT size: nfft/2 for split, nfft otherwise */
    kiss_fft_cfg sub_fwd;
    kiss_fft_cfg sub_inv;
    kiss_fft_cpx *tmpbuf;
    kiss_fft_cpx *twiddles; /* m/2 entries, split only; NULL otherwise */
};

/* ------------------------------------------------------------------ */
/* error reporting helpers                                            */
/* ------------------------------------------------------------------ */

static void err_reset(kiss_rfft_error *err, kiss_rfft_status code, const char *msg)
{
    if (err == NULL)
        return;
    err->code = code;
    err->n_violations = 0;
    if (msg != NULL) {
        snprintf(err->message, sizeof(err->message), "%s", msg);
        err->message[sizeof(err->message) - 1] = '\0';
    } else {
        err->message[0] = '\0';
    }
}

static void err_violation(kiss_rfft_error *err, int k, int nk, double residual)
{
    if (err == NULL)
        return;
    if (err->n_violations == 0)
        err->code = KISS_RFFT_ESYMMETRY;
    if (err->n_violations < KISS_RFFT_MAX_VIOLATIONS) {
        int slot = err->n_violations;
        err->index[slot] = k;
        err->mirror[slot] = nk;
        err->abs_err[slot] = residual;
        ++err->n_violations;
    }
}

/* nfft has no prime factor other than 2, 3 and 5 */
static int is_smooth(int nfft)
{
    int n = nfft;
    if (n <= 0)
        return 0;
    while (n % 2 == 0)
        n /= 2;
    while (n % 3 == 0)
        n /= 3;
    while (n % 5 == 0)
        n /= 5;
    return n == 1;
}

/* ------------------------------------------------------------------ */
/* allocation                                                         */
/* ------------------------------------------------------------------ */

kiss_rfft_cfg kiss_rfft_alloc(int nfft, void *mem, size_t *lenmem,
                              kiss_rfft_error *err)
{
    KISS_FFT_ALIGN_CHECK(mem)

    kiss_rfft_cfg st = NULL;
    size_t sub_fwd_size = 0, sub_inv_size = 0, memneeded;
    size_t off_cfg, off_inv, off_tmp, off_tw = 0;
    int m, use_split;

    if (!is_smooth(nfft)) {
        err_reset(err, KISS_RFFT_ESIZE,
                  "kiss_rfft: nfft must be a positive 2,3,5-smooth number");
        if (err != NULL)
            snprintf(err->message, sizeof(err->message),
                     "kiss_rfft: nfft=%d is not factorable by 2, 3 and 5 only",
                     nfft);
        if (lenmem != NULL)
            *lenmem = 0;
        return NULL;
    }

    use_split = (nfft % 4) == 0;
    m = use_split ? nfft / 2 : nfft;

    (void)kiss_fft_alloc(m, 0, NULL, &sub_fwd_size);
    (void)kiss_fft_alloc(m, 1, NULL, &sub_inv_size);
    if (sub_fwd_size == 0 || sub_inv_size == 0) {
        err_reset(err, KISS_RFFT_ESIZE,
                  "kiss_rfft: sub-transform size rejected by kiss_fft_alloc");
        if (lenmem != NULL)
            *lenmem = 0;
        return NULL;
    }

    /* state + two sub configs + m-point tmpbuf (+ split-only twiddles) */
    memneeded = KISS_FFT_PACK_SIZE_UP(sizeof(struct kiss_rfft_state));
    off_cfg = memneeded;
    memneeded += KISS_FFT_PACK_SIZE_UP(sub_fwd_size);
    off_inv = memneeded;
    memneeded += KISS_FFT_PACK_SIZE_UP(sub_inv_size);
    off_tmp = memneeded;
    memneeded += KISS_FFT_PACK_SIZE_UP(sizeof(kiss_fft_cpx) * (size_t)m);
    if (use_split) {
        off_tw = memneeded;
        memneeded += KISS_FFT_PACK_SIZE_UP(sizeof(kiss_fft_cpx) *
                                           (size_t)(m / 2));
    }

    if (lenmem == NULL) {
        st = (kiss_rfft_cfg)KISS_FFT_MALLOC(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_rfft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st == NULL) {
        err_reset(err, KISS_RFFT_EALLOC, "kiss_rfft: allocation failed");
        return NULL;
    }

    st->nfft = nfft;
    st->use_split = use_split;
    st->m = m;
    st->sub_fwd = (kiss_fft_cfg)((char *)st + off_cfg);
    st->sub_inv = (kiss_fft_cfg)((char *)st + off_inv);
    st->tmpbuf = (kiss_fft_cpx *)((char *)st + off_tmp);
    st->twiddles = use_split ? (kiss_fft_cpx *)((char *)st + off_tw) : NULL;

    if (kiss_fft_alloc(m, 0, st->sub_fwd, &sub_fwd_size) == NULL ||
        kiss_fft_alloc(m, 1, st->sub_inv, &sub_inv_size) == NULL) {
        err_reset(err, KISS_RFFT_EALLOC,
                  "kiss_rfft: sub-transform allocation failed");
        if (lenmem == NULL)
            KISS_FFT_FREE(st);
        return NULL;
    }

    if (use_split) {
        /* same phases as kiss_fftr's super_twiddles: W_m^(k-1/2) */
        int k;
        for (k = 0; k < m / 2; ++k) {
            double phase =
                -3.14159265358979323846264338327 * ((double)(k + 1) / m + .5);
            kf_cexp(st->twiddles + k, phase);
        }
    }

    err_reset(err, KISS_RFFT_OK, "ok");
    return st;
}

/* ------------------------------------------------------------------ */
/* forward transform                                                  */
/* ------------------------------------------------------------------ */

kiss_rfft_status kiss_rfft(kiss_rfft_cfg st, const kiss_fft_scalar *in,
                           kiss_fft_cpx *out, kiss_rfft_error *err)
{
    int k;

    if (st == NULL) {
        err_reset(err, KISS_RFFT_EALLOC, "kiss_rfft: NULL configuration");
        return KISS_RFFT_EALLOC;
    }
    if (in == NULL || out == NULL) {
        err_reset(err, KISS_RFFT_EARG, "kiss_rfft: NULL input or output buffer");
        return KISS_RFFT_EARG;
    }

    if (st->use_split) {
        /* even nfft with even m: pack two real sequences into one complex FFT */
        int m = st->m;
        kiss_fft_cpx tdc, fpk, fpnk, f1k, f2k, tw;

        kiss_fft(st->sub_fwd, (const kiss_fft_cpx *)in, st->tmpbuf);

        tdc = st->tmpbuf[0];
        out[0].r = tdc.r + tdc.i;
        out[0].i = (kiss_fft_scalar)0;
        out[m].r = tdc.r - tdc.i; /* Nyquist bin */
        out[m].i = (kiss_fft_scalar)0;

        for (k = 1; k <= m / 2; ++k) {
            fpk = st->tmpbuf[k];
            fpnk.r = st->tmpbuf[m - k].r;
            fpnk.i = -st->tmpbuf[m - k].i;

            C_ADD(f1k, fpk, fpnk);
            C_SUB(f2k, fpk, fpnk);
            C_MUL(tw, f2k, st->twiddles[k - 1]);

            out[k].r = HALF_OF(f1k.r + tw.r);
            out[k].i = HALF_OF(f1k.i + tw.i);
            out[m - k].r = HALF_OF(f1k.r - tw.r);
            out[m - k].i = HALF_OF(tw.i - f1k.i);
        }
        /* fill the exact Hermitian mirror of the upper half */
        for (k = 1; k < m; ++k) {
            out[m + k].r = out[m - k].r;
            out[m + k].i = -out[m - k].i;
        }
    } else {
        /* complex FFT of the real sequence with a zero imaginary part */
        int n = st->nfft;
        for (k = 0; k < n; ++k) {
            st->tmpbuf[k].r = in[k];
            st->tmpbuf[k].i = (kiss_fft_scalar)0;
        }
        kiss_fft(st->sub_fwd, st->tmpbuf, out);
        /* force the mirror exactly so the output is strictly Hermitian */
        for (k = 1; k < n; ++k) {
            out[n - k].r = out[k].r;
            out[n - k].i = -out[k].i;
        }
        out[0].i = (kiss_fft_scalar)0;
    }

    err_reset(err, KISS_RFFT_OK, "ok");
    return KISS_RFFT_OK;
}

/* ------------------------------------------------------------------ */
/* Hermitian symmetry validation                                      */
/* ------------------------------------------------------------------ */

kiss_rfft_status kiss_rfft_check_hermitian(int nfft, const kiss_fft_cpx *in,
                                           double tol, double tol_rel,
                                           kiss_rfft_error *err)
{
    int k, nk;
    double dr, di, lim_r, lim_i;

    if (in == NULL) {
        err_reset(err, KISS_RFFT_EARG,
                  "kiss_rfft_check_hermitian: NULL spectrum");
        return KISS_RFFT_EARG;
    }
    if (nfft <= 0) {
        err_reset(err, KISS_RFFT_ESIZE,
                  "kiss_rfft_check_hermitian: nfft must be positive");
        return KISS_RFFT_ESIZE;
    }

    if (tol < 0.0)
        tol = 64.0 * (double)KISS_RFFT_EPSILON;
    if (tol_rel < 0.0)
        tol_rel = 1024.0 * (double)KISS_RFFT_EPSILON;

    err_reset(err, KISS_RFFT_OK, "ok");

    /* NaN/Inf would make the comparisons below silently pass; reject first */
    for (k = 0; k < nfft; ++k) {
        if (!isfinite((double)in[k].r) || !isfinite((double)in[k].i)) {
            err_reset(err, KISS_RFFT_EVALUE,
                      "kiss_rfft: spectrum contains a non-finite value (NaN/Inf)");
            if (err != NULL)
                snprintf(err->message, sizeof(err->message),
                         "kiss_rfft: non-finite value at bin %d", k);
            return KISS_RFFT_EVALUE;
        }
    }

    /* DC is self-conjugate: its imaginary part must be zero */
    if (fabs((double)in[0].i) >
        tol + tol_rel * fabs((double)in[0].r)) {
        err_violation(err, 0, 0, fabs((double)in[0].i));
    }

    for (k = 1; k < (nfft + 1) / 2; ++k) {
        nk = nfft - k;
        /* require in[nk] == conj(in[k]): real equal, imag negated */
        dr = fabs((double)in[nk].r - (double)in[k].r);
        di = fabs((double)in[nk].i + (double)in[k].i);
        lim_r = tol + tol_rel *
                       fmax(fabs((double)in[nk].r), fabs((double)in[k].r));
        lim_i = tol + tol_rel *
                       fmax(fabs((double)in[nk].i), fabs((double)in[k].i));
        if (dr > lim_r || di > lim_i)
            err_violation(err, k, nk, fmax(dr, di));
    }

    /* even nfft: Nyquist is self-conjugate */
    if ((nfft & 1) == 0) {
        if (fabs((double)in[nfft / 2].i) >
            tol + tol_rel * fabs((double)in[nfft / 2].r)) {
            err_violation(err, nfft / 2, nfft / 2,
                          fabs((double)in[nfft / 2].i));
        }
    }

    if (err != NULL && err->n_violations > 0) {
        snprintf(err->message, sizeof(err->message),
                 "kiss_rfft: spectrum is not Hermitian symmetric; "
                 "%d violating index pair(s), first (%d,%d)",
                 err->n_violations, err->index[0], err->mirror[0]);
        return KISS_RFFT_ESYMMETRY;
    }
    return KISS_RFFT_OK;
}

/* ------------------------------------------------------------------ */
/* inverse transform                                                  */
/* ------------------------------------------------------------------ */

kiss_rfft_status kiss_irfft(kiss_rfft_cfg st, const kiss_fft_cpx *in,
                            kiss_fft_scalar *out, double tol, double tol_rel,
                            kiss_rfft_error *err)
{
    int k;
    kiss_rfft_status rc;

    if (st == NULL) {
        err_reset(err, KISS_RFFT_EALLOC, "kiss_irfft: NULL configuration");
        return KISS_RFFT_EALLOC;
    }
    if (in == NULL || out == NULL) {
        err_reset(err, KISS_RFFT_EARG, "kiss_irfft: NULL input or output buffer");
        return KISS_RFFT_EARG;
    }

    rc = kiss_rfft_check_hermitian(st->nfft, in, tol, tol_rel, err);
    if (rc != KISS_RFFT_OK)
        return rc;

    if (st->use_split) {
        /* rebuild the packed m-point spectrum and invert */
        int m = st->m;
        int nfft = st->nfft;
        /* The split forward stores half the true spectrum magnitude in the
         * packed bins; the m-point unnormalized inverse then scales by m.
         * The round-trip gain is therefore 2m = nfft, so dividing by nfft
         * reproduces the original samples. */
        kiss_fft_scalar scale = (kiss_fft_scalar)(1.0 / (double)nfft);
        kiss_fft_cpx fk, fnkc, fek, fok, tmp;

        st->tmpbuf[0].r = in[0].r + in[m].r;
        st->tmpbuf[0].i = in[0].r - in[m].r;

        /* Rebuild every packed bin. The forward split twiddles are stored with
         * negative phase; the inverse needs their conjugates, so w.i is
         * negated here. k and m-k are conjugate partners except at the
         * midpoint k = m/2, which keeps only the additive form. */
        for (k = 1; k <= m / 2; ++k) {
            kiss_fft_cpx w, addv, subv;

            fk = in[k];
            fnkc.r = in[m - k].r;
            fnkc.i = -in[m - k].i;

            w = st->twiddles[k - 1];
            w.i = -w.i;

            C_ADD(fek, fk, fnkc);
            C_SUB(tmp, fk, fnkc);
            C_MUL(fok, tmp, w);
            C_ADD(addv, fek, fok);
            C_SUB(subv, fek, fok);
            subv.i = -subv.i;

            st->tmpbuf[k] = addv;
            if (k != m - k)
                st->tmpbuf[m - k] = subv;
        }

        kiss_fft(st->sub_inv, st->tmpbuf, (kiss_fft_cpx *)out);
        for (k = 0; k < nfft; ++k)
            out[k] *= scale;
    } else {
        /* invert the nfft-point complex spectrum and take the real part */
        int n = st->nfft;
        kiss_fft_scalar scale = (kiss_fft_scalar)(1.0 / (double)n);

        kiss_fft(st->sub_inv, in, st->tmpbuf);
        for (k = 0; k < n; ++k)
            out[k] = st->tmpbuf[k].r * scale;
    }

    err_reset(err, KISS_RFFT_OK, "ok");
    return KISS_RFFT_OK;
}
