/*
 *  Copyright (c) 2026, Kiss FFT contributors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_hilbert.h"
#include "_kiss_fft_guts.h"

struct kiss_hilbert_state
{
    int nfft;       /* real frame length, in samples */
    int zero_pad;   /* zeros appended on each side */
    int ntrans;     /* FFT size actually used (fast size >= nfft + 2*pad) */
    kiss_fft_scalar *window; /* Hann window, nfft entries */
    kiss_fft_cfg cfg_fwd;
    kiss_fft_cfg cfg_inv;
    void *workspace; /* windowed input, two transform buffers, internal z */
};

static int scalar_compare(const void *a, const void *b)
{
    kiss_fft_scalar va = *(const kiss_fft_scalar *)a;
    kiss_fft_scalar vb = *(const kiss_fft_scalar *)b;
    return (va > vb) - (va < vb);
}

/* frequency-domain Hilbert filter, applied in place to a forward spectrum */
static void hilbert_filter(kiss_fft_cpx *spec, int ntrans)
{
    int k;

    /* DC stays as it is; for even lengths the Nyquist bin is also shared */
    for (k = 1; k < (ntrans + 1) / 2; ++k) {
        spec[k].r *= 2;
        spec[k].i *= 2;
    }
    /* negative-frequency bins carry the negative-frequency content away */
    for (k = ntrans / 2 + 1; k < ntrans; ++k) {
        spec[k].r = 0;
        spec[k].i = 0;
    }
    if (!(ntrans & 1))
        spec[ntrans / 2].i = 0; /* even length: real Nyquist bin, never doubled */
}

kiss_hilbert_cfg kiss_hilbert_alloc(int nfft, int zero_pad,
                                    void *mem, size_t *lenmem)
{
    KISS_FFT_ALIGN_CHECK(mem)

    kiss_hilbert_cfg st = NULL;
    size_t nfwd = 0, ninv = 0, nwork, memneeded;
    char *ptr = NULL;
    int ntrans, n;
    const kiss_fft_scalar two_pi = (kiss_fft_scalar)(2.0 * 3.14159265358979323846);

    if (nfft <= 1 || zero_pad < 0) {
        if (lenmem != NULL)
            *lenmem = 0;
        return NULL;
    }

    /* keep nfft + 2*pad in int range and leave room for the fast-size round up */
    if ((size_t)nfft > (size_t)INT_MAX - 2u * (size_t)zero_pad) {
        if (lenmem != NULL)
            *lenmem = (size_t)(-1);
        return NULL;
    }
    ntrans = kiss_fft_next_fast_size(nfft + 2 * zero_pad);

    (void)kiss_fft_alloc(ntrans, 0, NULL, &nfwd);
    (void)kiss_fft_alloc(ntrans, 1, NULL, &ninv);
    if (nfwd == 0 || ninv == 0)
        return NULL;

    /* window table, real input buffer, two complex transform buffers, an
     * analytic-signal buffer for callers of kiss_hilbert_track with z == NULL,
     * and a scratch array for the valid-margin percentile estimate */
    nwork = 2 * KISS_FFT_PACK_SIZE_UP((size_t)nfft * sizeof(kiss_fft_scalar))
          + KISS_FFT_PACK_SIZE_UP((size_t)ntrans * sizeof(kiss_fft_scalar))
          + 3 * KISS_FFT_PACK_SIZE_UP((size_t)ntrans * sizeof(kiss_fft_cpx));

    memneeded = KISS_FFT_PACK_SIZE_UP(sizeof(struct kiss_hilbert_state))
              + KISS_FFT_PACK_SIZE_UP(nfwd)
              + KISS_FFT_PACK_SIZE_UP(ninv)
              + KISS_FFT_PACK_SIZE_UP(nwork);

    if (lenmem == NULL) {
        ptr = (char *)KISS_FFT_MALLOC(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            ptr = (char *)mem;
        *lenmem = memneeded;
    }
    if (ptr == NULL)
        return NULL;

    st = (kiss_hilbert_cfg)ptr;
    memset(st, 0, memneeded);
    ptr += KISS_FFT_PACK_SIZE_UP(sizeof(struct kiss_hilbert_state));

    st->nfft = nfft;
    st->zero_pad = zero_pad;
    st->ntrans = ntrans;

    st->cfg_fwd = kiss_fft_alloc(ntrans, 0, ptr, &nfwd);
    ptr += KISS_FFT_PACK_SIZE_UP(nfwd);
    st->cfg_inv = kiss_fft_alloc(ntrans, 1, ptr, &ninv);
    ptr += KISS_FFT_PACK_SIZE_UP(ninv);
    st->workspace = ptr;

    st->window = (kiss_fft_scalar *)ptr;
    ptr += KISS_FFT_PACK_SIZE_UP((size_t)nfft * sizeof(kiss_fft_scalar));
    for (n = 0; n < nfft; ++n)
        st->window[n] = (kiss_fft_scalar)0.5 *
            ((kiss_fft_scalar)1 -
             (kiss_fft_scalar)cos(two_pi * (kiss_fft_scalar)n /
                                  (kiss_fft_scalar)(nfft - 1)));

    return st;
}

void kiss_hilbert_analytic(kiss_hilbert_cfg st,
                           const kiss_fft_scalar *x,
                           kiss_fft_cpx *z)
{
    const int nfft = st->nfft;
    const int pad = st->zero_pad;
    const int ntrans = st->ntrans;
    char *base = (char *)st->workspace;
    kiss_fft_scalar *win, *buf_real;
    kiss_fft_cpx *buf_fwd, *buf_inv;
    int n;

    /* skip the window table and the margin-estimator scratch, which share the
     * start of the workspace: window, scratch, buf_real, buf_fwd, buf_inv, z */
    base += 2 * KISS_FFT_PACK_SIZE_UP((size_t)nfft * sizeof(kiss_fft_scalar));
    buf_real = (kiss_fft_scalar *)base;
    base += KISS_FFT_PACK_SIZE_UP((size_t)ntrans * sizeof(kiss_fft_scalar));
    buf_fwd = (kiss_fft_cpx *)base;
    base += KISS_FFT_PACK_SIZE_UP((size_t)ntrans * sizeof(kiss_fft_cpx));
    buf_inv = (kiss_fft_cpx *)base;
    win = st->window;

    /* zero-padded, Hann-windowed frame: both edges are C1-continuous with
     * the surrounding zeros, so no hard-truncation leakage reaches the FFT */
    for (n = 0; n < pad; ++n)
        buf_real[n] = 0;
    for (n = 0; n < nfft; ++n)
        buf_real[pad + n] = x[n] * win[n];
    for (n = pad + nfft; n < ntrans; ++n)
        buf_real[n] = 0;

    for (n = 0; n < ntrans; ++n) {
        buf_fwd[n].r = buf_real[n];
        buf_fwd[n].i = 0;
    }

    kiss_fft(st->cfg_fwd, buf_fwd, buf_inv);
    hilbert_filter(buf_inv, ntrans);
    kiss_fft(st->cfg_inv, buf_inv, buf_fwd);

    /* inverse transform is unnormalized; scale back and keep the frame region */
    for (n = 0; n < nfft; ++n) {
        z[n].r = buf_fwd[n + pad].r / (kiss_fft_scalar)ntrans;
        z[n].i = buf_fwd[n + pad].i / (kiss_fft_scalar)ntrans;
    }
}

void kiss_hilbert_instfreq(kiss_hilbert_cfg st,
                           const kiss_fft_cpx *z,
                           kiss_fft_scalar *f_inst)
{
    const int nfft = st->nfft;
    const kiss_fft_scalar one_over_2pi =
        (kiss_fft_scalar)(1.0 / (2.0 * 3.14159265358979323846));
    int n;

    (void)st;

    /*
     * arg(z[n+1] * conj(z[n])) wrapped to (-pi, pi]: atan2 of the cross
     * product already gives the wrapped phase difference, no unwrap step.
     */
    for (n = 0; n < nfft - 1; ++n) {
        kiss_fft_scalar cr = z[n + 1].r * z[n].r + z[n + 1].i * z[n].i;
        kiss_fft_scalar ci = z[n + 1].i * z[n].r - z[n + 1].r * z[n].i;

        if (cr == 0 && ci == 0)
            f_inst[n] = 0;
        else
            f_inst[n] = (kiss_fft_scalar)atan2((double)ci, (double)cr)
                      * one_over_2pi;
    }
}

void kiss_hilbert_track(kiss_hilbert_cfg st,
                        const kiss_fft_scalar *x,
                        kiss_fft_cpx *z,
                        kiss_fft_scalar *f_inst,
                        int *valid_margin)
{
    char *base;
    kiss_fft_cpx *internal_z = NULL;

    if (z == NULL) {
        /* use the pre-allocated internal buffer: no hot-path allocation.
         * workspace layout: window, scratch, buf_real, buf_fwd, buf_inv, z */
        base = (char *)st->workspace;
        base += 2 * KISS_FFT_PACK_SIZE_UP((size_t)st->nfft * sizeof(kiss_fft_scalar));
        base += KISS_FFT_PACK_SIZE_UP((size_t)st->ntrans * sizeof(kiss_fft_scalar));
        base += 2 * KISS_FFT_PACK_SIZE_UP((size_t)st->ntrans * sizeof(kiss_fft_cpx));
        internal_z = (kiss_fft_cpx *)base;
        z = internal_z;
    }

    kiss_hilbert_analytic(st, x, z);
    kiss_hilbert_instfreq(st, z, f_inst);

    if (valid_margin != NULL)
        *valid_margin = kiss_hilbert_valid_margin(st, f_inst);
}

int kiss_hilbert_valid_margin(kiss_hilbert_cfg st,
                              const kiss_fft_scalar *f_inst)
{
    const int n = st->nfft - 1; /* number of IF entries */
    const int cap = st->nfft / 4;
    char *base = (char *)st->workspace;
    kiss_fft_scalar *scratch;
    kiss_fft_scalar f_lo, cycles, t;
    int k, margin, rank;

    /*
     * The edge guard scales with the longest signal period present. Sort the
     * instantaneous-frequency magnitudes and take the 5th percentile, which
     * tracks the lowest genuinely active frequency while ignoring the handful
     * of near-zero phase readings that the window taper produces at the edges.
     */
    base += KISS_FFT_PACK_SIZE_UP((size_t)st->nfft * sizeof(kiss_fft_scalar));
    scratch = (kiss_fft_scalar *)base;
    for (k = 0; k < n; ++k) {
        kiss_fft_scalar v = f_inst[k];
        scratch[k] = v < 0 ? -v : v;
    }
    qsort(scratch, (size_t)n, sizeof(kiss_fft_scalar), scalar_compare);
    rank = n / 20; /* 5th percentile */
    if (rank >= n)
        rank = n - 1;
    f_lo = scratch[rank];
    if (!(f_lo > (kiss_fft_scalar)1e-6))
        return cap; /* no measurable frequency: distrust the frame edges */

    /*
     * One period of guard suffices across most of the band; the region near
     * Nyquist needs more because the window side lobes sit closer in relative
     * terms. Grow from 1 to 4 cycles between 0.30 and 0.50 cycles/sample.
     */
    cycles = (kiss_fft_scalar)1.0;
    if (f_lo > (kiss_fft_scalar)0.30) {
        t = (f_lo - (kiss_fft_scalar)0.30) / (kiss_fft_scalar)0.20;
        if (t > 1)
            t = 1;
        cycles = (kiss_fft_scalar)1.0 + (kiss_fft_scalar)3.0 * t;
    }
    margin = (int)(cycles / f_lo + (kiss_fft_scalar)0.999999); /* ceil */
    if (margin < 4)
        margin = 4;
    if (margin > cap)
        margin = cap;
    return margin;
}
