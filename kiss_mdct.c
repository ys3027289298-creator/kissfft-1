/*
  Copyright (c) 2026, Mark Borgerding. All rights reserved.
  This file is part of KISS FFT - https://github.com/mborgerding/kissfft

  SPDX-License-Identifier: BSD-3-Clause
  See COPYING file for more information.
*/

#include "kiss_mdct.h"
#include "_kiss_fft_guts.h"
#include <stdint.h>

/*
  Both directions are expressed through one length-2N complex FFT.

  Forward (analysis), M = N/2, L = 2N:
    g[n] = x[n] * exp(j*pi*n/N),            n = 0 .. N-1
    g[n] = 0,                                n = N .. L-1
    G    = DFT_L(g)
    X[k] = Re( G[L-2k] * exp(j*c_k) )
    c_k  = (2*pi/N) * (k + 1/2) * (M/2 + 1/2)

  Inverse (synthesis), the exact transpose/adjoint chain. kiss_fft's inverse
  is unnormalized, so the (2/M) IMDCT normalization is applied explicitly:
    S[L-2k] = X[k] * exp(-j*c_k), all other bins zero
    s       = IDFT_L(S)
    y[n]   = (2/M) * Re( s[n] * exp(-j*pi*n/N) )

  These identities, and perfect reconstruction through a Princen-Bradley
  window, are verified in test/test_mdct.c.
*/

/*
  The MDCT/IMDCT normalization below assumes the floating-point kiss_fft
  convention (unnormalized transforms, fractional scalars). Fixed-point and
  SIMD builds use different internal scaling and are not wired up here; the
  public functions still compile but are documented as floating-point only.
*/

struct kiss_mdct_state {
    int n;          /* block length, even */
    int m;          /* number of MDCT coefficients, n/2 */
    int l;          /* FFT length, 2*n */
    kiss_fft_cfg fwd;
    kiss_fft_cfg inv;
    kiss_fft_cpx *pre;     /* n twiddles exp(j*pi*n/N)            */
    kiss_fft_cpx *post;    /* m twiddles exp(j*c_k)               */
    kiss_fft_cpx *tmp_in;  /* l complex scratch                   */
    kiss_fft_cpx *tmp_out; /* l complex scratch                   */
};

kiss_mdct_cfg kiss_mdct_alloc(int n, void *mem, size_t *lenmem)
{
    KISS_FFT_ALIGN_CHECK(mem)

    kiss_mdct_cfg st = NULL;
    size_t subsize = 0, subsize_inv = 0, memneeded;
    int k;

    if (n < 2 || (n & 1)) {
        KISS_FFT_ERROR("MDCT block length must be an even integer >= 2.");
        return NULL;
    }
    /* guard the 2*n FFT length and the scratch arithmetic against overflow */
    if (n >= (INT_MAX - 1) / 2 ||
        (size_t) n >= (SIZE_MAX / 2) / (5 * sizeof(kiss_fft_cpx))) {
        KISS_FFT_ERROR("MDCT block length too large.");
        return NULL;
    }

    /* probe the size of the two length-2N FFT configurations */
    kiss_fft_alloc(2 * n, 0, NULL, &subsize);
    kiss_fft_alloc(2 * n, 1, NULL, &subsize_inv);
    if (subsize < subsize_inv)
        subsize = subsize_inv;
    subsize_inv = subsize; /* keep both sub-configs the same size */

    memneeded = KISS_FFT_ALIGN_SIZE_UP(sizeof(struct kiss_mdct_state))
              + 2 * subsize
              + sizeof(kiss_fft_cpx) * (size_t)(n + n / 2 + 2 * (2 * n));

    if (lenmem == NULL) {
        st = (kiss_mdct_cfg) KISS_FFT_MALLOC(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_mdct_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->n = n;
    st->m = n / 2;
    st->l = 2 * n;

    st->fwd = (kiss_fft_cfg) (st + 1);
    st->inv = (kiss_fft_cfg) ((char *) st->fwd + subsize);
    st->pre = (kiss_fft_cpx *) ((char *) st->inv + subsize_inv);
    st->post = st->pre + n;
    st->tmp_in = st->post + n / 2;
    st->tmp_out = st->tmp_in + st->l;

    kiss_fft_alloc(2 * n, 0, st->fwd, &subsize);
    kiss_fft_alloc(2 * n, 1, st->inv, &subsize_inv);

    for (k = 0; k < n; ++k) {
        double phase = 3.141592653589793238462643383279502884 * k / n;
        kf_cexp(st->pre + k, phase);
    }
    for (k = 0; k < st->m; ++k) {
        double phase = 2.0 * 3.141592653589793238462643383279502884
                     * ((double) k + 0.5) * (st->m / 2.0 + 0.5) / n;
        kf_cexp(st->post + k, phase);
    }

    return st;
}

void kiss_mdct(kiss_mdct_cfg st, const kiss_fft_scalar *input,
               kiss_fft_scalar *output)
{
    int n, k;
    kiss_fft_cpx prod;

    if (st == NULL || input == NULL || output == NULL) {
        KISS_FFT_ERROR("kiss_mdct: NULL argument.");
        return;
    }

    for (n = 0; n < st->n; ++n) {
        st->tmp_in[n].r = input[n];
        st->tmp_in[n].i = 0;
    }
    for (n = st->n; n < st->l; ++n) {
        st->tmp_in[n].r = 0;
        st->tmp_in[n].i = 0;
    }

    /* pre-twiddle in place */
    for (n = 0; n < st->n; ++n) {
        prod.r = st->tmp_in[n].r;
        prod.i = st->tmp_in[n].i;
        C_MUL(st->tmp_in[n], prod, st->pre[n]);
    }

    kiss_fft(st->fwd, st->tmp_in, st->tmp_out);

    for (k = 0; k < st->m; ++k) {
        int bin = (2 * k == 0) ? 0 : st->l - 2 * k;
        C_MUL(prod, st->tmp_out[bin], st->post[k]);
        output[k] = prod.r;
    }
}

void kiss_imdct(kiss_mdct_cfg st, const kiss_fft_scalar *coeffs,
                kiss_fft_scalar *output)
{
    int n, k;
    kiss_fft_cpx term, conj_post, conj_pre, prod;

    if (st == NULL || coeffs == NULL || output == NULL) {
        KISS_FFT_ERROR("kiss_imdct: NULL argument.");
        return;
    }

    for (n = 0; n < st->l; ++n) {
        st->tmp_in[n].r = 0;
        st->tmp_in[n].i = 0;
    }

    for (k = 0; k < st->m; ++k) {
        int bin = (2 * k == 0) ? 0 : st->l - 2 * k;
        conj_post.r = st->post[k].r;
        conj_post.i = -st->post[k].i;
        term.r = coeffs[k];
        term.i = 0;
        C_MUL(st->tmp_in[bin], term, conj_post);
    }

    kiss_fft(st->inv, st->tmp_in, st->tmp_out);

    for (n = 0; n < st->n; ++n) {
        conj_pre.r = st->pre[n].r;
        conj_pre.i = -st->pre[n].i;
        C_MUL(prod, st->tmp_out[n], conj_pre);
        /* 2/M IMDCT normalization */
        C_MULBYSCALAR(prod, (kiss_fft_scalar) (2.0 / st->m));
        output[n] = prod.r;
    }
}

void kiss_mdct_sine_window(int n, kiss_fft_scalar *window)
{
    int i;
    if (n <= 0 || window == NULL) {
        KISS_FFT_ERROR("kiss_mdct_sine_window: invalid argument.");
        return;
    }
    for (i = 0; i < n; ++i)
        window[i] = (kiss_fft_scalar)
            sin(3.141592653589793238462643383279502884 * (i + 0.5) / n);
}
