/*
 *  FFT-based linear convolution and autocorrelation built on KISS FFT.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_conv.h"

#include <limits.h>
#include <string.h>

size_t kiss_conv_linear_len(size_t nx, size_t nh)
{
    if (nx == 0 || nh == 0 || nx > (size_t)-1 - nh)
        return 0;
    return nx + nh - 1;
}

size_t kiss_conv_autocorr_len(size_t nx)
{
    if (nx == 0 || nx - 1 > ((size_t)-1 - 1) / 2)
        return 0;
    return 2 * nx - 1;
}

#ifdef FIXED_POINT

kiss_conv_status kiss_conv_linear(const kiss_fft_cpx *x, size_t nx,
                                  const kiss_fft_cpx *h, size_t nh,
                                  kiss_fft_cpx *y)
{
    (void)x; (void)nx; (void)h; (void)nh; (void)y;
    return KISS_CONV_ERR_UNSUPPORTED;
}

kiss_conv_status kiss_conv_autocorr(const kiss_fft_cpx *x, size_t nx,
                                    kiss_fft_cpx *r)
{
    (void)x; (void)nx; (void)r;
    return KISS_CONV_ERR_UNSUPPORTED;
}

#else /* floating point */

typedef struct {
    kiss_fft_cfg fwd;
    kiss_fft_cfg inv;
    kiss_fft_cpx *fa;
    kiss_fft_cpx *fb;
    size_t nfft;
} kiss_conv_workspace;

static void workspace_free(kiss_conv_workspace *ws)
{
    if (ws->fwd)
        kiss_fft_free(ws->fwd);
    if (ws->inv)
        kiss_fft_free(ws->inv);
    if (ws->fa)
        KISS_FFT_FREE(ws->fa);
    if (ws->fb)
        KISS_FFT_FREE(ws->fb);
    memset(ws, 0, sizeof(*ws));
}

static kiss_conv_status workspace_alloc(kiss_conv_workspace *ws, size_t nconv)
{
    memset(ws, 0, sizeof(*ws));
    if (nconv == 0 || nconv > (size_t)INT_MAX)
        return KISS_CONV_ERR_LENGTH;
    ws->nfft = (size_t)kiss_fft_next_fast_size((int)nconv);
    ws->fwd = kiss_fft_alloc((int)ws->nfft, 0, NULL, NULL);
    ws->inv = kiss_fft_alloc((int)ws->nfft, 1, NULL, NULL);
    ws->fa = (kiss_fft_cpx *)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx) * ws->nfft);
    ws->fb = (kiss_fft_cpx *)KISS_FFT_MALLOC(sizeof(kiss_fft_cpx) * ws->nfft);
    if (!ws->fwd || !ws->inv || !ws->fa || !ws->fb) {
        workspace_free(ws);
        return KISS_CONV_ERR_ALLOC;
    }
    return KISS_CONV_OK;
}

kiss_conv_status kiss_conv_linear(const kiss_fft_cpx *x, size_t nx,
                                  const kiss_fft_cpx *h, size_t nh,
                                  kiss_fft_cpx *y)
{
    kiss_conv_workspace ws;
    kiss_conv_status st;
    size_t nconv, k;
    kiss_fft_scalar scale;

    if (!x || !h || !y)
        return KISS_CONV_ERR_NULL;
    nconv = kiss_conv_linear_len(nx, nh);
    st = workspace_alloc(&ws, nconv);
    if (st != KISS_CONV_OK)
        return st;

    memset(ws.fa, 0, sizeof(kiss_fft_cpx) * ws.nfft);
    memset(ws.fb, 0, sizeof(kiss_fft_cpx) * ws.nfft);
    memcpy(ws.fa, x, sizeof(kiss_fft_cpx) * nx);
    memcpy(ws.fb, h, sizeof(kiss_fft_cpx) * nh);

    kiss_fft(ws.fwd, ws.fa, ws.fa);
    kiss_fft(ws.fwd, ws.fb, ws.fb);
    for (k = 0; k < ws.nfft; ++k) {
        kiss_fft_cpx p;
        p.r = ws.fa[k].r * ws.fb[k].r - ws.fa[k].i * ws.fb[k].i;
        p.i = ws.fa[k].r * ws.fb[k].i + ws.fa[k].i * ws.fb[k].r;
        ws.fa[k] = p;
    }
    kiss_fft(ws.inv, ws.fa, ws.fa);

    scale = (kiss_fft_scalar)1 / (kiss_fft_scalar)ws.nfft;
    for (k = 0; k < nconv; ++k) {
        y[k].r = ws.fa[k].r * scale;
        y[k].i = ws.fa[k].i * scale;
    }

    workspace_free(&ws);
    return KISS_CONV_OK;
}

kiss_conv_status kiss_conv_autocorr(const kiss_fft_cpx *x, size_t nx,
                                    kiss_fft_cpx *r)
{
    kiss_conv_workspace ws;
    kiss_conv_status st;
    size_t nconv, k;
    kiss_fft_scalar scale;

    if (!x || !r)
        return KISS_CONV_ERR_NULL;
    nconv = kiss_conv_autocorr_len(nx);
    st = workspace_alloc(&ws, nconv);
    if (st != KISS_CONV_OK)
        return st;

    memset(ws.fa, 0, sizeof(kiss_fft_cpx) * ws.nfft);
    memcpy(ws.fa, x, sizeof(kiss_fft_cpx) * nx);

    kiss_fft(ws.fwd, ws.fa, ws.fa);
    for (k = 0; k < ws.nfft; ++k) {
        ws.fa[k].r = ws.fa[k].r * ws.fa[k].r + ws.fa[k].i * ws.fa[k].i;
        ws.fa[k].i = 0;
    }
    kiss_fft(ws.inv, ws.fa, ws.fa);

    /* ws.fa[k] now holds the linear autocorrelation at lag k for
       k = 0 .. nx-1 and at lag k - nfft for k = nfft-nx+1 .. nfft-1. */
    scale = (kiss_fft_scalar)1 / (kiss_fft_scalar)ws.nfft;
    for (k = 0; k < nx; ++k) {
        r[nx - 1 + k].r = ws.fa[k].r * scale;
        r[nx - 1 + k].i = ws.fa[k].i * scale;
    }
    for (k = 1; k < nx; ++k) {
        r[nx - 1 - k].r = ws.fa[ws.nfft - k].r * scale;
        r[nx - 1 - k].i = ws.fa[ws.nfft - k].i * scale;
    }

    workspace_free(&ws);
    return KISS_CONV_OK;
}

#endif /* FIXED_POINT */
