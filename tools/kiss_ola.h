/*
 * Overlap-add streaming helper for KISS FFT (real float signals).
 *
 * Splits a live stream into overlapping frames, runs each frame through
 * FFT -> (optional user processing) -> IFFT, and recombines frames with
 * overlap-add so frame boundaries stay glitch-free.
 *
 * Push/pop interface: input and output chunk sizes are fully decoupled.
 * Output sample k corresponds to input sample k (latency = nfft - hop).
 *
 * This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef KISS_OLA_H
#define KISS_OLA_H

#include <stdlib.h>
#include "kiss_fftr.h"

typedef struct {
    int nfft;             /* frame length (must be even)            */
    int hop;              /* frame step = nfft * (1 - overlap)      */
    int nframes;          /* frames processed so far                */
    long pushed;          /* total input samples accepted           */
    long finalized;       /* output samples ready to pop            */
    long popped;          /* output samples consumed                */
    float *inring;        /* ring with the last nfft input samples  */
    float *outring;       /* overlap-add accumulator, 4*nfft ring   */
    float *wring;         /* per-slot overlap coverage count        */
    float *frame, *tbuf;
    kiss_fft_cpx *freq;
    kiss_fftr_cfg fwd, inv;
} kiss_ola_state;

typedef kiss_ola_state *kiss_ola;

/* overlap in (0,1); nfft*(1-overlap) must divide nfft (e.g. 0.5, 0.75) */
static kiss_ola kiss_ola_alloc(int nfft, float overlap)
{
    kiss_ola st;
    int hop = nfft - (int)(overlap * nfft + 0.5f);
    if (nfft < 4 || (nfft & 1) || hop < 1 || nfft % hop)
        return 0;
    st = (kiss_ola)calloc(1, sizeof(*st));
    if (!st)
        return 0;
    st->nfft = nfft;
    st->hop = hop;
    st->inring = (float*)calloc(nfft, sizeof(float));
    st->outring = (float*)calloc(4 * nfft, sizeof(float));
    st->wring = (float*)calloc(4 * nfft, sizeof(float));
    st->frame = (float*)malloc(nfft * sizeof(float));
    st->tbuf = (float*)malloc(nfft * sizeof(float));
    st->freq = (kiss_fft_cpx*)malloc((nfft / 2 + 1) * sizeof(kiss_fft_cpx));
    st->fwd = kiss_fftr_alloc(nfft, 0, 0, 0);
    st->inv = kiss_fftr_alloc(nfft, 1, 0, 0);
    if (!st->inring || !st->outring || !st->wring || !st->frame ||
        !st->tbuf || !st->freq || !st->fwd || !st->inv) {
        free(st);
        return 0;
    }
    return st;
}

static void kiss_ola_free(kiss_ola st)
{
    if (!st)
        return;
    free(st->inring); free(st->outring); free(st->wring); free(st->frame);
    free(st->tbuf); free(st->freq);
    kiss_fft_free(st->fwd); kiss_fft_free(st->inv);
    free(st);
}

/* process one frame: gather -> FFT -> IFFT -> overlap-add into outring */
static void kiss_ola__frame(kiss_ola st)
{
    long base = (long)st->nframes * st->hop;
    int i, n = st->nfft;
    for (i = 0; i < n; i++)
        st->frame[i] = st->inring[(base + i) % n];
    kiss_fftr(st->fwd, st->frame, st->freq);
    kiss_fftri(st->inv, st->freq, st->tbuf);
    for (i = 0; i < n; i++)
        st->outring[(base + i) % (4 * n)] += st->tbuf[i] / n;
    for (i = 0; i < n; i++)
        st->wring[(base + i) % (4 * n)] += 1.f;
    st->nframes++;
    st->finalized = (long)st->nframes * st->hop;
}

/* push n input samples; returns samples actually consumed (< n if the
 * output ring is full, i.e. caller is not popping fast enough) */
static int kiss_ola_push(kiss_ola st, const float *in, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        while (st->pushed - (long)st->nframes * st->hop >= st->nfft) {
            if ((long)st->nframes * st->hop + st->nfft - st->popped > 3 * st->nfft)
                return i;
            kiss_ola__frame(st);
        }
        st->inring[st->pushed % st->nfft] = in[i];
        st->pushed++;
    }
    return i;
}

/* pop up to maxn finalized output samples; returns the actual count */
static int kiss_ola_pop(kiss_ola st, float *out, int maxn)
{
    long avail = st->finalized - st->popped;
    int i, idx, n = (int)(avail < maxn ? avail : maxn);
    for (i = 0; i < n; i++) {
        idx = (int)((st->popped + i) % (4 * st->nfft));
        out[i] = st->outring[idx] / st->wring[idx]; /* normalize by coverage */
        st->outring[idx] = 0.f;                     /* recycle ring slots    */
        st->wring[idx] = 0.f;
    }
    st->popped += n;
    return n;
}

/* number of output samples currently available */
static long kiss_ola_ready(const kiss_ola st)
{
    return st->finalized - st->popped;
}

#endif /* KISS_OLA_H */
