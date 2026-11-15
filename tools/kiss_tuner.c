/*
 *  Copyright (c) 2003-2026, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_tuner.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define KISS_TUNER_MAX_MEDIAN_WINDOW 63
#define KISS_TUNER_DEFAULT_SILENCE_RMS 1e-4f
/* plausible fundamental range for peak picking */
#define KISS_TUNER_FMIN_HZ 50.0f
#define KISS_TUNER_FMAX_HZ 2000.0f
/* a sub-harmonic with more than this fraction of the peak magnitude
 * is considered the true fundamental (octave-down correction) */
#define KISS_TUNER_SUBHARMONIC_RATIO 0.3f
#define KISS_TUNER_MAX_OCTAVE_STEPS 2
/* minimum share of the total spectral magnitude the peak must hold for the
 * frame to be considered tonal; octave correction is skipped on noise-like
 * frames so transients cannot corrupt the smoothed track */
#define KISS_TUNER_TONALITY_MIN 0.05f

struct kiss_tuner_state {
    int nfft;
    int sample_rate;
    int median_window;
    int hist_count;
    int hist_pos;
    float silence_rms;
    kiss_fftr_cfg fftr;
    float *hann;        /* nfft */
    kiss_fft_scalar *windowed; /* nfft */
    kiss_fft_cpx *spectrum;    /* nfft/2+1 */
    float *mags;        /* nfft/2+1 */
    float *history;     /* median_window */
    float *sorted;      /* median_window, scratch for median */
};

static int clamp_median_window(int w)
{
    if (w < 1)
        w = 1;
    if (w > KISS_TUNER_MAX_MEDIAN_WINDOW)
        w = KISS_TUNER_MAX_MEDIAN_WINDOW;
    if ((w & 1) == 0)
        ++w;
    if (w > KISS_TUNER_MAX_MEDIAN_WINDOW)
        w = KISS_TUNER_MAX_MEDIAN_WINDOW - 1;
    return w;
}

kiss_tuner_cfg kiss_tuner_alloc(int nfft, int sample_rate, int median_window, void *mem, size_t *lenmem)
{
    kiss_tuner_cfg st = NULL;
    size_t subsize = 0, memneeded;
    char *ptr;
    int i;

    if (nfft <= 0 || (nfft & 1) || sample_rate <= 0)
        return NULL;

    median_window = clamp_median_window(median_window);

    kiss_fftr_alloc(nfft, 0, NULL, &subsize);
    memneeded = sizeof(struct kiss_tuner_state) + subsize
        + sizeof(float) * nfft                /* hann */
        + sizeof(kiss_fft_scalar) * nfft      /* windowed */
        + sizeof(kiss_fft_cpx) * (nfft / 2 + 1) /* spectrum */
        + sizeof(float) * (nfft / 2 + 1)      /* mags */
        + sizeof(float) * median_window       /* history */
        + sizeof(float) * median_window;      /* sorted */

    if (lenmem == NULL) {
        st = (kiss_tuner_cfg) malloc(memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_tuner_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->nfft = nfft;
    st->sample_rate = sample_rate;
    st->median_window = median_window;
    st->hist_count = 0;
    st->hist_pos = 0;
    st->silence_rms = KISS_TUNER_DEFAULT_SILENCE_RMS;

    ptr = (char *) (st + 1);
    st->fftr = (kiss_fftr_cfg) ptr;
    ptr += subsize;
    st->hann = (float *) ptr;
    ptr += sizeof(float) * nfft;
    st->windowed = (kiss_fft_scalar *) ptr;
    ptr += sizeof(kiss_fft_scalar) * nfft;
    st->spectrum = (kiss_fft_cpx *) ptr;
    ptr += sizeof(kiss_fft_cpx) * (nfft / 2 + 1);
    st->mags = (float *) ptr;
    ptr += sizeof(float) * (nfft / 2 + 1);
    st->history = (float *) ptr;
    ptr += sizeof(float) * median_window;
    st->sorted = (float *) ptr;

    kiss_fftr_alloc(nfft, 0, st->fftr, &subsize);

    for (i = 0; i < nfft; ++i)
        st->hann[i] = (float) (0.5 - 0.5 * cos(2.0 * 3.14159265358979323846 * i / (nfft - 1)));

    return st;
}

void kiss_tuner_set_silence_threshold(kiss_tuner_cfg cfg, float rms_threshold)
{
    if (cfg && rms_threshold >= 0.0f)
        cfg->silence_rms = rms_threshold;
}

void kiss_tuner_free(kiss_tuner_cfg cfg)
{
    free(cfg);
}

static float median_of_history(kiss_tuner_cfg st)
{
    int i, j;
    memcpy(st->sorted, st->history, sizeof(float) * (size_t) st->hist_count);
    for (i = 1; i < st->hist_count; ++i) {
        float key = st->sorted[i];
        for (j = i - 1; j >= 0 && st->sorted[j] > key; --j)
            st->sorted[j + 1] = st->sorted[j];
        st->sorted[j + 1] = key;
    }
    return st->sorted[st->hist_count / 2];
}

static float mag_at_freq(const kiss_tuner_cfg st, float freq)
{
    int k = (int) (freq * st->nfft / st->sample_rate + 0.5f);
    if (k < 0)
        k = 0;
    if (k > st->nfft / 2)
        k = st->nfft / 2;
    return st->mags[k];
}

kiss_tuner_status kiss_tuner_process(kiss_tuner_cfg cfg, const kiss_fft_scalar *frame, float *f0_hz)
{
    int i, kmin, kmax, kpeak, nsteps;
    double energy = 0.0;
    double fmin_bin, fmax_bin;
    float mag_sum = 0.0f;
    float raw_freq, smoothed_freq, delta, peak_mag;
    float yl, yc, yr, denom;

    if (!cfg || !frame || !f0_hz)
        return KISS_TUNER_SILENCE;

    /* --- silence gate: explicit status, history left untouched --- */
    for (i = 0; i < cfg->nfft; ++i) {
        double v = (double) frame[i];
        energy += v * v;
    }
    if (sqrt(energy / cfg->nfft) < (double) cfg->silence_rms) {
        *f0_hz = 0.0f;
        return KISS_TUNER_SILENCE;
    }

    /* --- stage 1: spectral peak pick on the Hann-windowed magnitude spectrum --- */
    for (i = 0; i < cfg->nfft; ++i)
        cfg->windowed[i] = (kiss_fft_scalar) (frame[i] * cfg->hann[i]);
    kiss_fftr(cfg->fftr, cfg->windowed, cfg->spectrum);
    for (i = 0; i <= cfg->nfft / 2; ++i) {
        cfg->mags[i] = (float) hypot((double) cfg->spectrum[i].r, (double) cfg->spectrum[i].i);
        mag_sum += cfg->mags[i];
    }

    fmin_bin = ceil((double) KISS_TUNER_FMIN_HZ * cfg->nfft / cfg->sample_rate);
    fmax_bin = floor((double) KISS_TUNER_FMAX_HZ * cfg->nfft / cfg->sample_rate);
    kmin = (int) fmin_bin;
    kmax = (int) fmax_bin;
    if (kmin < 1)
        kmin = 1;
    if (kmax > cfg->nfft / 2 - 1)
        kmax = cfg->nfft / 2 - 1;
    if (kmax < kmin)
        kmax = kmin;

    kpeak = kmin;
    for (i = kmin + 1; i <= kmax; ++i) {
        if (cfg->mags[i] > cfg->mags[kpeak])
            kpeak = i;
    }

    /* --- stage 2: parabolic interpolation for sub-bin accuracy --- */
    yl = cfg->mags[kpeak - 1];
    yc = cfg->mags[kpeak];
    yr = cfg->mags[kpeak + 1];
    denom = yl - 2.0f * yc + yr;
    delta = 0.0f;
    if (fabsf(denom) > 1e-12f) {
        delta = 0.5f * (yl - yr) / denom;
        if (delta > 0.5f)
            delta = 0.5f;
        else if (delta < -0.5f)
            delta = -0.5f;
    }
    raw_freq = (kpeak + delta) * (float) cfg->sample_rate / cfg->nfft;

    /* --- stage 3: median smoothing over the configurable window --- */
    cfg->history[cfg->hist_pos] = raw_freq;
    cfg->hist_pos = (cfg->hist_pos + 1) % cfg->median_window;
    if (cfg->hist_count < cfg->median_window)
        ++cfg->hist_count;
    smoothed_freq = median_of_history(cfg);

    /* --- stage 4: octave correction via sub-harmonic spectral check ---
     * only on tonal frames, where the spectrum is trustworthy */
    if (cfg->mags[kpeak] > KISS_TUNER_TONALITY_MIN * mag_sum) {
        peak_mag = mag_at_freq(cfg, smoothed_freq);
        for (nsteps = 0; nsteps < KISS_TUNER_MAX_OCTAVE_STEPS; ++nsteps) {
            float half = smoothed_freq * 0.5f;
            if (half < KISS_TUNER_FMIN_HZ)
                break;
            if (mag_at_freq(cfg, half) <= KISS_TUNER_SUBHARMONIC_RATIO * peak_mag)
                break;
            smoothed_freq = half;
            peak_mag = mag_at_freq(cfg, smoothed_freq);
        }
    }

    *f0_hz = smoothed_freq;
    return KISS_TUNER_OK;
}
