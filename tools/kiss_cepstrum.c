/*
 *  Cepstral pitch detection built on the KISS FFT real transforms.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_cepstrum.h"
#include "kiss_fftr.h"
#include <tgmath.h>

/* Below this squared magnitude the log is clamped, so pure silence never
 * reaches log(0). Audio at full scale has squared magnitudes of order nfft^2,
 * so this floor is far below any audible component. */
#define KISS_CEPSTRUM_LOG_FLOOR 1e-20f

struct kiss_cepstrum_state {
    kiss_cepstrum_config cfg;
    int qmin;              /* first quefrency bin inside the pitch range */
    int qmax;              /* last quefrency bin inside the pitch range  */
    kiss_fft_scalar rms;   /* RMS of the frame analyzed by compute()     */
    kiss_fftr_cfg fwd;
    kiss_fftr_cfg inv;
    kiss_fft_cpx *spec;    /* nfft/2+1 bins: log magnitude spectrum      */
    kiss_fft_scalar *win;  /* nfft Hann window coefficients              */
    kiss_fft_scalar *work; /* nfft samples: windowed frame, then cepstrum */
};

void kiss_cepstrum_default_config(kiss_cepstrum_config *cfg,
                                  int nfft, int sample_rate)
{
    cfg->nfft = nfft;
    cfg->sample_rate = sample_rate;
    cfg->min_pitch_hz = 60;
    cfg->max_pitch_hz = 1000;
    cfg->lifter_order = 8;
    cfg->silence_rms = (kiss_fft_scalar)1e-4;
    cfg->voicing_threshold = (kiss_fft_scalar)0.05;
}

/* Smallest/largest quefrency bin whose implied pitch lies in [fmin, fmax]. */
static int quefrency_min(int sample_rate, int fmax)
{
    return (sample_rate + fmax - 1) / fmax; /* ceil(fs / fmax) */
}

static int quefrency_max(int sample_rate, int fmin)
{
    return sample_rate / fmin;               /* floor(fs / fmin) */
}

static kiss_fft_scalar frame_rms(const kiss_fft_scalar *frame, int nfft)
{
    kiss_fft_scalar sumsq = 0;
    int k;
    for (k = 0; k < nfft; ++k)
        sumsq += frame[k] * frame[k];
    return sqrt(sumsq / nfft);
}

static int config_valid(const kiss_cepstrum_config *cfg, int *qmin, int *qmax)
{
    if (cfg == NULL || cfg->nfft <= 0 || (cfg->nfft & 1) != 0)
        return 0;
    if (cfg->sample_rate <= 0)
        return 0;
    if (cfg->min_pitch_hz <= 0 || cfg->max_pitch_hz <= 0 ||
        cfg->min_pitch_hz >= cfg->max_pitch_hz)
        return 0;
    if (cfg->max_pitch_hz > cfg->sample_rate / 2)
        return 0;
    if (cfg->silence_rms < 0 || cfg->voicing_threshold < 0)
        return 0;
    if (cfg->lifter_order < 0 || cfg->lifter_order > cfg->nfft / 2)
        return 0;

    *qmin = quefrency_min(cfg->sample_rate, cfg->max_pitch_hz);
    *qmax = quefrency_max(cfg->sample_rate, cfg->min_pitch_hz);
    if (*qmax > cfg->nfft / 2)
        *qmax = cfg->nfft / 2;
    return *qmin <= *qmax;
}

kiss_cepstrum_cfg kiss_cepstrum_alloc(const kiss_cepstrum_config *cfg)
{
    int qmin, qmax, order, k;
    struct kiss_cepstrum_state *st;

    if (!config_valid(cfg, &qmin, &qmax))
        return NULL;

    st = (struct kiss_cepstrum_state *)KISS_FFT_MALLOC(sizeof(*st));
    if (st == NULL)
        return NULL;

    st->cfg = *cfg;
    st->qmin = qmin;
    st->qmax = qmax;
    st->fwd = NULL;
    st->inv = NULL;
    st->spec = NULL;
    st->win = NULL;
    st->work = NULL;

    /* The lifter must never erase the whole pitch search range. */
    order = cfg->lifter_order;
    if (order > qmax)
        order = qmax;
    st->cfg.lifter_order = order;

    st->fwd = kiss_fftr_alloc(cfg->nfft, 0, NULL, NULL);
    st->inv = kiss_fftr_alloc(cfg->nfft, 1, NULL, NULL);
    st->spec = (kiss_fft_cpx *)KISS_FFT_MALLOC(
        (size_t)(cfg->nfft / 2 + 1) * sizeof(kiss_fft_cpx));
    st->win = (kiss_fft_scalar *)KISS_FFT_MALLOC(
        (size_t)cfg->nfft * sizeof(kiss_fft_scalar));
    st->work = (kiss_fft_scalar *)KISS_FFT_MALLOC(
        (size_t)cfg->nfft * sizeof(kiss_fft_scalar));

    if (st->fwd == NULL || st->inv == NULL || st->spec == NULL ||
        st->win == NULL || st->work == NULL) {
        if (st->fwd != NULL) KISS_FFT_FREE(st->fwd);
        if (st->inv != NULL) KISS_FFT_FREE(st->inv);
        KISS_FFT_FREE(st->spec);
        KISS_FFT_FREE(st->win);
        KISS_FFT_FREE(st->work);
        KISS_FFT_FREE(st);
        return NULL;
    }

    /* Periodic Hann window, matching the usual STFT convention. */
    for (k = 0; k < cfg->nfft; ++k) {
        kiss_fft_scalar phase =
            (kiss_fft_scalar)(2 * 3.14159265358979323846) * k / cfg->nfft;
        st->win[k] = (kiss_fft_scalar)0.5 * ((kiss_fft_scalar)1 - cos(phase));
    }

    return st;
}

int kiss_cepstrum_set_lifter_order(kiss_cepstrum_cfg st, int order)
{
    if (st == NULL || order < 0 || order > st->cfg.nfft / 2)
        return -1;
    if (order > st->qmax)
        order = st->qmax;
    st->cfg.lifter_order = order;
    return order;
}

kiss_fft_scalar kiss_cepstrum_compute(kiss_cepstrum_cfg st,
                                      const kiss_fft_scalar *frame,
                                      kiss_fft_scalar *cepstrum)
{
    const int nfft = st->cfg.nfft;
    const int nspec = nfft / 2 + 1;
    int k;

    for (k = 0; k < nfft; ++k)
        st->work[k] = frame[k] * st->win[k];
    st->rms = frame_rms(frame, nfft);

    /* Forward transform: frame -> short-time spectrum. */
    kiss_fftr(st->fwd, st->work, st->spec);

    /* Log magnitude spectrum. Squared magnitudes avoid a square root per
     * bin; the factor 1/2 keeps the result equivalent to log(|X|). */
    for (k = 0; k < nspec; ++k) {
        kiss_fft_scalar mag2 =
            st->spec[k].r * st->spec[k].r +
            st->spec[k].i * st->spec[k].i;
        if (mag2 < (kiss_fft_scalar)KISS_CEPSTRUM_LOG_FLOOR)
            mag2 = (kiss_fft_scalar)KISS_CEPSTRUM_LOG_FLOOR;
        st->spec[k].r = (kiss_fft_scalar)0.5 * log(mag2);
        st->spec[k].i = 0;
    }

    /* Inverse transform: log spectrum -> real cepstrum. kiss_fftri leaves
     * a factor of nfft in the output, so normalize it back out. */
    kiss_fftri(st->inv, st->spec, st->work);
    for (k = 0; k < nfft; ++k)
        st->work[k] /= nfft;

    /* High-time liftering: drop the low-quefrency content (spectral
     * envelope / formants) and its mirror, keeping the periodic pitch
     * ripples. order == 0 leaves the cepstrum untouched. */
    if (st->cfg.lifter_order > 0) {
        for (k = 0; k < st->cfg.lifter_order; ++k) {
            st->work[k] = 0;
            st->work[nfft - 1 - k] = 0;
        }
    }

    for (k = 0; k < nfft; ++k)
        cepstrum[k] = st->work[k];

    return st->rms;
}

int kiss_cepstrum_find_peak(kiss_cepstrum_cfg st,
                            const kiss_fft_scalar *cepstrum,
                            kiss_fft_scalar *peak)
{
    int best = st->qmin;
    kiss_fft_scalar bestval = cepstrum[st->qmin];
    int q;

    for (q = st->qmin + 1; q <= st->qmax; ++q) {
        if (cepstrum[q] > bestval) {
            bestval = cepstrum[q];
            best = q;
        }
    }
    if (peak != NULL)
        *peak = bestval;
    return best;
}

int kiss_cepstrum_detect(kiss_cepstrum_cfg st,
                         const kiss_fft_scalar *frame,
                         kiss_cepstrum_pitch *result)
{
    kiss_fft_scalar peak;
    int bin;

    result->voiced = 0;
    result->quefrency_bin = 0;
    result->pitch_hz = 0;
    result->rms = 0;
    result->peak = 0;

    /* Silence is decided on the raw frame, before any transform: it saves
     * the FFT work and keeps the log floor out of the reported result. */
    result->rms = frame_rms(frame, st->cfg.nfft);
    if (result->rms < st->cfg.silence_rms)
        return KISS_CEPSTRUM_SILENCE;

    /* st->work doubles as the output cepstrum; compute() copies into it. */
    kiss_cepstrum_compute(st, frame, st->work);
    result->rms = st->rms;
    bin = kiss_cepstrum_find_peak(st, st->work, &peak);
    result->peak = peak;

    if (peak < st->cfg.voicing_threshold)
        return KISS_CEPSTRUM_UNVOICED;

    result->voiced = 1;
    result->quefrency_bin = bin;
    result->pitch_hz = (kiss_fft_scalar)st->cfg.sample_rate / bin;
    return KISS_CEPSTRUM_VOICED;
}
