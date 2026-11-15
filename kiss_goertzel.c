/*
 *  Real-time keypress tone detection: framing + multi-point Goertzel
 *  + peak decision. See kiss_goertzel.h for the pipeline description.
 *
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_goertzel.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

struct kiss_goertzel_state {
    int frame_len;
    int hop;
    float sample_rate;
    int nfreqs;
    float threshold;
    float silence_db;
    float freqs[KISS_GOERTZEL_MAX_DET];
    double coeff[KISS_GOERTZEL_MAX_DET]; /* 2*cos(2*pi*f/fs) per tone */
    double silence_rms;                  /* RMS gate derived from silence_db */
    int frame_counter;
};

/* unit roundoff of the scalar type used by the full FFT */
static double kg_fft_roundoff(void)
{
#ifdef FIXED_POINT
    return 1.0; /* one LSB */
#else
    if (sizeof(kiss_fft_scalar) == sizeof(float))
        return (double)FLT_EPSILON * 0.5;
    return DBL_EPSILON * 0.5;
#endif
}

kiss_goertzel_cfg kiss_goertzel_alloc(
    int frame_len, int hop, float sample_rate,
    const float *freqs, int nfreqs,
    float threshold, float silence_db)
{
    kiss_goertzel_cfg st;
    int i;

    /* parameter validation: any violation is a hard error */
    if (frame_len < 8)
        return NULL;
    if (hop < 1)
        return NULL;
    if (!(sample_rate > 0.0f))
        return NULL;
    if (freqs == NULL || nfreqs < 1 || nfreqs > KISS_GOERTZEL_MAX_DET)
        return NULL;
    if (!(threshold > 0.0f && threshold <= 1.0f))
        return NULL;
    if (!(silence_db < 0.0f && silence_db >= -400.0f))
        return NULL;
    for (i = 0; i < nfreqs; ++i) {
        if (!(freqs[i] > 0.0f && freqs[i] < 0.5f * sample_rate))
            return NULL;
    }

    st = (kiss_goertzel_cfg)KISS_FFT_MALLOC(sizeof(struct kiss_goertzel_state));
    if (st == NULL)
        return NULL;

    st->frame_len = frame_len;
    st->hop = hop;
    st->sample_rate = sample_rate;
    st->nfreqs = nfreqs;
    st->threshold = threshold;
    st->silence_db = silence_db;
    st->silence_rms = pow(10.0, (double)silence_db / 20.0);
    st->frame_counter = 0;
    for (i = 0; i < nfreqs; ++i) {
        double w;
        st->freqs[i] = freqs[i];
        w = 2.0 * M_PI * (double)freqs[i] / (double)sample_rate;
        st->coeff[i] = 2.0 * cos(w);
    }
    return st;
}

/*
 * Generalized Goertzel power at one target frequency.
 * Evaluates |X(f)|^2 with X(f) = sum_n x[n] * exp(-j*2*pi*f*n/fs),
 * i.e. exactly the DFT coefficient of the frame at frequency f
 * (identical to an FFT bin when f is on the bin grid).
 * The recurrence runs in double precision.
 */
static double kg_tone_power(const kiss_goertzel_cfg st,
                            const kiss_fft_scalar *frame, int tone)
{
    const int n = st->frame_len;
    const double c = st->coeff[tone];
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    int i;

    for (i = 0; i < n; ++i) {
        s0 = (double)frame[i] + c * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return s1 * s1 + s2 * s2 - c * s1 * s2;
}

double kiss_goertzel_tone_power(kiss_goertzel_cfg cfg,
                                const kiss_fft_scalar *frame, int tone_index)
{
    if (cfg == NULL || frame == NULL || tone_index < 0 ||
        tone_index >= cfg->nfreqs)
        return -1.0;
    return kg_tone_power(cfg, frame, tone_index);
}

static void kg_analyze_frame(kiss_goertzel_cfg st,
                             const kiss_fft_scalar *frame,
                             kiss_goertzel_frame_result *res)
{
    const int n = st->frame_len;
    double energy = 0.0;
    double rms;
    int i;

    res->frame_index = st->frame_counter;
    res->is_silence = 0;
    res->n_det = 0;

    for (i = 0; i < n; ++i)
        energy += (double)frame[i] * (double)frame[i];
    rms = sqrt(energy / (double)n);

    /* stage 3a: silence gate */
    if (rms < st->silence_rms) {
        res->is_silence = 1;
        return;
    }

    /* stage 2 + 3b: per-tone Goertzel power and peak decision.
       confidence = 2*|X(f)|^2 / (N * sum x^2); by Parseval this is the
       fraction of the frame's spectral power carried by the tone, so a
       lone full-scale sine scores ~1 and white noise scores ~2/N. */
    for (i = 0; i < st->nfreqs; ++i) {
        double p = kg_tone_power(st, frame, i);
        double conf = 2.0 * p / ((double)n * energy);
        if (conf > 1.0)
            conf = 1.0;
        if (conf >= (double)st->threshold && res->n_det < KISS_GOERTZEL_MAX_DET) {
            int d = res->n_det++;
            res->det_id[d] = i;
            res->det_freq[d] = st->freqs[i];
            res->det_conf[d] = (float)conf;
        }
    }
}

int kiss_goertzel_process(kiss_goertzel_cfg cfg,
                          const kiss_fft_scalar *samples, int nsamples,
                          kiss_goertzel_frame_result *results, int max_results)
{
    int pos = 0;
    int nframes = 0;

    if (cfg == NULL || samples == NULL || results == NULL)
        return -1;
    if (nsamples < 0 || max_results < 0)
        return -2;

    while (pos + cfg->frame_len <= nsamples && nframes < max_results) {
        results[nframes].sample_offset = pos;
        kg_analyze_frame(cfg, samples + pos, &results[nframes]);
        ++cfg->frame_counter;
        ++nframes;
        pos += cfg->hop;
    }
    return nframes;
}

double kiss_goertzel_error_bound(kiss_goertzel_cfg cfg, double sum_abs_x)
{
    const double u_g = DBL_EPSILON * 0.5; /* Goertzel runs in double */
    double u_f;
    double log2n;

    if (cfg == NULL || sum_abs_x < 0.0)
        return -1.0;

    u_f = kg_fft_roundoff();
    log2n = ceil(log((double)cfg->frame_len) / log(2.0));

    /* Goertzel recurrence: error grows ~ 2*N*u_g*sum|x|.
       Full FFT: classical roundoff bound ~ log2(N) stages, each
       contributing a small multiple of u_f; factor 6 is conservative. */
    return (2.0 * (double)cfg->frame_len * u_g + 6.0 * log2n * u_f) * sum_abs_x;
}

void kiss_goertzel_free(kiss_goertzel_cfg cfg)
{
    KISS_FFT_FREE(cfg);
}
