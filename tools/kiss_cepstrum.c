/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Cepstral pitch (fundamental period) analysis tool.
 *
 * Two-stage pipeline per input frame:
 *   stage 1: log magnitude spectrum -> real cepstrum (via kiss_fftr/kiss_fftri)
 *   stage 2: smooth cepstrum (moving average) -> pick peak in quefrency range
 *            -> pitch period in samples (parabolic-interpolated)
 *
 * Cepstrum length and smoothing window are configurable.
 * Low-energy frames report an explicit LOW_ENERGY status.
 * Invalid length/configuration reports INVALID_LENGTH and exits non-zero.
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "kiss_fftr.h"

#define CEPSTRUM_TOOL_VERSION "1.0.0"

enum cep_status {
    CEP_OK = 0,
    CEP_LOW_ENERGY = 1,
    CEP_INVALID_LENGTH = 2,
    CEP_NO_PEAK = 3
};

static const char * cep_status_str(int st)
{
    switch (st) {
    case CEP_OK: return "OK";
    case CEP_LOW_ENERGY: return "LOW_ENERGY";
    case CEP_INVALID_LENGTH: return "INVALID_LENGTH";
    default: return "NO_PEAK";
    }
}

struct cep_cfg {
    int nfft;       /* frame / FFT length, must be even                */
    int ceplen;     /* number of cepstral coefficients kept: 2..nfft/2 */
    int smoothwin;  /* moving-average window over cepstrum: 1..ceplen  */
    int pmin;       /* quefrency search range: 1 <= pmin < pmax        */
    int pmax;       /*                               pmax < ceplen     */
    double ethresh; /* mean-square energy threshold for LOW_ENERGY     */
};

static int cep_cfg_valid(const struct cep_cfg *c)
{
    if (c->nfft < 8 || (c->nfft & 1))
        return 0;
    if (c->ceplen < 2 || c->ceplen > c->nfft / 2)
        return 0;
    if (c->smoothwin < 1 || c->smoothwin > c->ceplen)
        return 0;
    if (c->pmin < 1 || c->pmin >= c->pmax || c->pmax >= c->ceplen)
        return 0;
    if (!(c->ethresh >= 0.0))
        return 0;
    return 1;
}

/*
 * Run the two-stage pipeline on one frame.
 * Returns a cep_status code; on CEP_OK *period holds the pitch period
 * in samples (parabolic interpolated) and *peakq the integer quefrency
 * of the cepstral peak.
 */
static int cepstrum_pitch(const kiss_fft_scalar *frame,
                          const struct cep_cfg *c,
                          double *period, int *peakq,
                          kiss_fftr_cfg fwd, kiss_fftr_cfg inv,
                          kiss_fft_cpx *spec, kiss_fft_scalar *cep,
                          kiss_fft_scalar *work)
{
    int k, q, nfreq = c->nfft / 2 + 1;
    double energy = 0.0;

    if (!cep_cfg_valid(c))
        return CEP_INVALID_LENGTH;

    for (k = 0; k < c->nfft; ++k)
        energy += (double)frame[k] * (double)frame[k];
    energy /= c->nfft;
    if (energy < c->ethresh)
        return CEP_LOW_ENERGY;

    /* stage 1: log magnitude spectrum -> cepstrum */
    kiss_fftr(fwd, frame, spec);
    for (k = 0; k < nfreq; ++k) {
        double mag = hypot((double)spec[k].r, (double)spec[k].i);
        spec[k].r = (kiss_fft_scalar)log(mag + 1e-12);
        spec[k].i = 0;
    }
    kiss_fftri(inv, spec, cep);

    /* stage 2: smooth cepstrum (centered moving average) then peak pick */
    {
        int half = c->smoothwin / 2;
        for (q = 0; q < c->ceplen; ++q) {
            double acc = 0.0;
            int cnt = 0;
            int j;
            for (j = q - half; j <= q + half; ++j) {
                if (j >= 0 && j < c->ceplen) {
                    acc += cep[j];
                    ++cnt;
                }
            }
            work[q] = (kiss_fft_scalar)(acc / cnt);
        }
    }

    /*
     * Peak picking with sub-harmonic (octave) guard: the cepstrum of a
     * harmonic signal peaks at the true period AND its integer multiples
     * with nearly equal height, so a plain argmax can lock onto 2x the
     * true period.  Take the first quefrency whose smoothed cepstrum
     * exceeds 90% of the global max, then refine to the local peak.
     */
    {
        int best, qend;
        double vmax = work[c->pmin];
        int half = c->smoothwin / 2;
        for (q = c->pmin + 1; q <= c->pmax; ++q)
            if (work[q] > vmax)
                vmax = work[q];
        if (vmax <= 0)
            return CEP_NO_PEAK;
        best = c->pmin;
        while (best < c->pmax && work[best] < 0.9 * vmax)
            ++best;
        qend = best + half;
        if (qend > c->pmax)
            qend = c->pmax;
        for (q = best + 1; q <= qend; ++q)
            if (work[q] > work[best])
                best = q;
        *peakq = best;
        /* parabolic interpolation for sub-sample period estimate */
        if (best > c->pmin && best < c->pmax) {
            double y0 = work[best - 1], y1 = work[best], y2 = work[best + 1];
            double denom = y0 - 2.0 * y1 + y2;
            double delta = (denom != 0.0) ? 0.5 * (y0 - y2) / denom : 0.0;
            *period = best + delta;
        } else {
            *period = best;
        }
    }
    return CEP_OK;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "cepstral pitch analysis tool (version %s)\n"
        "usage: %s -n nfft [-m ceplen] [-w smoothwin] [-e ethresh] [-p pmin:pmax] [-t]\n"
        "  -n nfft      frame/FFT length (even, >=8)\n"
        "  -m ceplen    cepstrum length kept for analysis (2..nfft/2, default nfft/2)\n"
        "  -w smoothwin cepstrum smoothing window (1..ceplen, default 1 = no smoothing)\n"
        "  -e ethresh   low-energy threshold on mean-square (default 1e-8)\n"
        "  -p pmin:pmax quefrency search range in samples (default nfft/16:3*nfft/8)\n"
        "  -t           run built-in self test and exit\n"
        "reads raw kiss_fft_scalar frames from stdin, one status line per frame\n",
        CEPSTRUM_TOOL_VERSION, prog);
}

#define CEP_SELFTEST_PITCH_BOUND 1.0 /* max allowed |estimated-true| period, samples */

static int cep_selftest(void)
{
    int nfail = 0;
    struct cep_cfg c;
    kiss_fftr_cfg fwd, inv;
    kiss_fft_cpx *spec;
    kiss_fft_scalar *cep, *work, *frame;
    const int nfft = 1024;

    c.nfft = nfft;
    c.ceplen = nfft / 2;
    c.smoothwin = 5;
    c.pmin = nfft / 16;
    c.pmax = 3 * nfft / 8;
    c.ethresh = 1e-8;

    fwd = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    inv = kiss_fftr_alloc(nfft, 1, NULL, NULL);
    spec = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (nfft / 2 + 1));
    cep = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * nfft);
    work = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * nfft);
    frame = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * nfft);

    /* test 1..3: synthetic harmonic signals with known pitch period */
    {
        static const double f0s[3] = {8.0, 13.0, 16.0};
        static const int wins[3] = {5, 5, 1};
        int t;
        for (t = 0; t < 3; ++t) {
            double f0 = f0s[t];
            double trueperiod = nfft / f0;
            double period = 0.0, err;
            int peakq = 0, st, n, h;
            c.smoothwin = wins[t];
            for (n = 0; n < nfft; ++n) {
                double v = 0.0;
                for (h = 1; h <= 8; ++h)
                    v += cos(2.0 * M_PI * h * f0 * n / nfft) / h;
                frame[n] = (kiss_fft_scalar)v;
            }
            st = cepstrum_pitch(frame, &c, &period, &peakq, fwd, inv, spec, cep, work);
            err = fabs(period - trueperiod);
            if (st == CEP_OK && err <= CEP_SELFTEST_PITCH_BOUND) {
                printf("selftest pitch f0=%.0f w=%d: status=OK true_period=%.4f est_period=%.4f err=%.4f bound=%.1f PASS\n",
                       f0, wins[t], trueperiod, period, err, CEP_SELFTEST_PITCH_BOUND);
            } else {
                printf("selftest pitch f0=%.0f w=%d: status=%s true_period=%.4f est_period=%.4f err=%.4f bound=%.1f FAIL\n",
                       f0, wins[t], cep_status_str(st), trueperiod, period, err, CEP_SELFTEST_PITCH_BOUND);
                ++nfail;
            }
        }
    }
    c.smoothwin = 5;

    /* test 3: low-energy frame must report LOW_ENERGY */
    {
        double period = 0.0;
        int peakq = 0, st, n;
        for (n = 0; n < nfft; ++n)
            frame[n] = (kiss_fft_scalar)1e-6;
        st = cepstrum_pitch(frame, &c, &period, &peakq, fwd, inv, spec, cep, work);
        if (st == CEP_LOW_ENERGY)
            printf("selftest low_energy: status=LOW_ENERGY PASS\n");
        else {
            printf("selftest low_energy: status=%s (expected LOW_ENERGY) FAIL\n", cep_status_str(st));
            ++nfail;
        }
    }

    /* test 4: invalid lengths must report INVALID_LENGTH */
    {
        struct cep_cfg bad;
        double period = 0.0;
        int peakq = 0, nbad = 0;
        bad = c; bad.nfft = 1023;                 /* odd nfft            */
        nbad += (cepstrum_pitch(frame, &bad, &period, &peakq, fwd, inv, spec, cep, work) == CEP_INVALID_LENGTH);
        bad = c; bad.ceplen = nfft;               /* ceplen > nfft/2     */
        nbad += (cepstrum_pitch(frame, &bad, &period, &peakq, fwd, inv, spec, cep, work) == CEP_INVALID_LENGTH);
        bad = c; bad.smoothwin = 0;               /* window < 1          */
        nbad += (cepstrum_pitch(frame, &bad, &period, &peakq, fwd, inv, spec, cep, work) == CEP_INVALID_LENGTH);
        bad = c; bad.pmin = 100; bad.pmax = 100;  /* empty search range  */
        nbad += (cepstrum_pitch(frame, &bad, &period, &peakq, fwd, inv, spec, cep, work) == CEP_INVALID_LENGTH);
        bad = c; bad.pmax = bad.ceplen;           /* pmax out of range   */
        nbad += (cepstrum_pitch(frame, &bad, &period, &peakq, fwd, inv, spec, cep, work) == CEP_INVALID_LENGTH);
        if (nbad == 5)
            printf("selftest invalid_length: 5/5 bad configs rejected with INVALID_LENGTH PASS\n");
        else {
            printf("selftest invalid_length: %d/5 bad configs rejected FAIL\n", nbad);
            ++nfail;
        }
    }

    free(frame); free(work); free(cep); free(spec);
    kiss_fftr_free(fwd); kiss_fftr_free(inv);

    printf("selftest %s (%d failure%s)\n", nfail ? "FAILED" : "PASSED", nfail, nfail == 1 ? "" : "s");
    return nfail ? 1 : 0;
}

int main(int argc, char **argv)
{
    struct cep_cfg c;
    int opt, selftest = 0;
    kiss_fftr_cfg fwd, inv;
    kiss_fft_cpx *spec;
    kiss_fft_scalar *cep, *work, *frame;
    long frameno = 0;

    c.nfft = 1024;
    c.ceplen = 0; /* 0 => default nfft/2 after parsing */
    c.smoothwin = 1;
    c.pmin = 0;
    c.pmax = 0;
    c.ethresh = 1e-8;

    while ((opt = getopt(argc, argv, "n:m:w:e:p:th")) != -1) {
        switch (opt) {
        case 'n': c.nfft = atoi(optarg); break;
        case 'm': c.ceplen = atoi(optarg); break;
        case 'w': c.smoothwin = atoi(optarg); break;
        case 'e': c.ethresh = atof(optarg); break;
        case 'p':
            if (sscanf(optarg, "%d:%d", &c.pmin, &c.pmax) != 2) {
                fprintf(stderr, "error: invalid -p range '%s' (want pmin:pmax)\n", optarg);
                return CEP_INVALID_LENGTH;
            }
            break;
        case 't': selftest = 1; break;
        default:
            usage(argv[0]);
            return CEP_INVALID_LENGTH;
        }
    }

    if (selftest)
        return cep_selftest();

    if (c.ceplen == 0)
        c.ceplen = c.nfft / 2;
    if (c.pmin == 0 && c.pmax == 0) {
        c.pmin = c.nfft / 16;
        c.pmax = 3 * c.nfft / 8;
    }

    if (!cep_cfg_valid(&c)) {
        fprintf(stderr,
            "error: INVALID_LENGTH nfft=%d ceplen=%d smoothwin=%d pmin=%d pmax=%d ethresh=%g\n",
            c.nfft, c.ceplen, c.smoothwin, c.pmin, c.pmax, c.ethresh);
        usage(argv[0]);
        return CEP_INVALID_LENGTH;
    }

    fwd = kiss_fftr_alloc(c.nfft, 0, NULL, NULL);
    inv = kiss_fftr_alloc(c.nfft, 1, NULL, NULL);
    spec = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (c.nfft / 2 + 1));
    cep = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * c.nfft);
    work = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * c.nfft);
    frame = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * c.nfft);
    if (!fwd || !inv || !spec || !cep || !work || !frame) {
        fprintf(stderr, "error: allocation failed\n");
        return 1;
    }

    while (fread(frame, sizeof(kiss_fft_scalar), c.nfft, stdin) == (size_t)c.nfft) {
        double period = 0.0;
        int peakq = 0;
        int st = cepstrum_pitch(frame, &c, &period, &peakq, fwd, inv, spec, cep, work);
        if (st == CEP_OK)
            printf("frame=%ld status=OK period=%.4f peak_quefrency=%d\n", frameno, period, peakq);
        else
            printf("frame=%ld status=%s\n", frameno, cep_status_str(st));
        ++frameno;
    }

    free(frame); free(work); free(cep); free(spec);
    kiss_fftr_free(fwd); kiss_fftr_free(inv);
    return 0;
}
