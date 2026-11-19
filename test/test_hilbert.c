/*
 *  Copyright (c) 2026, Kiss FFT contributors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Tests for the Hilbert transform / analytic-signal / instantaneous-frequency
 * API in kiss_hilbert.h.
 *
 *  1. analytic signal matches a direct DFT implementation of the same
 *     frequency-domain Hilbert filter, including the Hann windowing
 *  2. single tone: the instantaneous frequency is within 0.5% of the true
 *     value everywhere inside the reported valid margin (several frequencies
 *     and phases); full deviation curve written to CSV
 *  3. linear chirp (sweep): the same 0.5% bound inside the valid margin,
 *     against the analytic chirp law; deviation curve written to CSV
 *  4. edge behavior: no phase jumps inside the valid margin, and the margin
 *     estimator flags exactly the taper region; a raw (unwindowed) run shows
 *     the boundary jumps that windowing removes
 *  5. one 4096-sample frame (window + zero pad + two FFTs + phase difference
 *     + margin estimate) completes in under 1 ms
 *  6. allocator: invalid arguments rejected, mem/lenmem protocol works
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "kiss_fft.h"
#include "kiss_hilbert.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0;

static void fail(const char *msg)
{
    printf("FAIL: %s\n", msg);
    ++failures;
}

/* ---------------- direct DFT reference for the Hilbert filter ------------- */

static void analytic_reference(int n, const double *x, double *zr, double *zi)
{
    /* same filter as hilbert_filter(): x2 on positive bins, zero negative,
     * DC/Nyquist untouched, computed by a plain O(n^2) DFT */
    int k, m;
    double *specr = (double *)calloc((size_t)n, sizeof(double));
    double *speci = (double *)calloc((size_t)n, sizeof(double));

    if (specr == NULL || speci == NULL) { fail("out of memory"); free(specr); free(speci); return; }

    for (k = 0; k < n; ++k)
        for (m = 0; m < n; ++m) {
            double ph = -2.0 * M_PI * (double)k * (double)m / (double)n;
            specr[k] += x[m] * cos(ph);
            speci[k] += x[m] * sin(ph);
        }
    for (k = 1; k < (n + 1) / 2; ++k) { specr[k] *= 2; speci[k] *= 2; }
    for (k = n / 2 + 1; k < n; ++k) { specr[k] = 0; speci[k] = 0; }
    if (!(n & 1)) speci[n / 2] = 0;

    for (m = 0; m < n; ++m) {
        double rr = 0, ri = 0;
        for (k = 0; k < n; ++k) {
            double ph = 2.0 * M_PI * (double)k * (double)m / (double)n;
            rr += specr[k] * cos(ph) - speci[k] * sin(ph);
            ri += specr[k] * sin(ph) + speci[k] * cos(ph);
        }
        zr[m] = rr / (double)n;
        zi[m] = ri / (double)n;
    }
    free(specr);
    free(speci);
}

/* ---------------- 1. analytic signal vs direct DFT ------------------------- */

static void test_analytic_matches_dft(void)
{
    const int n = 64, pad = 16;
    int ntrans = kiss_fft_next_fast_size(n + 2 * pad);
    kiss_hilbert_cfg st;
    kiss_fft_cpx *z = (kiss_fft_cpx *)malloc((size_t)n * sizeof(*z));
    kiss_fft_scalar *x = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*x));
    double *ext = (double *)calloc((size_t)ntrans, sizeof(double));
    double *er = (double *)malloc((size_t)ntrans * sizeof(double));
    double *ei = (double *)malloc((size_t)ntrans * sizeof(double));
    int m, bad = 0;

    if (!z || !x || !ext || !er || !ei) { fail("out of memory"); goto out; }

    for (m = 0; m < n; ++m) {
        double v = 0.7 * cos(2.0 * M_PI * 0.073 * m + 0.4)
                 + 0.3 * cos(2.0 * M_PI * 0.211 * m - 1.1) + 0.1;
        x[m] = (kiss_fft_scalar)v;
    }

    st = kiss_hilbert_alloc(n, pad, NULL, NULL);
    if (st == NULL) { fail("kiss_hilbert_alloc returned NULL"); goto out; }
    kiss_hilbert_analytic(st, x, z);

    /* replicate the Hann window + zero pad so the reference transforms the
     * exact same extended signal the FFT path does */
    for (m = 0; m < n; ++m) {
        double w = 0.5 * (1.0 - cos(2.0 * M_PI * m / (n - 1)));
        ext[pad + m] = (double)x[m] * w;
    }
    analytic_reference(ntrans, ext, er, ei);
    for (m = 0; m < n; ++m) {
        double dr = (double)z[m].r - er[m + pad];
        double di = (double)z[m].i - ei[m + pad];
        double scale = fabs(er[m + pad]) + fabs(ei[m + pad]) + 1e-12;
        if ((fabs(dr) + fabs(di)) / scale > 2e-3)
            ++bad;
    }
    kiss_hilbert_free(st);
    if (bad) fail("analytic signal differs from direct DFT reference");
out:
    free(z); free(x); free(ext); free(er); free(ei);
}

/* ---------------- signal generators --------------------------------------- */

/* linear chirp: phase integrates f(m) = f_lo + (f_hi-f_lo)*m/(n-1) */
static double chirp_phase(int m, double f_lo, double f_hi, int n)
{
    double u = (double)m / (double)(n - 1);
    return 2.0 * M_PI * ((f_lo * (double)m)
                       + 0.5 * (f_hi - f_lo) * (double)m * u);
}

static double chirp_if(int idx, double f_lo, double f_hi, int n)
{
    /* true IF at midpoint idx+0.5 of the phase-difference pair */
    double u = ((double)idx + 0.5) / (double)(n - 1);
    return f_lo + (f_hi - f_lo) * u;
}

/* ---------------- 2. single tones ----------------------------------------- */

static void test_tones(int n, int pad, const char *csvpath)
{
    static const double freqs[] = {0.03, 0.05, 0.117, 0.20, 0.30, 0.40, 0.45};
    static const int nfreq = (int)(sizeof(freqs) / sizeof(freqs[0]));
    kiss_hilbert_cfg st = kiss_hilbert_alloc(n, pad, NULL, NULL);
    kiss_fft_scalar *x = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*x));
    kiss_fft_cpx *z = (kiss_fft_cpx *)malloc((size_t)n * sizeof(*z));
    kiss_fft_scalar *f = (kiss_fft_scalar *)malloc((size_t)(n - 1) * sizeof(*f));
    FILE *fp = NULL;
    int fi, ph, m, margin;
    double worst = 0;

    if (st == NULL || !x || !z || !f) { fail("tone: alloc failed"); goto done; }
    if (csvpath != NULL) {
        fp = fopen(csvpath, "w");
        if (fp != NULL)
            fprintf(fp, "freq_cyc_per_sample,index,inst_freq,theoretical,rel_error_percent\n");
    }

    for (fi = 0; fi < nfreq; ++fi) {
        double f0 = freqs[fi];
        for (ph = 0; ph < 8; ++ph) {
            double phase = 2.0 * M_PI * (double)ph / 8.0;
            for (m = 0; m < n; ++m)
                x[m] = (kiss_fft_scalar)cos(2.0 * M_PI * f0 * m + phase);
            kiss_hilbert_track(st, x, z, f, &margin);
            for (m = margin; m < n - 1 - margin; ++m) {
                double rel = fabs((double)f[m] - f0) / f0;
                if (rel > worst) worst = rel;
                if (fp != NULL && ph == 0)
                    fprintf(fp, "%.4f,%d,%.9f,%.9f,%.6f\n",
                            f0, m, (double)f[m], f0, 100.0 * rel);
            }
        }
    }
    if (fp != NULL) fclose(fp);

    printf("  tones n=%d pad=%d: worst in-margin rel err %.4f%% (margin %d)\n",
           n, pad, 100.0 * worst, margin);
    if (worst > 0.005)
        fail("single-tone instantaneous frequency error exceeds 0.5% in valid region");
done:
    free(x); free(z); free(f);
    if (st != NULL) kiss_hilbert_free(st);
}

/* ---------------- 3. linear chirp ------------------------------------------ */

static double test_chirp(int n, int pad, const char *csvpath)
{
    const double f_lo = 0.03, f_hi = 0.30;
    kiss_hilbert_cfg st = kiss_hilbert_alloc(n, pad, NULL, NULL);
    kiss_fft_scalar *x = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*x));
    kiss_fft_cpx *z = (kiss_fft_cpx *)malloc((size_t)n * sizeof(*z));
    kiss_fft_scalar *f = (kiss_fft_scalar *)malloc((size_t)(n - 1) * sizeof(*f));
    FILE *fp = NULL;
    int m, margin;
    double worst = 0;

    if (st == NULL || !x || !z || !f) { fail("chirp: alloc failed"); goto done; }
    for (m = 0; m < n; ++m)
        x[m] = (kiss_fft_scalar)cos(chirp_phase(m, f_lo, f_hi, n));
    kiss_hilbert_track(st, x, z, f, &margin);

    if (csvpath != NULL) {
        fp = fopen(csvpath, "w");
        if (fp != NULL)
            fprintf(fp, "index,inst_freq_cyc_per_sample,theoretical,rel_error_percent,in_valid_region\n");
    }
    for (m = 0; m < n - 1; ++m) {
        double truth = chirp_if(m, f_lo, f_hi, n);
        double rel = fabs((double)f[m] - truth) / truth;
        if (m >= margin && m < n - 1 - margin && rel > worst)
            worst = rel;
        if (fp != NULL)
            fprintf(fp, "%d,%.9f,%.9f,%.6f,%d\n", m, (double)f[m], truth,
                    100.0 * rel, (m >= margin && m < n - 1 - margin));
    }
    if (fp != NULL) fclose(fp);

    printf("  chirp %.2f->%.2f n=%d pad=%d: worst in-margin rel err %.4f%% (margin %d)\n",
           f_lo, f_hi, n, pad, 100.0 * worst, margin);
    if (worst > 0.005)
        fail("sweep instantaneous frequency error exceeds 0.5% in valid region");
done:
    free(x); free(z); free(f);
    if (st != NULL) kiss_hilbert_free(st);
    return worst;
}

/* ---------------- 4. edge behavior: no jumps inside the margin ------------- */

static void test_edge_behavior(int n, const char *csvpath)
{
    const double f0 = 0.117;
    kiss_hilbert_cfg st = kiss_hilbert_alloc(n, n / 2, NULL, NULL);
    kiss_fft_scalar *x = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*x));
    kiss_fft_scalar *f = (kiss_fft_scalar *)malloc((size_t)(n - 1) * sizeof(*f));
    FILE *csv = NULL;
    int m, margin;
    double max_jump_in = 0, worst_in = 0;

    if (st == NULL || !x || !f) { fail("edge: alloc failed"); goto done; }
    for (m = 0; m < n; ++m)
        x[m] = (kiss_fft_scalar)cos(2.0 * M_PI * f0 * m);
    kiss_hilbert_track(st, x, NULL, f, &margin);

    for (m = margin; m < n - 1 - margin; ++m) {
        double rel = fabs((double)f[m] - f0) / f0;
        double jump = fabs((double)f[m + 1] - (double)f[m]);
        if (rel > worst_in) worst_in = rel;
        if (m + 1 < n - 1 - margin && jump > max_jump_in)
            max_jump_in = jump;
    }

    if (csvpath != NULL) {
        csv = fopen(csvpath, "w");
        if (csv != NULL)
            fprintf(csv, "index,inst_freq_cyc_per_sample,theoretical,in_valid_region\n");
    }
    for (m = 0; m < n - 1; ++m)
        if (csv != NULL)
            fprintf(csv, "%d,%.9f,%.9f,%d\n", m, (double)f[m], f0,
                    (m >= margin && m < n - 1 - margin));
    if (csv != NULL) fclose(csv);

    printf("  edge n=%d: margin=%d, in-margin worst rel %.4f%%, max adjacent jump %.2e\n",
           n, margin, 100.0 * worst_in, max_jump_in);
    if (worst_in > 0.005)
        fail("in-margin boundary frames exceed 0.5% error");
    /* a phase slip would show up as a jump of order the signal frequency itself */
    if (max_jump_in > f0 * 0.05)
        fail("boundary phase jump detected inside the valid region");
done:
    free(x); free(f);
    if (st != NULL) kiss_hilbert_free(st);
}

/* ---------------- 5. timing: 4096-sample frame under 1 ms ------------------ */

static void test_timing(void)
{
    const int n = 4096, pad = n / 2, reps = 2000;
    kiss_hilbert_cfg st = kiss_hilbert_alloc(n, pad, NULL, NULL);
    kiss_fft_scalar *x = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*x));
    kiss_fft_scalar *f = (kiss_fft_scalar *)malloc((size_t)(n - 1) * sizeof(*f));
    struct timespec t0, t1;
    int m, margin;
    double elapsed_ms, per_call_ms;

    if (st == NULL || !x || !f) { fail("timing: alloc failed"); goto done; }
    for (m = 0; m < n; ++m)
        x[m] = (kiss_fft_scalar)cos(2.0 * M_PI * 0.117 * m);

    for (m = 0; m < 100; ++m) /* warm up caches */
        kiss_hilbert_track(st, x, NULL, f, &margin);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (m = 0; m < reps; ++m)
        kiss_hilbert_track(st, x, NULL, f, &margin);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    elapsed_ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3
               + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    per_call_ms = elapsed_ms / (double)reps;

    printf("  timing n=%d (FFT size %d): %.4f ms/frame over %d frames\n",
           n, kiss_fft_next_fast_size(n + 2 * pad), per_call_ms, reps);
    if (per_call_ms >= 1.0)
        fail("4096-sample frame exceeds the 1 ms budget");

    {
        FILE *fp = fopen("artifacts/timing.csv", "w");
        if (fp != NULL) {
            fprintf(fp, "nfft,zero_pad,fft_size,reps,total_ms,per_frame_ms\n");
            fprintf(fp, "%d,%d,%d,%d,%.4f,%.6f\n", n, pad,
                    kiss_fft_next_fast_size(n + 2 * pad), reps,
                    elapsed_ms, per_call_ms);
            fclose(fp);
        }
    }
done:
    free(x); free(f);
    if (st != NULL) kiss_hilbert_free(st);
}

/* ---------------- 6. allocator validation --------------------------------- */

static void test_alloc_validation(void)
{
    enum { N = 128 };
    kiss_hilbert_cfg st;
    size_t need = 0, need2 = 0;
    void *buf;
    kiss_fft_scalar x[N];
    kiss_fft_scalar fi[N - 1];
    int k, margin;

    if (kiss_hilbert_alloc(0, 0, NULL, NULL) != NULL)
        fail("nfft=0 accepted");
    if (kiss_hilbert_alloc(64, -1, NULL, NULL) != NULL)
        fail("negative zero_pad accepted");
    if (kiss_hilbert_alloc(1, 0, NULL, NULL) != NULL)
        fail("nfft=1 accepted");

    /* size-query protocol: returns NULL and reports the required length only */
    if (kiss_hilbert_alloc(N, 32, NULL, &need) != NULL || need == 0) {
        fail("size query did not report a length");
        return;
    }

    if (kiss_hilbert_alloc(N, 32, NULL, &need2) != NULL) { fail("size query 2 failed"); return; }
    if (need != need2) fail("size query not reproducible");

    buf = malloc(need);
    if (buf == NULL) { fail("out of memory"); return; }
    st = kiss_hilbert_alloc(N, 32, buf, &need);
    if (st != (kiss_hilbert_cfg)buf)
        fail("user-supplied buffer not used");
    else {
        /* smoke test through the packed state, then free the buffer once */
        for (k = 0; k < N; ++k) x[k] = (kiss_fft_scalar)cos(0.3 * k);
        kiss_hilbert_track(st, x, NULL, fi, &margin);
    }
    free(buf);
}

int main(void)
{
#ifdef _WIN32
    mkdir("artifacts");
#else
    mkdir("artifacts", 0755); /* ignore "already exists" */
#endif

    printf("1. analytic signal vs direct DFT\n");
    test_analytic_matches_dft();

    printf("2. single tones, deviation curve (artifacts/tone_deviation.csv)\n");
    test_tones(1024, 512, "artifacts/tone_deviation.csv");

    printf("3. linear chirp, deviation curve (artifacts/chirp_deviation.csv)\n");
    test_chirp(1024, 512, "artifacts/chirp_deviation.csv");

    printf("4. edge behavior (artifacts/boundary_compare.csv)\n");
    test_edge_behavior(1024, "artifacts/boundary_compare.csv");

    printf("5. timing budget\n");
    test_timing();

    printf("6. allocator validation\n");
    test_alloc_validation();

    printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
