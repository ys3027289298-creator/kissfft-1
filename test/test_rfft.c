/*
 *  Copyright (c) 2026, The kissfft authors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Tests for the canonical real FFT layer (kiss_rfft / kiss_irfft):
 *
 *   1. allocation rejects sizes that are not 2,3,5-smooth
 *   2. forward transform matches a naive analytic DFT (1e-10, double build)
 *   3. forward/inverse round trip error (1e-12, double build)
 *   4. the emitted spectrum is exactly Hermitian symmetric
 *   5. asymmetric spectra are rejected and every violating index pair reported
 *   6. NaN/Inf spectra are rejected
 *   7. 10000 rounds of 1024-point transforms run within the time budget
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L /* clock_gettime */
#endif

#include "kiss_rfft.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0;

static void fail(const char *msg)
{
    printf("FAIL: %s\n", msg);
    ++failures;
}

/* thresholds tight in double, relaxed for the float build */
#define IS_DOUBLE (sizeof(kiss_fft_scalar) == sizeof(double))
#define DFT_TOL (IS_DOUBLE ? 1e-10 : 2e-4)
#define ROUNDTRIP_TOL (IS_DOUBLE ? 1e-12 : 1e-5)
#define SYMM_TOL (IS_DOUBLE ? 1e-12 : 1e-6)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* 1. allocation validation                                           */
/* ------------------------------------------------------------------ */

static void test_alloc_sizes(void)
{
    static const int good[] = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 25, 27,
                               30, 32, 100, 125, 256, 375, 1000, 1024};
    static const int bad[] = {0, -1, -1024, 7, 11, 13, 14, 17, 19, 23, 49, 91};
    unsigned i;

    for (i = 0; i < sizeof(good) / sizeof(good[0]); ++i) {
        kiss_rfft_error e;
        kiss_rfft_cfg cfg = kiss_rfft_alloc(good[i], NULL, NULL, &e);
        if (cfg == NULL) {
            fail("kiss_rfft_alloc rejected a smooth size");
            printf("       nfft=%d code=%d msg=%s\n", good[i], e.code, e.message);
        }
        kiss_rfft_free(cfg);
    }
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        kiss_rfft_error e;
        kiss_rfft_cfg cfg = kiss_rfft_alloc(bad[i], NULL, NULL, &e);
        if (cfg != NULL || e.code != KISS_RFFT_ESIZE)
            fail("kiss_rfft_alloc accepted a non-smooth / non-positive size");
        kiss_rfft_free(cfg);
    }

    /* user-supplied buffer path: size query must be stable and sufficient */
    {
        size_t need = 0;
        void *buf;
        kiss_rfft_cfg cfg;
        kiss_rfft_error e;
        if (kiss_rfft_alloc(30, NULL, &need, &e) != NULL || need == 0)
            fail("size query with mem=NULL returned a config or zero size");
        buf = malloc(need);
        if (buf == NULL) {
            fail("out of memory in user-buffer test");
            return;
        }
        cfg = kiss_rfft_alloc(30, buf, &need, &e);
        if (cfg != (kiss_rfft_cfg)buf)
            fail("kiss_rfft_alloc did not place the config in the supplied buffer");
        kiss_rfft_free(cfg); /* malloc'd block, plain free is correct */
    }
}

/* ------------------------------------------------------------------ */
/* naive analytic DFT reference, evaluated in double                  */
/* ------------------------------------------------------------------ */

static void naive_dft(const double *x, int n, double *re, double *im)
{
    int k, j;
    for (k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        for (j = 0; j < n; ++j) {
            double phase = -2.0 * M_PI * (double)k * (double)j / (double)n;
            sr += x[j] * cos(phase);
            si += x[j] * sin(phase);
        }
        re[k] = sr;
        im[k] = si;
    }
}

static void make_signal(double *x, int n)
{
    int j;
    srand(12345 + (unsigned)n);
    for (j = 0; j < n; ++j) {
        int r = rand();
        double noise = (double)r / (double)RAND_MAX - 0.5;
        x[j] = 0.7 * sin(3.0 * 2.0 * M_PI * j / n)
             + 0.3 * cos(7.0 * 2.0 * M_PI * j / n)
             + 0.1 * noise;
    }
}

/* ------------------------------------------------------------------ */
/* 2-4. analytic DFT, round trip and exact Hermitian symmetry          */
/* ------------------------------------------------------------------ */

static void test_transforms(int n)
{
    kiss_rfft_cfg cfg;
    kiss_rfft_error e;
    kiss_fft_scalar *in, *back;
    kiss_fft_cpx *spec;
    double *xref, *dft_re, *dft_im;
    int k;
    double worst_dft = 0.0, worst_rt = 0.0;

    cfg = kiss_rfft_alloc(n, NULL, NULL, &e);
    if (cfg == NULL) {
        fail("allocation failed for a smooth size");
        return;
    }
    in = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*in));
    back = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*back));
    spec = (kiss_fft_cpx *)malloc((size_t)n * sizeof(*spec));
    xref = (double *)malloc((size_t)n * sizeof(*xref));
    dft_re = (double *)malloc((size_t)n * sizeof(*dft_re));
    dft_im = (double *)malloc((size_t)n * sizeof(*dft_im));
    if (!in || !back || !spec || !xref || !dft_re || !dft_im) {
        fail("out of memory");
        goto done;
    }

    make_signal(xref, n);
    for (k = 0; k < n; ++k)
        in[k] = (kiss_fft_scalar)xref[k];

    if (kiss_rfft(cfg, in, spec, &e) != KISS_RFFT_OK) {
        fail("kiss_rfft returned an error on a valid input");
        goto done;
    }

    /* 2. analytic DFT comparison */
    naive_dft(xref, n, dft_re, dft_im);
    for (k = 0; k < n; ++k) {
        double dr = fabs((double)spec[k].r - dft_re[k]);
        double di = fabs((double)spec[k].i - dft_im[k]);
        double d = fmax(dr, di);
        if (d > worst_dft)
            worst_dft = d;
    }

    /* 4. exact Hermitian symmetry of the emitted spectrum */
    for (k = 1; k < n; ++k) {
        if (spec[n - k].r != spec[k].r || spec[n - k].i != -spec[k].i) {
            fail("forward output is not exactly Hermitian symmetric");
            printf("       n=%d pair (%d,%d)\n", n, k, n - k);
            break;
        }
    }
    if (spec[0].i != (kiss_fft_scalar)0)
        fail("DC bin has a nonzero imaginary part");
    if ((n & 1) == 0 && spec[n / 2].i != (kiss_fft_scalar)0)
        fail("Nyquist bin has a nonzero imaginary part");

    /* 3. round trip */
    if (kiss_irfft(cfg, spec, back, -1.0, -1.0, &e) != KISS_RFFT_OK) {
        fail("kiss_irfft rejected a symmetric spectrum");
        goto done;
    }
    for (k = 0; k < n; ++k) {
        double d = fabs((double)back[k] - xref[k]);
        if (d > worst_rt)
            worst_rt = d;
    }
    if (worst_dft > DFT_TOL) {
        fail("forward transform differs from the analytic DFT beyond tolerance");
        printf("       n=%d worst=%g tol=%g\n", n, worst_dft, DFT_TOL);
    }
    if (worst_rt > ROUNDTRIP_TOL) {
        fail("round trip error beyond tolerance");
        printf("       n=%d worst=%g tol=%g\n", n, worst_rt, ROUNDTRIP_TOL);
    }

done:
    free(in);
    free(back);
    free(spec);
    free(xref);
    free(dft_re);
    free(dft_im);
    kiss_rfft_free(cfg);
}

/* ------------------------------------------------------------------ */
/* 5-6. asymmetry / non-finite rejection with exact index reporting    */
/* ------------------------------------------------------------------ */

static void test_rejection(void)
{
    const int n = 30; /* even, mixed radix 2*3*5 */
    kiss_rfft_cfg cfg;
    kiss_rfft_error e;
    kiss_fft_scalar *in, *out;
    kiss_fft_cpx *spec, *spec_copy;
    int k;

    cfg = kiss_rfft_alloc(n, NULL, NULL, &e);
    in = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*in));
    out = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*out));
    spec = (kiss_fft_cpx *)malloc((size_t)n * sizeof(*spec));
    spec_copy = (kiss_fft_cpx *)malloc((size_t)n * sizeof(*spec_copy));
    if (!cfg || !in || !out || !spec || !spec_copy) {
        fail("allocation failed in rejection test");
        goto done;
    }
    for (k = 0; k < n; ++k)
        in[k] = (kiss_fft_scalar)cos(2.0 * M_PI * k / n);
    if (kiss_rfft(cfg, in, spec, &e) != KISS_RFFT_OK) {
        fail("setup forward transform failed");
        goto done;
    }
    memcpy(spec_copy, spec, (size_t)n * sizeof(*spec));

    /* corrupt a single mirror pair: bin 7 must disagree with bin n-7 */
    spec[7].i = (kiss_fft_scalar)(spec[7].i + 0.5);
    memset(out, 0x5A, (size_t)n * sizeof(*out));
    if (kiss_irfft(cfg, spec, out, -1.0, -1.0, &e) != KISS_RFFT_ESYMMETRY)
        fail("asymmetric spectrum was not rejected");
    else if (e.n_violations != 1 || e.index[0] != 7 ||
             e.mirror[0] != n - 7) {
        fail("violating index pair reported incorrectly");
        printf("       got (%d,%d), want (7,%d)\n",
               e.n_violations ? e.index[0] : -1,
               e.n_violations ? e.mirror[0] : -1, n - 7);
    }
    /* output buffer must remain untouched on failure */
    for (k = 0; k < n; ++k) {
        unsigned char *p = (unsigned char *)(out + k);
        size_t b;
        for (b = 0; b < sizeof(*out); ++b)
            if (p[b] != 0x5A) {
                fail("kiss_irfft wrote output after a symmetry failure");
                goto restore1;
            }
    }
restore1:
    memcpy(spec, spec_copy, (size_t)n * sizeof(*spec));

    /* DC self-conjugate violation: pair (0,0) */
    spec[0].i = (kiss_fft_scalar)1.25;
    if (kiss_irfft(cfg, spec, out, -1.0, -1.0, &e) != KISS_RFFT_ESYMMETRY ||
        e.n_violations != 1 || e.index[0] != 0 || e.mirror[0] != 0)
        fail("DC imaginary violation not reported as pair (0,0)");
    memcpy(spec, spec_copy, (size_t)n * sizeof(*spec));

    /* Nyquist self-conjugate violation: pair (n/2,n/2) */
    spec[n / 2].i = (kiss_fft_scalar)-0.75;
    if (kiss_irfft(cfg, spec, out, -1.0, -1.0, &e) != KISS_RFFT_ESYMMETRY ||
        e.n_violations != 1 || e.index[0] != n / 2 ||
        e.mirror[0] != n / 2)
        fail("Nyquist imaginary violation not reported as pair (n/2,n/2)");
    memcpy(spec, spec_copy, (size_t)n * sizeof(*spec));

    /* several corruptions at once: every distinct pair must be listed */
    spec[3].r = (kiss_fft_scalar)(spec[3].r + 1.0);
    spec[11].i = (kiss_fft_scalar)(spec[11].i - 1.0);
    spec[0].i = (kiss_fft_scalar)0.4;
    if (kiss_irfft(cfg, spec, out, -1.0, -1.0, &e) != KISS_RFFT_ESYMMETRY) {
        fail("multi-violation spectrum was not rejected");
    } else {
        int seen3 = 0, seen11 = 0, seen0 = 0, v;
        for (v = 0; v < e.n_violations; ++v) {
            seen3 |= (e.index[v] == 3 && e.mirror[v] == n - 3);
            seen11 |= (e.index[v] == 11 && e.mirror[v] == n - 11);
            seen0 |= (e.index[v] == 0 && e.mirror[v] == 0);
        }
        if (!(seen3 && seen11 && seen0) || e.n_violations != 3) {
            fail("not every violating pair was reported");
            printf("       n_violations=%d\n", e.n_violations);
        }
    }
    memcpy(spec, spec_copy, (size_t)n * sizeof(*spec));

    /* standalone checker agrees, then accepts the restored spectrum */
    if (kiss_rfft_check_hermitian(n, spec, -1.0, -1.0, &e) != KISS_RFFT_OK)
        fail("standalone Hermitian check rejected a valid spectrum");
    if (kiss_rfft_check_hermitian(n, spec_copy, SYMM_TOL, 0.0, &e) !=
        KISS_RFFT_OK)
        fail("absolute-tolerance Hermitian check rejected a valid spectrum");

    /* 6. non-finite values */
    spec[5].r = (kiss_fft_scalar)NAN;
    if (kiss_irfft(cfg, spec, out, -1.0, -1.0, &e) != KISS_RFFT_EVALUE)
        fail("NaN spectrum was not rejected with KISS_RFFT_EVALUE");
    spec[5].r = (kiss_fft_scalar)INFINITY;
    if (kiss_rfft_check_hermitian(n, spec, -1.0, -1.0, &e) !=
        KISS_RFFT_EVALUE)
        fail("Inf spectrum was not rejected with KISS_RFFT_EVALUE");

done:
    free(in);
    free(out);
    free(spec);
    free(spec_copy);
    kiss_rfft_free(cfg);
}

/* ------------------------------------------------------------------ */
/* 7. timing: 10000 rounds of 1024-point transforms                    */
/* ------------------------------------------------------------------ */

static void test_performance(void)
{
    const int n = 1024;
    const int rounds = 10000;
    const double budget = 0.200; /* 200 ms */
    kiss_rfft_cfg cfg;
    kiss_rfft_error e;
    kiss_fft_scalar *in, *out;
    kiss_fft_cpx *spec;
    int k, r;
    double t0, t_fwd, t_pair;

    cfg = kiss_rfft_alloc(n, NULL, NULL, &e);
    in = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*in));
    out = (kiss_fft_scalar *)malloc((size_t)n * sizeof(*out));
    spec = (kiss_fft_cpx *)malloc((size_t)n * sizeof(*spec));
    if (!cfg || !in || !out || !spec) {
        fail("allocation failed in performance test");
        goto done;
    }
    for (k = 0; k < n; ++k)
        in[k] = (kiss_fft_scalar)sin(2.0 * M_PI * k / n);

    /* warm up */
    for (r = 0; r < 100; ++r) {
        kiss_rfft(cfg, in, spec, &e);
        kiss_irfft(cfg, spec, out, -1.0, -1.0, &e);
    }

    t0 = now_sec();
    for (r = 0; r < rounds; ++r)
        kiss_rfft(cfg, in, spec, &e);
    t_fwd = now_sec() - t0;

    t0 = now_sec();
    for (r = 0; r < rounds; ++r) {
        kiss_rfft(cfg, in, spec, &e);
        kiss_irfft(cfg, spec, out, -1.0, -1.0, &e);
    }
    t_pair = now_sec() - t0;

    printf("timing: %d x %d-point rfft = %.3f ms; "
           "rfft+irfft rounds = %.3f ms (budget %.0f ms)\n",
           rounds, n, 1e3 * t_fwd, 1e3 * t_pair, 1e3 * budget);

    if (t_fwd > budget) {
        fail("10000 forward 1024-point transforms exceeded 200 ms");
    }

done:
    free(in);
    free(out);
    free(spec);
    kiss_rfft_free(cfg);
}

int main(void)
{
    static const int sizes[] = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16,
                                25, 27, 30, 32, 75, 100, 125, 250, 256,
                                375, 1000, 1024};
    unsigned i;

    printf("== allocation validation ==\n");
    test_alloc_sizes();

    printf("== analytic DFT / round trip / symmetry (%s) ==\n",
           IS_DOUBLE ? "double" : "float");
    for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i)
        test_transforms(sizes[i]);

    printf("== asymmetry and non-finite rejection ==\n");
    test_rejection();

    printf("== performance ==\n");
    test_performance();

    printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
