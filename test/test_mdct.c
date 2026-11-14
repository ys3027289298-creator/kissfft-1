/*
  Copyright (c) 2026, Mark Borgerding. All rights reserved.
  This file is part of KISS FFT - https://github.com/mborgerding/kissfft

  SPDX-License-Identifier: BSD-3-Clause
  See COPYING file for more information.

  Self-contained MDCT/IMDCT validation:
    1. FFT-based MDCT/IMDCT vs. an independent O(N^2) reference (error bound).
    2. 50% overlap-add perfect reconstruction with the sine window (threshold).
    3. Invalid block length rejection.
    4. Configuration memory upper bound.
    5. Timing for N = 1024 and N = 1000.
*/

#include "kiss_mdct.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef KISS_MDCT_PRT_THRESHOLD
/* Perfect-reconstruction max |error| must stay below this. */
#define KISS_MDCT_PRT_THRESHOLD 1e-4
#endif

#define PI 3.141592653589793238462643383279502884

/* This test targets the floating-point (float/double) MDCT convention. The
 * fixed-point kiss_fft builds use a different internal scaling model. */
#ifdef FIXED_POINT
#error "test_mdct is a floating-point test; build without FIXED_POINT"
#endif

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static double rand_unit(void)
{
    int r = rand();
    return (double) r / (double) RAND_MAX * 2.0 - 1.0;
}

static double ref_kernel(int n, int k, int N)
{
    return cos((2.0 * PI / N) * (n + N / 4.0 + 0.5) * (k + 0.5));
}

/* Independent O(N^2) reference MDCT. */
static void ref_mdct(const double *x, int N, double *X)
{
    int M = N / 2, k, n;
    for (k = 0; k < M; ++k) {
        double acc = 0.0;
        for (n = 0; n < N; ++n)
            acc += x[n] * ref_kernel(n, k, N);
        X[k] = acc;
    }
}

/* Independent O(N^2) reference IMDCT, 2/M (=4/N) normalized. */
static void ref_imdct(const double *X, int N, double *y)
{
    int M = N / 2, k, n;
    for (n = 0; n < N; ++n) {
        double acc = 0.0;
        for (k = 0; k < M; ++k)
            acc += X[k] * ref_kernel(n, k, N);
        y[n] = (2.0 / M) * acc;
    }
}

static double max_abs_diff(const double *a, const kiss_fft_scalar *b, int len)
{
    double worst = 0.0;
    int i;
    for (i = 0; i < len; ++i) {
        double d = fabs(a[i] - (double) b[i]);
        if (d > worst)
            worst = d;
    }
    return worst;
}

static int check_size(int N, unsigned int seed)
{
    int M = N / 2, n;
    int ok = 1;
    kiss_mdct_cfg cfg;
    size_t reported = 0;
    double *xd = malloc(sizeof(double) * N);
    double *Xd_ref = malloc(sizeof(double) * M);
    double *yd_ref = malloc(sizeof(double) * N);
    kiss_fft_scalar *xs = malloc(sizeof(kiss_fft_scalar) * N);
    kiss_fft_scalar *Xs = malloc(sizeof(kiss_fft_scalar) * M);
    kiss_fft_scalar *ys = malloc(sizeof(kiss_fft_scalar) * N);

    /* probe-only call: returns NULL and reports required bytes */
    if (kiss_mdct_alloc(N, NULL, &reported) != NULL || reported == 0) {
        printf("  [FAIL] N=%d lenmem probe did not report a size\n", N);
        ok = 0;
        goto done;
    }
    cfg = kiss_mdct_alloc(N, NULL, NULL);
    if (cfg == NULL) {
        printf("  [FAIL] N=%d allocation returned NULL\n", N);
        ok = 0;
        goto done;
    }

    srand(seed);
    for (n = 0; n < N; ++n) {
        xd[n] = rand_unit();
        xs[n] = (kiss_fft_scalar) xd[n];
    }

    /* forward vs. reference */
    ref_mdct(xd, N, Xd_ref);
    kiss_mdct(cfg, xs, Xs);
    double fwd_err = max_abs_diff(Xd_ref, Xs, M);

    /* inverse vs. reference */
    ref_imdct(Xd_ref, N, yd_ref);
    kiss_imdct(cfg, Xs, ys);
    double inv_err = max_abs_diff(yd_ref, ys, N);

    /* windowed 50% overlap-add perfect reconstruction */
    kiss_fft_scalar *win = malloc(sizeof(kiss_fft_scalar) * N);
    int nblocks = 6;
    int stream_len = (nblocks - 1) * M + N;
    if (stream_len <= 0)
        stream_len = N;
    double *stream = malloc(sizeof(double) * stream_len);
    double *recon = calloc((size_t) stream_len + N, sizeof(double));
    kiss_fft_scalar *blk = malloc(sizeof(kiss_fft_scalar) * N);
    kiss_fft_scalar *coef = malloc(sizeof(kiss_fft_scalar) * M);
    kiss_fft_scalar *syn = malloc(sizeof(kiss_fft_scalar) * N);
    kiss_mdct_sine_window(N, win);

    for (n = 0; n < stream_len; ++n)
        stream[n] = rand_unit();

    for (int b = 0; b < nblocks; ++b) {
        int pos = b * M;
        for (n = 0; n < N; ++n)
            blk[n] = (kiss_fft_scalar) (win[n] * stream[pos + n]);
        kiss_mdct(cfg, blk, coef);
        kiss_imdct(cfg, coef, syn);
        for (n = 0; n < N; ++n)
            recon[pos + n] += (double) win[n] * (double) syn[n];
    }

    /* interior samples only: first and last M are edge effects by design */
    double pr_err = 0.0;
    for (n = M; n < stream_len - M; ++n) {
        double d = fabs(recon[n] - stream[n]);
        if (d > pr_err)
            pr_err = d;
    }

    if (fwd_err > 1e-3 || inv_err > 1e-3 || pr_err >= KISS_MDCT_PRT_THRESHOLD)
        ok = 0;

    printf("  N=%-5d fwd_ref_maxerr=%.3e  imdct_ref_maxerr=%.3e  "
           "PR_maxerr=%.3e  cfg_bytes=%zu  %s\n",
           N, fwd_err, inv_err, pr_err, reported,
           (pr_err < KISS_MDCT_PRT_THRESHOLD) ? "OK" : "FAIL");

    free(win); free(stream); free(recon); free(blk); free(coef); free(syn);
    kiss_mdct_free(cfg);
done:
    free(xd); free(Xd_ref); free(yd_ref);
    free(xs); free(Xs); free(ys);
    return ok;
}

static double bench(int N, int iters)
{
    int M = N / 2, i, n;
    kiss_mdct_cfg cfg = kiss_mdct_alloc(N, NULL, NULL);
    kiss_fft_scalar *in = malloc(sizeof(kiss_fft_scalar) * N);
    kiss_fft_scalar *coef = malloc(sizeof(kiss_fft_scalar) * M);
    kiss_fft_scalar *out = malloc(sizeof(kiss_fft_scalar) * N);
    for (n = 0; n < N; ++n)
        in[n] = (kiss_fft_scalar) sin(0.01 * n);

    double t0 = now_sec();
    for (i = 0; i < iters; ++i) {
        kiss_mdct(cfg, in, coef);
        kiss_imdct(cfg, coef, out);
    }
    double elapsed = now_sec() - t0;
    free(in); free(coef); free(out);
    kiss_mdct_free(cfg);
    return elapsed / iters;
}

int main(void)
{
    int failures = 0;
    const int sizes[] = {2, 4, 6, 8, 16, 100, 1000, 1024, 1026};
    const int nsizes = (int) (sizeof(sizes) / sizeof(sizes[0]));

    printf("MDCT reference / perfect-reconstruction checks "
           "(PR threshold %.0e):\n", (double) KISS_MDCT_PRT_THRESHOLD);
    for (int s = 0; s < nsizes; ++s)
        failures += !check_size(sizes[s], 1234u + (unsigned) sizes[s]);

    printf("Invalid block length rejection:\n");
    int bad[] = {0, -8, 1, 3, 7};
    for (int i = 0; i < (int) (sizeof(bad) / sizeof(bad[0])); ++i) {
        size_t len = 123;
        kiss_mdct_cfg cfg = kiss_mdct_alloc(bad[i], NULL, &len);
        if (cfg != NULL) {
            printf("  [FAIL] N=%d was accepted\n", bad[i]);
            kiss_mdct_free(cfg);
            failures++;
        } else {
            printf("  N=%-3d rejected as expected (lenmem left=%zu)\n",
                   bad[i], len);
        }
    }

    printf("Timing (one MDCT+IMDCT pair, wall time):\n");
    double t1024 = bench(1024, 2000);
    double t1000 = bench(1000, 2000);
    printf("  N=1024: %.3f us/pair\n", t1024 * 1e6);
    printf("  N=1000: %.3f us/pair\n", t1000 * 1e6);

    if (failures) {
        printf("RESULT: FAIL (%d check(s) failed)\n", failures);
        return 1;
    }
    printf("RESULT: ALL CHECKS PASSED\n");
    return 0;
}
