/*
 *  Validation for tools/kiss_conv.c (FFT-based linear convolution and
 *  autocorrelation).  Compares against the direct O(n^2) method using a
 *  double-precision reference, checks the documented error bound, times
 *  both methods, and exercises the failure paths.
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 */

#include "kiss_conv.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int nfail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (line %d)\n", msg, __LINE__); \
        ++nfail; \
    } \
} while (0)

/* deterministic pseudo-random sequence in [-1, 1) */
static double prng(unsigned *state)
{
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 8) / 8388608.0 - 1.0;
}

static void fill_complex(kiss_fft_cpx *x, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        x[i].r = (kiss_fft_scalar)prng(&seed);
        x[i].i = (kiss_fft_scalar)prng(&seed);
    }
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double scalar_eps(void)
{
    return sizeof(kiss_fft_scalar) == sizeof(double) ? DBL_EPSILON : FLT_EPSILON;
}

/* exact-ish reference: direct convolution accumulated in double */
static void conv_direct_d(const kiss_fft_cpx *x, size_t nx,
                          const kiss_fft_cpx *h, size_t nh,
                          double *yr, double *yi)
{
    size_t i, j;
    for (i = 0; i < nx + nh - 1; ++i)
        yr[i] = yi[i] = 0.0;
    for (i = 0; i < nx; ++i)
        for (j = 0; j < nh; ++j) {
            yr[i + j] += (double)x[i].r * h[j].r - (double)x[i].i * h[j].i;
            yi[i + j] += (double)x[i].r * h[j].i + (double)x[i].i * h[j].r;
        }
}

/* what users do today: direct convolution in kiss_fft_scalar precision */
static void conv_direct_s(const kiss_fft_cpx *x, size_t nx,
                          const kiss_fft_cpx *h, size_t nh,
                          kiss_fft_cpx *y)
{
    size_t i, j;
    for (i = 0; i < nx + nh - 1; ++i)
        y[i].r = y[i].i = 0;
    for (i = 0; i < nx; ++i)
        for (j = 0; j < nh; ++j) {
            y[i + j].r += x[i].r * h[j].r - x[i].i * h[j].i;
            y[i + j].i += x[i].r * h[j].i + x[i].i * h[j].r;
        }
}

static double max_err_vs_ref(const kiss_fft_cpx *y, const double *yr,
                             const double *yi, size_t n)
{
    double m = 0.0;
    size_t i;
    for (i = 0; i < n; ++i) {
        double e = hypot((double)y[i].r - yr[i], (double)y[i].i - yi[i]);
        if (e > m)
            m = e;
    }
    return m;
}

static double norm2(const kiss_fft_cpx *x, size_t n)
{
    double s = 0.0;
    size_t i;
    for (i = 0; i < n; ++i)
        s += (double)x[i].r * x[i].r + (double)x[i].i * x[i].i;
    return sqrt(s);
}

static double conv_err_bound(const kiss_fft_cpx *x, size_t nx,
                             const kiss_fft_cpx *h, size_t nh)
{
    size_t nfft = (size_t)kiss_fft_next_fast_size((int)(nx + nh - 1));
    return scalar_eps() * (3.0 * log2((double)nfft) + 5.0)
         * norm2(x, nx) * norm2(h, nh);
}

static void test_conv_case(size_t nx, size_t nh, int timed)
{
    size_t nconv = nx + nh - 1;
    kiss_fft_cpx *x = malloc(sizeof(*x) * nx);
    kiss_fft_cpx *h = malloc(sizeof(*h) * nh);
    kiss_fft_cpx *y = malloc(sizeof(*y) * nconv);
    kiss_fft_cpx *yd = malloc(sizeof(*yd) * nconv);
    double *yr = malloc(sizeof(*yr) * nconv);
    double *yi = malloc(sizeof(*yi) * nconv);
    double err_fft, err_direct, bound, t0, t1, t_fft, t_direct;
    int iter_fft, iter_direct, it;

    fill_complex(x, nx, 12345u);
    fill_complex(h, nh, 67890u);
    conv_direct_d(x, nx, h, nh, yr, yi);

    CHECK(kiss_conv_linear(x, nx, h, nh, y) == KISS_CONV_OK, "conv status");
    err_fft = max_err_vs_ref(y, yr, yi, nconv);
    bound = conv_err_bound(x, nx, h, nh);
    CHECK(err_fft <= bound, "conv error within bound");

    conv_direct_s(x, nx, h, nh, yd);
    err_direct = max_err_vs_ref(yd, yr, yi, nconv);

    if (timed) {
        iter_fft = 200;
        iter_direct = (int)(200000000.0 / ((double)nx * nh)) + 1;
        t0 = now_sec();
        for (it = 0; it < iter_fft; ++it)
            kiss_conv_linear(x, nx, h, nh, y);
        t1 = now_sec();
        t_fft = (t1 - t0) / iter_fft;
        t0 = now_sec();
        for (it = 0; it < iter_direct; ++it)
            conv_direct_s(x, nx, h, nh, yd);
        t1 = now_sec();
        t_direct = (t1 - t0) / iter_direct;
        printf("conv n=%zu x %zu: max|err| fft=%.3e direct=%.3e bound=%.3e | "
               "time/call fft=%.3f ms direct=%.3f ms (%.1fx speedup)\n",
               nx, nh, err_fft, err_direct, bound,
               t_fft * 1e3, t_direct * 1e3, t_direct / t_fft);
    } else {
        printf("conv n=%zu x %zu: max|err| fft=%.3e direct=%.3e bound=%.3e %s\n",
               nx, nh, err_fft, err_direct, bound,
               err_fft <= bound ? "OK" : "BOUND-VIOLATED");
    }

    free(x); free(h); free(y); free(yd); free(yr); free(yi);
}

static void test_autocorr_case(size_t nx)
{
    size_t nr = 2 * nx - 1;
    kiss_fft_cpx *x = malloc(sizeof(*x) * nx);
    kiss_fft_cpx *r = malloc(sizeof(*r) * nr);
    double err = 0.0, bound;
    size_t k, i;

    fill_complex(x, nx, 24680u);
    CHECK(kiss_conv_autocorr(x, nx, r) == KISS_CONV_OK, "autocorr status");

    for (k = 0; k < nr; ++k) {
        long lag = (long)k - ((long)nx - 1);
        double er = 0.0, ei = 0.0, e;
        for (i = 0; i < nx; ++i) {
            long j = (long)i + lag;
            if (j >= 0 && j < (long)nx) {
                er += (double)x[j].r * x[i].r + (double)x[j].i * x[i].i;
                ei += (double)x[i].r * x[j].i - (double)x[i].i * x[j].r;
            }
        }
        e = hypot((double)r[k].r - er, (double)r[k].i - ei);
        if (e > err)
            err = e;
    }
    bound = conv_err_bound(x, nx, x, nx);
    CHECK(err <= bound, "autocorr error within bound");
    printf("autocorr n=%zu: max|err|=%.3e bound=%.3e %s\n",
           nx, err, bound, err <= bound ? "OK" : "BOUND-VIOLATED");

    free(x); free(r);
}

static void test_failure_paths(void)
{
    kiss_fft_cpx a[4] = {{1, 0}}, b[4] = {{1, 0}}, y[8];

    CHECK(kiss_conv_linear(a, 0, b, 4, y) == KISS_CONV_ERR_LENGTH,
          "conv nx=0 rejected");
    CHECK(kiss_conv_linear(a, 4, b, 0, y) == KISS_CONV_ERR_LENGTH,
          "conv nh=0 rejected");
    CHECK(kiss_conv_linear(NULL, 4, b, 4, y) == KISS_CONV_ERR_NULL,
          "conv NULL x rejected");
    CHECK(kiss_conv_linear(a, 4, b, 4, NULL) == KISS_CONV_ERR_NULL,
          "conv NULL y rejected");
    CHECK(kiss_conv_linear(a, (size_t)-1 / 2, b, (size_t)-1 / 2, y)
              == KISS_CONV_ERR_LENGTH,
          "conv length overflow rejected");
    CHECK(kiss_conv_linear(a, (size_t)INT_MAX + 1, b, 1, y)
              == KISS_CONV_ERR_LENGTH,
          "conv nfft > INT_MAX rejected");
    CHECK(kiss_conv_linear_len(0, 5) == 0, "linear_len(0,5)==0");
    CHECK(kiss_conv_linear_len((size_t)-1, 2) == 0, "linear_len overflow==0");

    CHECK(kiss_conv_autocorr(a, 0, y) == KISS_CONV_ERR_LENGTH,
          "autocorr n=0 rejected");
    CHECK(kiss_conv_autocorr(NULL, 4, y) == KISS_CONV_ERR_NULL,
          "autocorr NULL x rejected");
    CHECK(kiss_conv_autocorr(a, 4, NULL) == KISS_CONV_ERR_NULL,
          "autocorr NULL r rejected");
    CHECK(kiss_conv_autocorr(a, (size_t)-1, y) == KISS_CONV_ERR_LENGTH,
          "autocorr length overflow rejected");
    CHECK(kiss_conv_autocorr_len(0) == 0, "autocorr_len(0)==0");

    printf("failure paths: %s\n", nfail == 0 ? "all rejected as expected"
                                            : "SOME NOT REJECTED");
}

int main(void)
{
    printf("kiss_fft_scalar size: %zu byte(s), eps=%.3e\n",
           sizeof(kiss_fft_scalar), scalar_eps());

    /* correctness on small / odd sizes */
    test_conv_case(1, 1, 0);
    test_conv_case(3, 5, 0);
    test_conv_case(100, 1, 0);
    test_conv_case(37, 64, 0);
    test_autocorr_case(1);
    test_autocorr_case(7);

    /* error + timing comparison on the requested sizes */
    test_conv_case(1000, 1000, 1);
    test_conv_case(4096, 4096, 1);
    test_autocorr_case(1000);
    test_autocorr_case(4096);

    test_failure_paths();

    if (nfail == 0)
        printf("test_conv: ALL PASSED\n");
    else
        printf("test_conv: %d FAILURE(S)\n", nfail);
    return nfail ? 1 : 0;
}
