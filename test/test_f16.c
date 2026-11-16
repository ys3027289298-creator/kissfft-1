/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * One-shot validation of the binary16 input/output layer (kiss_fft_f16.h).
 *
 * Sections:
 *   [1] converter correctness (exhaustive over half patterns, random f32)
 *   [2] failure paths (bad lengths, bad pointers, overflow -> ERANGE)
 *   [3] numeric error at 1024/4096 against a double-precision naive DFT,
 *       plus float32 bit identity and determinism
 *   [4] timing: float32 vs float16 (conversions included)
 *   [5] memory footprint: mem_bytes() vs measured malloc, and the assertion
 *       that a transform performs no allocation
 *
 * Build (portable, no half-precision hardware or linker support needed):
 *   gcc -O2 -Wall -Wextra -std=c11 -I. test/test_f16.c \
 *       kiss_fft_f16.c kiss_fft.c -lm -o test/test_f16
 *
 * With heap accounting through the linker:
 *   gcc -O2 -Wall -Wextra -std=c11 -DTEST_F16_WRAP_MALLOC -I. \
 *       test/test_f16.c kiss_fft_f16.c kiss_fft.c -lm \
 *       -Wl,--wrap=malloc -Wl,--wrap=free -o test/test_f16
 */

#define _POSIX_C_SOURCE 199309L

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kiss_fft.h"
#include "kiss_fft_f16.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------- */
/* check bookkeeping                                                         */
/* ------------------------------------------------------------------------- */

static long checks = 0;
static long failures = 0;

static void check(int cond, const char *what, int a, int b)
{
    ++checks;
    if (!cond) {
        ++failures;
        printf("    FAIL: %s (%d, %d)\n", what, a, b);
    }
}

static void check_d(int cond, const char *what, double a, double b)
{
    ++checks;
    if (!cond) {
        ++failures;
        printf("    FAIL: %s (%.17g, %.17g)\n", what, a, b);
    }
}

/* ------------------------------------------------------------------------- */
/* optional malloc/free accounting (-Wl,--wrap=malloc -Wl,--wrap=free)       */
/* ------------------------------------------------------------------------- */

#ifdef TEST_F16_WRAP_MALLOC
void *__real_malloc(size_t n);
void __real_free(void *p);

/* 16-byte header keeps the returned pointer suitably aligned */
typedef union {
    size_t n;
    long double align_ld;
    void *align_p;
    long long align_ll;
} alloc_hdr;

static size_t live_bytes = 0;
static long live_allocs = 0;
static long total_allocs = 0;
static size_t peak_bytes = 0;
static long allocs_since_mark = 0;
static size_t bytes_since_mark = 0;

void *__wrap_malloc(size_t n)
{
    alloc_hdr *h = (alloc_hdr *)__real_malloc(sizeof(alloc_hdr) + n);
    if (h == NULL)
        return NULL;
    h->n = n;
    live_bytes += n;
    ++live_allocs;
    ++total_allocs;
    ++allocs_since_mark;
    bytes_since_mark += n;
    if (live_bytes > peak_bytes)
        peak_bytes = live_bytes;
    return (void *)(h + 1);
}

void __wrap_free(void *p)
{
    alloc_hdr *h;
    if (p == NULL)
        return;
    h = ((alloc_hdr *)p) - 1;
    live_bytes -= h->n;
    --live_allocs;
    __real_free(h);
}
# define WRAP_MARK() (allocs_since_mark = 0, bytes_since_mark = 0)
# define WRAP_ALLOCS_SINCE() (allocs_since_mark)
#else
# define WRAP_MARK() ((void)0)
# define WRAP_ALLOCS_SINCE() 0
#endif

/* ------------------------------------------------------------------------- */
/* independent reference conversions                                         */
/* ------------------------------------------------------------------------- */

/* binary16 bits -> exact value as long double (NaN preserved as NaN) */
static long double ref_half_to_value(uint16_t h)
{
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1F;
    int m = h & 0x3FF;
    long double v;

    if (e == 0x1F)
        return m ? (long double)NAN : (s ? -(long double)INFINITY : (long double)INFINITY);
    if (e == 0)
        v = ldexpl((long double)m, -24);
    else
        v = ldexpl((long double)(m | 0x400), e - 25);
    return s ? -v : v;
}

/* value -> binary16 bits, round-to-nearest-even, via long double and rintl */
static int ref_value_to_half(long double v, uint16_t *out)
{
    int s = signbit(v) ? 1 : 0;
    long double a = fabsl(v);
    int e16;

    if (isnan(v)) {
        *out = 0x7E00;
        return 0;
    }
    if (isinf(a)) {
        *out = (uint16_t)((s << 15) | 0x7C00u);
        return 0;
    }
    if (a == 0) {
        *out = (uint16_t)(s << 15);
        return 0;
    }
    if (a >= 65520.0L) { /* halfway between 65504 and 65536 */
        *out = (uint16_t)((s << 15) | 0x7C00u);
        return 1;
    }
    frexpl(a, &e16);
    --e16; /* a in [2^e16, 2^(e16+1)) */

    if (e16 < -14) {
        long q = lrintl(ldexpl(a, 24)); /* ties-to-even */
        if (q < 0)
            q = 0;
        if (q > 0x400) /* 0x400 is the smallest normal half (2^-14) */
            q = 0x400;
        *out = (uint16_t)((s << 15) | (uint16_t)q);
        return 0;
    }
    {
        long m = lrintl((ldexpl(a, -e16) - 1.0L) * 1024.0L);
        if (m == 1024) {
            m = 0;
            ++e16;
        }
        if (e16 > 15) {
            *out = (uint16_t)((s << 15) | 0x7C00u);
            return 1;
        }
        *out = (uint16_t)((s << 15) | ((uint16_t)(e16 + 15) << 10) | (uint16_t)m);
        return 0;
    }
}

/* ------------------------------------------------------------------------- */
/* [1] converters                                                            */
/* ------------------------------------------------------------------------- */

static void test_converters(void)
{
    long h;
    unsigned x = 0x12345678u;
    long i;

    printf("[1] converter correctness\n");

    /* every half pattern widens exactly; every finite one round-trips */
    for (h = 0; h < 65536; ++h) {
        uint16_t hh = (uint16_t)h;
        float f = kiss_fft_f16_to_f32(hh);
        long double ref = ref_half_to_value(hh);

        if (isnan(ref)) {
            uint32_t fbits;
            memcpy(&fbits, &f, sizeof fbits);
            check(isnan(f) != 0, "NaN widening", hh, 0);
            check((fbits & 0x00400000u) != 0u, "sNaN quieted", hh, 0);
            continue;
        }
        check_d((long double)f == ref, "widen exact", (double)f, (double)ref);
        if ((hh & 0x7FFFu) < 0x7C00u) { /* finite, either sign */
            int ov = 0;
            uint16_t back = kiss_fft_f32_to_f16(f, &ov);
            check(back == hh, "f16->f32->f16 identity", hh, back);
            check(ov == 0, "in-range narrowing does not overflow", hh, ov);
        }
    }

    /* rounding boundaries: exact tie cases on the normal grid */
    {
        static const struct { uint32_t bits; uint16_t want; } tie[] = {
            /* 1.0 + 2^-11 (exact halfway between 1.0 and the next half) */
            { 0x3F801000u, 0x3C00u }, /* ties to even -> 1.0 */
            /* 1.0 + 3*2^-11 -> ties to the even upper neighbour */
            { 0x3F803000u, 0x3C02u },
            /* 1.0 + 2^-11 + 1 -> rounds up */
            { 0x3F801001u, 0x3C01u },
            /* 1.0 + 2^-11 - 1 -> rounds down */
            { 0x3F800FFFu, 0x3C00u },
            /* just below 0.25 (0.25 - 2^-25): rounds up across the binade
             * boundary to 0.25 = 0x3400, not to its lower neighbour 0x33FF */
            { 0x3E7FFFFFu, 0x3400u },
            /* largest finite half */
            { 0x477FEFFFu, 0x7BFFu },
            /* 65520 exactly: halfway between 65504 and 65536, ties to the
             * even encoding -> infinity */
            { 0x477FF000u, 0x7C00u },
            /* just below 65520 */
            { 0x477FEFFFu, 0x7BFFu },
        };
        size_t k;
        for (k = 0; k < sizeof tie / sizeof tie[0]; ++k) {
            float f;
            uint16_t got;
            int ov = 0;
            memcpy(&f, &tie[k].bits, sizeof f);
            got = kiss_fft_f32_to_f16(f, &ov);
            check(got == tie[k].want, "tie/boundary bit pattern", got, tie[k].want);
        }
    }

    /* subnormal grid: value m * 2^-24 for every m, and the ties between */
    for (i = 0; i < 1024; ++i) {
        uint16_t got;
        int ov = 0;
        float f = (float)ldexpl((long double)i, -24);
        got = kiss_fft_f32_to_f16(f, &ov);
        check(got == (uint16_t)i, "subnormal grid exact", i, got);
        check(ov == 0, "subnormal does not overflow", i, ov);

        {
            /* halfway between i and i+1 (only representable in f32 if
             * (2i+1) has <= 24 significant bits; use 2^-25*k) */
            long double mid = ldexpl((long double)(2 * i + 1), -25);
            uint16_t r;
            int rov = ref_value_to_half(mid, &r);
            got = kiss_fft_f32_to_f16((float)mid, &ov);
            check(got == r, "subnormal tie matches reference", got, r);
            check(ov == rov, "subnormal tie overflow flag", ov, rov);
        }
    }

    /* signed zero, infinities, NaN */
    {
        uint16_t got;
        int ov = 0;
        uint32_t nbits = 0x7FC01234u;
        float nanf;
        memcpy(&nanf, &nbits, sizeof nanf);

        check(kiss_fft_f16_to_f32(0x0000u) == 0.0f, "widen +0", 0, 0);
        check(signbit(kiss_fft_f16_to_f32(0x8000u)) != 0, "widen -0 keeps sign", 0, 0);
        got = kiss_fft_f32_to_f16(0.0f, &ov);
        check(got == 0x0000u && ov == 0, "narrow +0", got, 0);
        got = kiss_fft_f32_to_f16(-0.0f, &ov);
        check(got == 0x8000u && ov == 0, "narrow -0", got, 0);
        got = kiss_fft_f32_to_f16(INFINITY, &ov);
        check(got == 0x7C00u && ov == 0, "narrow +inf is not an overflow", got, ov);
        got = kiss_fft_f32_to_f16(-INFINITY, &ov);
        check(got == 0xFC00u && ov == 0, "narrow -inf is not an overflow", got, ov);
        got = kiss_fft_f32_to_f16(nanf, &ov);
        check((got & 0x7C00u) == 0x7C00u && (got & 0x03FFu) != 0u, "narrow NaN", got, 0);
        check(ov == 0, "NaN is not an overflow", ov, 0);
    }

    /* random float32 patterns against the long-double reference */
    for (i = 0; i < 400000; ++i) {
        uint32_t bits;
        uint16_t got, want;
        int ov, wov;
        float f;

        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        bits = x;
        memcpy(&f, &bits, sizeof f);
        got = kiss_fft_f32_to_f16(f, &ov);
        wov = ref_value_to_half((long double)f, &want);
        if (isnan(f))
            continue;
        check(got == want, "random f32->f16 matches reference", got, want);
        check(ov == wov, "random overflow flag matches reference", ov, wov);
    }
}

/* ------------------------------------------------------------------------- */
/* [2] failure paths                                                         */
/* ------------------------------------------------------------------------- */

static void test_failures(void)
{
    static const int bad_nfft[] = { 0, -1, -8, INT_MAX, KISS_FFT_F16_MAX_NFFT + 1 };
    size_t k;
    kiss_fft_f16_cfg cfg;
    kiss_f16_cpx_t in[128];
    kiss_f16_cpx_t out[128];
    int err;

    printf("[2] failure paths\n");

    for (k = 0; k < sizeof bad_nfft / sizeof bad_nfft[0]; ++k) {
        WRAP_MARK();
        cfg = kiss_fft_f16_alloc(bad_nfft[k], 0);
        check(cfg == NULL, "invalid nfft rejected", bad_nfft[k], 0);
        check(kiss_fft_f16_mem_bytes(bad_nfft[k]) == 0, "invalid nfft mem_bytes is 0",
              bad_nfft[k], 0);
        check(WRAP_ALLOCS_SINCE() == 0, "invalid nfft allocates nothing", bad_nfft[k], 0);
    }
    check(kiss_fft_f16_alloc(KISS_FFT_F16_MAX_NFFT + 1, 0) == NULL,
          "MAX_NFFT+1 rejected", 0, 0);

    cfg = kiss_fft_f16_alloc(64, 0);
    check(cfg != NULL, "64-point alloc", 0, 0);
    if (cfg == NULL)
        return;

    for (k = 0; k < 64; ++k) {
        in[k].r = 0;
        in[k].i = 0;
    }

    check(kiss_fft_f16(cfg, NULL, out) == KISS_FFT_F16_EINVAL, "NULL fin -> EINVAL", 0, 0);
    check(kiss_fft_f16(cfg, in, NULL) == KISS_FFT_F16_EINVAL, "NULL fout -> EINVAL", 0, 0);
    check(kiss_fft_f16_strided(cfg, in, 0, out) == KISS_FFT_F16_EINVAL,
          "stride 0 -> EINVAL", 0, 0);
    check(kiss_fft_f16_strided(cfg, in, -1, out) == KISS_FFT_F16_EINVAL,
          "negative stride -> EINVAL", 0, 0);
    /* (Note: on a 64-bit platform no int stride can overflow the byte offset
     * for the accepted length range, so the size_t overflow guard in
     * kiss_fft_f16_strided() is not reachable from here; it only matters on
     * 32-bit targets.) */
    check(kiss_fft_f16(NULL, in, out) == KISS_FFT_F16_EINVAL, "NULL cfg -> EINVAL", 0, 0);
    kiss_fft_f16_free(cfg);
    kiss_fft_f16_free(NULL); /* accepted */

    /* DC overflow: 128 samples of 512.0 -> DC bin = 65536 -> half +inf */
    cfg = kiss_fft_f16_alloc(128, 0);
    check(cfg != NULL, "128-point alloc", 0, 0);
    if (cfg == NULL)
        return;
    for (k = 0; k < 128; ++k) {
        in[k].r = kiss_fft_f32_to_f16(512.0f, NULL);
        in[k].i = 0;
    }
    err = kiss_fft_f16(cfg, in, out);
    check(err == KISS_FFT_F16_ERANGE, "DC overflow -> ERANGE", err, KISS_FFT_F16_ERANGE);
    check(out[0].r == 0x7C00u, "overflowing DC bin is +inf", out[0].r, 0x7C00u);
    check(out[0].i == 0x0000u, "overflowing DC imaginary is zero", out[0].i, 0);
    kiss_fft_f16_free(cfg);

    /* an in-range signal must not report overflow */
    cfg = kiss_fft_f16_alloc(128, 0);
    if (cfg != NULL) {
        for (k = 0; k < 128; ++k) {
            in[k].r = kiss_fft_f32_to_f16(0.0f, NULL);
            in[k].i = 0;
        }
        in[3].r = kiss_fft_f32_to_f16(1.0f, NULL);
        check(kiss_fft_f16(cfg, in, out) == KISS_FFT_F16_OK,
              "in-range transform is OK", 0, 0);
        kiss_fft_f16_free(cfg);
    }
}

/* ------------------------------------------------------------------------- */
/* [3] numeric error                                                         */
/* ------------------------------------------------------------------------- */

/* deterministic two-tone complex test signal, stored as binary16 */
static void make_signal(int n, kiss_f16_cpx_t *wire, kiss_fft_cpx *exact)
{
    int j;
    for (j = 0; j < n; ++j) {
        double re = 0.5 * cos(2 * M_PI * 7 * j / n) + 0.3 * sin(2 * M_PI * 13 * j / n);
        double im = 0.5 * sin(2 * M_PI * 7 * j / n) - 0.3 * cos(2 * M_PI * 13 * j / n);
        wire[j].r = kiss_fft_f32_to_f16((float)re, NULL);
        wire[j].i = kiss_fft_f32_to_f16((float)im, NULL);
        /* the exact input is the widened binary16 value */
        exact[j].r = kiss_fft_f16_to_f32(wire[j].r);
        exact[j].i = kiss_fft_f16_to_f32(wire[j].i);
    }
}

/* naive O(n^2) double-precision DFT of the exact float32 input */
static void naive_dft(const kiss_fft_cpx *x, int n, double *re, double *im)
{
    int k, j;
    for (k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        for (j = 0; j < n; ++j) {
            double a = -2.0 * M_PI * (double)k * (double)j / (double)n;
            double c = cos(a), s = sin(a);
            sr += (double)x[j].r * c - (double)x[j].i * s;
            si += (double)x[j].r * s + (double)x[j].i * c;
        }
        re[k] = sr;
        im[k] = si;
    }
}

static double max_abs(double a, double b)
{
    return a > b ? a : b;
}

static void numeric_case(int n)
{
    kiss_f16_cpx_t *wire = (kiss_f16_cpx_t *)malloc(sizeof(kiss_f16_cpx_t) * (size_t)n);
    kiss_f16_cpx_t *spectrum = (kiss_f16_cpx_t *)malloc(sizeof(kiss_f16_cpx_t) * (size_t)n);
    kiss_f16_cpx_t *manual = (kiss_f16_cpx_t *)malloc(sizeof(kiss_f16_cpx_t) * (size_t)n);
    kiss_f16_cpx_t *roundtrip = (kiss_f16_cpx_t *)malloc(sizeof(kiss_f16_cpx_t) * (size_t)n);
    kiss_fft_cpx *exact = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
    kiss_fft_cpx *tmp = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
    kiss_fft_cpx *tmp2 = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
    double *ref_re = (double *)malloc(sizeof(double) * (size_t)n);
    double *ref_im = (double *)malloc(sizeof(double) * (size_t)n);
    double xmax = 0.0, max_err = 0.0, sumsq = 0.0, worst_roundtrip = 0.0;
    double bound;
    int k, err;
    kiss_fft_f16_cfg fwd, inv;
    kiss_fft_cfg eng;

    if (!wire || !spectrum || !manual || !roundtrip || !exact || !tmp || !tmp2 ||
        !ref_re || !ref_im) {
        printf("    FAIL: out of memory\n");
        ++failures;
        return;
    }

    make_signal(n, wire, exact);
    for (k = 0; k < n; ++k)
        xmax = max_abs(xmax, max_abs(fabs((double)exact[k].r), fabs((double)exact[k].i)));
    naive_dft(exact, n, ref_re, ref_im);

    fwd = kiss_fft_f16_alloc(n, 0);
    inv = kiss_fft_f16_alloc(n, 1);
    eng = kiss_fft_alloc(n, 0, NULL, NULL);
    check(fwd != NULL && inv != NULL && eng != NULL, "numeric setup", n, 0);
    if (!fwd || !inv || !eng)
        return;

    WRAP_MARK();
    err = kiss_fft_f16(fwd, wire, spectrum);
    check(WRAP_ALLOCS_SINCE() == 0, "transform allocates nothing", WRAP_ALLOCS_SINCE(), 0);
    check(err == KISS_FFT_F16_OK, "forward transform status", err, 0);

    /* bit identity with the manual widen -> engine -> narrow path */
    for (k = 0; k < n; ++k) {
        tmp[k].r = kiss_fft_f16_to_f32(wire[k].r);
        tmp[k].i = kiss_fft_f16_to_f32(wire[k].i);
    }
    kiss_fft(eng, tmp, tmp2);
    for (k = 0; k < n; ++k) {
        manual[k].r = kiss_fft_f32_to_f16(tmp2[k].r, NULL);
        manual[k].i = kiss_fft_f32_to_f16(tmp2[k].i, NULL);
    }
    check(memcmp(manual, spectrum, sizeof(kiss_f16_cpx_t) * (size_t)n) == 0,
          "f16 output is bit-identical to manual widen/engine/narrow", n, 0);
    check(kiss_fft_f16(fwd, wire, spectrum) == KISS_FFT_F16_OK,
          "second forward transform status", 0, 0);
    check(memcmp(manual, spectrum, sizeof(kiss_f16_cpx_t) * (size_t)n) == 0,
          "float32 engine is deterministic", n, 0);

    /* in-place (fin == fout, stride 1) is supported: the widened copy is taken
     * before anything is written */
    memcpy(roundtrip, wire, sizeof(kiss_f16_cpx_t) * (size_t)n);
    check(kiss_fft_f16(fwd, roundtrip, roundtrip) == KISS_FFT_F16_OK,
          "in-place transform status", n, 0);
    check(memcmp(roundtrip, spectrum, sizeof(kiss_f16_cpx_t) * (size_t)n) == 0,
          "in-place output matches out-of-place output", n, 0);

    /* error against the exact DFT of the widened input */
    for (k = 0; k < n; ++k) {
        double er = fabs((double)kiss_fft_f16_to_f32(spectrum[k].r) - ref_re[k]);
        double ei = fabs((double)kiss_fft_f16_to_f32(spectrum[k].i) - ref_im[k]);
        double e = max_abs(er, ei);
        max_err = max_abs(max_err, e);
        sumsq += er * er + ei * ei;
    }

    bound = kiss_fft_f16_error_bound(n, xmax);
    printf("    n=%d xmax=%.6f max_err=%.6e rms=%.6e bound=%.6e margin=%.1fx\n",
           n, xmax, max_err, sqrt(sumsq / (2.0 * (double)n)), bound, bound / max_err);
    check_d(max_err <= bound, "measured error is inside the analytic bound", max_err, bound);
    check_d(bound > 0.0, "bound is positive", bound, 0.0);
    check_d(kiss_fft_f16_error_bound(0, xmax) == 0.0, "bound of invalid nfft is 0", 0.0, 0.0);

    /* forward + inverse: the engine is unnormalised, so divide by n */
    err = kiss_fft_f16(inv, spectrum, roundtrip);
    check(err == KISS_FFT_F16_OK, "inverse transform status", err, 0);
    for (k = 0; k < n; ++k) {
        double dr = fabs((double)kiss_fft_f16_to_f32(roundtrip[k].r) / (double)n -
                         (double)exact[k].r);
        double di = fabs((double)kiss_fft_f16_to_f32(roundtrip[k].i) / (double)n -
                         (double)exact[k].i);
        double e = max_abs(dr, di);
        if (e > worst_roundtrip)
            worst_roundtrip = e;
    }
    /* xmax <= 0.8, so one half-ULP of 1.0 (2^-10) is a comfortable ceiling */
    printf("    inverse round trip worst per-sample error=%.6e\n", worst_roundtrip);
    check_d(worst_roundtrip <= ldexp(1.0, -10), "round trip within one half-ULP of 1.0",
            worst_roundtrip, ldexp(1.0, -10));

    kiss_fft_free(eng);
    kiss_fft_f16_free(inv);
    kiss_fft_f16_free(fwd);
    free(ref_im);
    free(ref_re);
    free(tmp2);
    free(tmp);
    free(exact);
    free(roundtrip);
    free(manual);
    free(spectrum);
    free(wire);
}

/* ------------------------------------------------------------------------- */
/* [4] timing                                                                */
/* ------------------------------------------------------------------------- */

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void timing_case(int n, int batches, int iters)
{
    kiss_f16_cpx_t *wire = (kiss_f16_cpx_t *)malloc(sizeof(kiss_f16_cpx_t) * (size_t)n);
    kiss_f16_cpx_t *out16 = (kiss_f16_cpx_t *)malloc(sizeof(kiss_f16_cpx_t) * (size_t)n);
    kiss_fft_cpx *in32 = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
    kiss_fft_cpx *out32 = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
    kiss_fft_cpx *exact = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
    double *t32 = (double *)malloc(sizeof(double) * (size_t)batches);
    double *t16 = (double *)malloc(sizeof(double) * (size_t)batches);
    kiss_fft_cfg eng;
    kiss_fft_f16_cfg cfg;
    int k, b;

    if (!wire || !out16 || !in32 || !out32 || !exact || !t32 || !t16) {
        printf("    FAIL: out of memory\n");
        ++failures;
        return;
    }
    make_signal(n, wire, exact);
    memcpy(in32, exact, sizeof(kiss_fft_cpx) * (size_t)n);

    eng = kiss_fft_alloc(n, 0, NULL, NULL);
    cfg = kiss_fft_f16_alloc(n, 0);
    check(eng != NULL && cfg != NULL, "timing setup", n, 0);
    if (!eng || !cfg)
        return;

    /* warm up both paths */
    for (k = 0; k < 100; ++k) {
        kiss_fft(eng, in32, out32);
        (void)kiss_fft_f16(cfg, wire, out16);
    }
    for (b = 0; b < batches; ++b) {
        double s = now_seconds();
        for (k = 0; k < iters; ++k)
            kiss_fft(eng, in32, out32);
        t32[b] = (now_seconds() - s) / (double)iters;
    }
    for (b = 0; b < batches; ++b) {
        double s = now_seconds();
        for (k = 0; k < iters; ++k)
            (void)kiss_fft_f16(cfg, wire, out16);
        t16[b] = (now_seconds() - s) / (double)iters;
    }
    qsort(t32, (size_t)batches, sizeof *t32, cmp_double);
    qsort(t16, (size_t)batches, sizeof *t16, cmp_double);

    printf("    n=%d  float32 min/median %6.3f/%6.3f us   "
           "float16 min/median %6.3f/%6.3f us   median ratio %.2fx\n",
           n, t32[0] * 1e6, t32[batches / 2] * 1e6,
           t16[0] * 1e6, t16[batches / 2] * 1e6,
           t16[batches / 2] / t32[batches / 2]);

    kiss_fft_f16_free(cfg);
    kiss_fft_free(eng);
    free(t16);
    free(t32);
    free(exact);
    free(out32);
    free(in32);
    free(out16);
    free(wire);
}

/* ------------------------------------------------------------------------- */
/* [5] memory                                                                */
/* ------------------------------------------------------------------------- */

static void test_memory(void)
{
    static const int sizes[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    size_t k;

    printf("[5] memory footprint\n");
    for (k = 0; k < sizeof sizes / sizeof sizes[0]; ++k) {
        int n = sizes[k];
        size_t reported = kiss_fft_f16_mem_bytes(n);
        printf("    n=%-5d mem=%7zu B (%6.2f KiB)\n", n, reported,
               (double)reported / 1024.0);
#ifdef TEST_F16_WRAP_MALLOC
        {
            kiss_fft_f16_cfg c;
            size_t before = live_bytes;
            WRAP_MARK();
            c = kiss_fft_f16_alloc(n, 0);
            check(c != NULL, "mem test alloc", n, 0);
            check(live_bytes - before == reported, "mem_bytes equals measured footprint",
                  (int)(live_bytes - before), (int)reported);
            check(bytes_since_mark == reported, "all footprint comes from the alloc",
                  (int)bytes_since_mark, (int)reported);
            kiss_fft_f16_free(c);
            check(live_bytes == before, "free returns all heap memory", (int)live_bytes,
                  (int)before);
        }
#endif
    }
    /* closed form check on this platform: 24*nfft + 296 */
    check(kiss_fft_f16_mem_bytes(1024) == 24u * 1024u + 296u, "closed form at 1024",
          (int)kiss_fft_f16_mem_bytes(1024), 24 * 1024 + 296);
    check(kiss_fft_f16_mem_bytes(4096) == 24u * 4096u + 296u, "closed form at 4096",
          (int)kiss_fft_f16_mem_bytes(4096), 24 * 4096 + 296);
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("== kissfft binary16 wrapper validation ==\n");

    test_converters();
    test_failures();
    printf("[3] numeric error and bit identity\n");
    numeric_case(1024);
    numeric_case(4096);
    printf("[4] timing (5 batches; 20000 transforms/batch at 1024, 5000 at 4096)\n");
    timing_case(1024, 5, 20000);
    timing_case(4096, 5, 5000);
    test_memory();

#ifdef TEST_F16_WRAP_MALLOC
    printf("    malloc accounting: %ld allocations, peak=%zu B, live=%zu B\n",
           total_allocs, peak_bytes, live_bytes);
    check(live_allocs == 0 && live_bytes == 0, "no heap memory leaked", (int)live_allocs,
          (int)live_bytes);
#endif

    if (failures == 0) {
        printf("== ALL CHECKS PASSED: %ld checks, 0 failures ==\n", checks);
        return 0;
    }
    printf("== FAILED: %ld checks, %ld failures ==\n", checks, failures);
    return 1;
}
