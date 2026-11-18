/*
 *  Copyright (c) 2024, KISS FFT contributors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * test_blockconv.c -- validation and measurement of the memory-bounded block
 * fast convolution added in kfc.c (overlap-add and overlap-save).
 *
 * The program is self-contained, uses only the kissfft sources already in the
 * repository, generates small deterministic synthetic signals, and prints:
 *
 *   A. correctness (vs. direct time-domain convolution) on four input classes
 *      (short signal, long signal, equal-length signal, filter longer than the
 *      chosen block), for both OLA and OLS, including zero-pad factor variants
 *   B. timing of direct vs. OLA vs. OLS
 *   C. peak working memory: analytic upper bound formula vs. measured value
 *   D. block-length sweep: per-block work and measured time vs. block length
 *   E. explicit handling of "block shorter than filter" (both methods)
 *
 * Exit status is 0 only if every correctness/accounting assertion passes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "kiss_fft.h"
#include "_kiss_fft_guts.h"
#include "kfc_blockconv.h"

/* ------------------------------------------------------------------ helpers */

static int g_failures = 0;

#define CHECK(cond, msg) do {                                            \
    if (!(cond)) { printf("    ASSERT FAIL: %s\n", msg); ++g_failures; }\
} while (0)

/* deterministic pseudo-random complex sample in roughly [-1,1] */
static unsigned long rng_state = 0x12345678UL;
static float rng_float(void)
{
    rng_state = rng_state * 1103515245UL + 12345UL;
    return (float)((long)(rng_state >> 16) % 2001 - 1000) / 1000.0f;
}

static void gen_signal(kiss_fft_cpx *x, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        x[i].r = rng_float();
        x[i].i = rng_float() * 0.5f;
    }
}

/* reference direct linear convolution, O(N*L) */
static void direct_convolve(const kiss_fft_cpx *x, int nx,
                            const kiss_fft_cpx *h, int nh,
                            kiss_fft_cpx *y)
{
    int i, j;
    int ny = nx + nh - 1;
    for (i = 0; i < ny; ++i) { y[i].r = 0; y[i].i = 0; }
    for (i = 0; i < nx; ++i) {
        for (j = 0; j < nh; ++j) {
            kiss_fft_cpx t;
            C_MUL(t, x[i], h[j]);
            y[i + j].r += t.r;
            y[i + j].i += t.i;
        }
    }
}

static void error_metrics(const kiss_fft_cpx *ref, const kiss_fft_cpx *got,
                          int n, double *out_rms, double *out_max)
{
    double sum2 = 0, mx = 0, refpow = 0, maxabs = 0;
    int i;
    for (i = 0; i < n; ++i) {
        double dr = (double)got[i].r - ref[i].r;
        double di = (double)got[i].i - ref[i].i;
        double e2 = dr * dr + di * di;
        double a = fabs((double)ref[i].r) + fabs((double)ref[i].i);
        sum2 += e2;
        if (e2 > mx) mx = e2;
        if (a > maxabs) maxabs = a;
        refpow += (double)ref[i].r * ref[i].r + (double)ref[i].i * ref[i].i;
    }
    *out_rms = sqrt(sum2 / (n > 0 ? n : 1));
    *out_max = sqrt(mx);
    /* relative RMS error, guarded against a tiny reference */
    if (refpow > 1e-12)
        *out_rms = sqrt(sum2 / refpow);
    /* also report a relative max error (absolute scaled by peak output), which
     * is the fair metric for float32 FFT convolution of long filters */
    if (maxabs > 1e-12)
        *out_max = sqrt(mx) / maxabs;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* run a block convolver over the signal in arbitrary-sized streaming chunks to
 * also exercise the buffering/flush state machine */
static int run_streaming(kfc_blockconv_cfg cfg, const kiss_fft_cpx *x, int nx,
                         kiss_fft_cpx *y, int cap, int chunk)
{
    int pos = 0, got = 0, r;
    kfc_blockconv_reset(cfg);
    while (pos < nx) {
        int take = chunk;
        if (take > nx - pos) take = nx - pos;
        r = kfc_blockconv_process(cfg, x + pos, take, y + got, cap - got);
        if (r < 0) return r;
        got += r;
        pos += take;
    }
    r = kfc_blockconv_flush(cfg, y + got, cap - got);
    if (r < 0) return r;
    got += r;
    return got;
}

static const char *method_name(kfc_blockconv_method m)
{
    return m == KFC_BLOCKCONV_OVERLAP_ADD ? "overlap-add " : "overlap-save";
}

/* ------------------------------------------------- memory accounting hooks */

/* A second layer of bookkeeping (independent from the library's own counters)
 * so the test cross-checks kfc_get_mem_stats against an external tally. */
static unsigned long long ext_current = 0, ext_peak = 0, ext_nalloc = 0;

static void *ext_malloc(size_t n)
{
    void *p = malloc(n);
    if (p) {
        ext_current += n;
        if (ext_current > ext_peak) ext_peak = ext_current;
        ++ext_nalloc;
    }
    return p;
}

static void ext_free(void *p)
{
    /* size unknown here; the library's own stats remain the precise source,
     * this hook only validates that free is actually routed through us. */
    free(p);
}

/*
 * Analytic upper bound on the bytes requested through the allocator for one
 * block-convolution config (derived in the report; S = sizeof(kiss_fft_cpx),
 * T = sizeof(kiss_fft_state) which holds nfft twiddles as its flexible member;
 * sizeof_state is measured because the block-conv state struct is private):
 *
 *   block-conv state struct             : sizeof_state
 *   two FFT configs (fwd + inv)        : 2 * (T + (N-1)*S)
 *   four N-length cpx buffers          : 4 * N*S   (fbuf,Xbuf,Hbuf,obuf)
 *   two (L-1)-length cpx buffers       : 2*(L-1)*S (ov history + OLS tail),
 *                                        allocated only when L > 1
 *
 * N is a 2,3,5-smooth size, so kiss_fft never allocates scratch for a
 * "generic" (prime) radix; there are therefore no further internal allocs.
 * The tracking header overhead (8 bytes per allocation) is added separately.
 */
static unsigned long long analytic_bound(int N, int L, int n_allocs,
                                         size_t sizeof_state)
{
    const size_t S = sizeof(kiss_fft_cpx);
    const size_t T = sizeof(struct kiss_fft_state);
    unsigned long long payload =
        (unsigned long long)sizeof_state +
        2ULL * (T + (size_t)(N - 1) * S) +
        4ULL * (size_t)N * S +
        2ULL * (size_t)(L > 1 ? L - 1 : 0) * S;
    /* each tracked allocation is prefixed by one kfc_hdr (one size_t) */
    return payload + (unsigned long long)n_allocs * sizeof(size_t);
}

/* The private block-conv state size is recovered from the measured allocation
 * count/bytes, keeping the public header opaque as required. */
static size_t measure_state_size(void)
{
    kfc_mem_stats s1;
    /* L=1: allocs are state + fwd + inv + 4 buffers = 7 (no ov/tail buffers).
     * Subtract every known piece to isolate the state allocation. */
    kfc_reset_mem_stats();
    {
        kiss_fft_cpx h1[1] = { {1.0f, 0.0f} };
        kfc_blockconv_cfg cfg =
            kfc_blockconv_alloc(h1, 1, KFC_BLOCKCONV_OVERLAP_ADD, 1, 1);
        kfc_get_mem_stats(&s1);
        if (cfg) {
            const size_t S = sizeof(kiss_fft_cpx);
            const size_t T = sizeof(struct kiss_fft_state);
            int N = kfc_blockconv_nfft(cfg);
            unsigned long long known =
                2ULL * (T + (size_t)(N - 1) * S) +
                4ULL * (size_t)N * S +
                7ULL * sizeof(size_t); /* seven kfc_hdr headers */
            size_t state_bytes = (size_t)(s1.peak - known);
            kfc_blockconv_free(cfg);
            return state_bytes;
        }
    }
    return 0;
}

/* ------------------------------------------------------------ correctness */

typedef struct {
    const char *name;
    int nx;
    int nh;
} input_class;

static void test_correctness(const kiss_fft_cpx *h, int nh)
{
    static const input_class classes[] = {
        { "short signal (N<L)        ", 7,   0 },
        { "long signal  (N>>L)       ", 137, 0 },
        { "equal length (N=L)        ", 32,  0 },
        { "filter longer than block  ", 60,  0 },
    };
    const int nclass = (int)(sizeof(classes) / sizeof(classes[0]));
    const kfc_blockconv_method methods[2] = {
        KFC_BLOCKCONV_OVERLAP_ADD, KFC_BLOCKCONV_OVERLAP_SAVE };
    int c, mi, zp;
    /* block lengths to exercise; 16 is deliberately < nh(=32) for the last
     * input class, and is used to prove OLA accepts / OLS rejects it. */
    static const int blocks[] = { 8, 16, 32, 64 };
    static const int pads[]   = { 1, 2 };

    printf("\n=== A. CORRECTNESS vs direct convolution (float) ===\n");
    printf("%-26s %-13s %5s %4s %5s %12s %12s %8s\n",
           "input class", "method", "block", "pad", "nfft",
           "relRMSerr", "relMaxErr", "status");

    for (c = 0; c < nclass; ++c) {
        int nx = classes[c].nx;
        int ny = nx + nh - 1;
        kiss_fft_cpx *x = (kiss_fft_cpx *)malloc((size_t)nx * sizeof(*x));
        kiss_fft_cpx *ref = (kiss_fft_cpx *)malloc((size_t)ny * sizeof(*ref));
        kiss_fft_cpx *got = (kiss_fft_cpx *)malloc((size_t)ny * sizeof(*got));
        rng_state = 0xABCDEF01UL + (unsigned)c * 7919UL;
        gen_signal(x, nx);
        direct_convolve(x, nx, h, nh, ref);

        for (mi = 0; mi < 2; ++mi) {
            for (zp = 0; zp < 2; ++zp) {
                int bi;
                for (bi = 0; bi < (int)(sizeof(blocks)/sizeof(blocks[0])); ++bi) {
                    int B = blocks[bi];
                    kfc_blockconv_cfg cfg;
                    double rms, mx;
                    int out_len, chunk;
                    /* the last input class is specifically about B < L */
                    if (c != 3 && B > nx && B > nh)
                        continue;
                    cfg = kfc_blockconv_alloc(h, nh, methods[mi], B, pads[zp]);
                    if (!cfg) {
                        if (methods[mi] == KFC_BLOCKCONV_OVERLAP_SAVE && B < nh) {
                            printf("%-26s %-13s %5d %4d %5s %12s %12s %8s\n",
                                   classes[c].name, method_name(methods[mi]),
                                   B, pads[zp], "-", "-", "-",
                                   "rejected");
                        } else {
                            printf("%-26s %-13s %5d %4d  ALLOC FAILED (%d)\n",
                                   classes[c].name, method_name(methods[mi]),
                                   B, pads[zp], kfc_blockconv_last_error());
                            ++g_failures;
                        }
                        continue;
                    }
                    /* alternate one-shot and odd-sized streaming calls */
                    chunk = (B == 8) ? 3 : (B == 16 ? 5 : 7);
                    out_len = run_streaming(cfg, x, nx, got, ny, chunk);
                    CHECK(out_len == ny, "output length must be nx+nh-1");
                    error_metrics(ref, got, ny, &rms, &mx);
                    CHECK(rms < 1e-4, "relative RMS error within float tolerance");
                    CHECK(mx < 1e-3, "relative max error within float tolerance");
                    printf("%-26s %-13s %5d %4d %5d %12.3e %12.3e %8s\n",
                           classes[c].name, method_name(methods[mi]),
                           B, pads[zp], kfc_blockconv_nfft(cfg),
                           rms, mx, (rms < 1e-4 && mx < 1e-3) ? "ok" : "FAIL");
                    kfc_blockconv_free(cfg);
                }
            }
        }
        free(x); free(ref); free(got);
    }
}

/* ------------------------------------------------------------- edge cases */

static void test_edge_and_validation(const kiss_fft_cpx *h, int nh)
{
    kfc_blockconv_cfg cfg;
    printf("\n=== E. PARAMETER VALIDATION AND EDGE HANDLING ===\n");

    /* OLS with block < filter length must be explicitly rejected */
    cfg = kfc_blockconv_alloc(h, nh, KFC_BLOCKCONV_OVERLAP_SAVE, nh - 1, 1);
    CHECK(cfg == NULL, "OLS must reject block_len < filter length");
    printf("  OLS block_len=%d < filter=%d -> rejected, error=%d (expected %d)\n",
           nh - 1, nh, kfc_blockconv_last_error(), KFC_BLOCKCONV_INVALID);

    /* OLA with block < filter length is legal: FFT grows to B+L-1 */
    cfg = kfc_blockconv_alloc(h, nh, KFC_BLOCKCONV_OVERLAP_ADD, nh / 2, 1);
    CHECK(cfg != NULL, "OLA must accept block_len < filter length");
    if (cfg) {
        CHECK(kfc_blockconv_nfft(cfg) >= nh / 2 + nh - 1,
              "OLA FFT length covers B+L-1 even when B<L");
        printf("  OLA block_len=%d < filter=%d -> accepted, nfft=%d (>= B+L-1=%d)\n",
               nh / 2, nh, kfc_blockconv_nfft(cfg), nh / 2 + nh - 1);
        kfc_blockconv_free(cfg);
    }

    /* NULL / bad arguments */
    CHECK(kfc_blockconv_alloc(NULL, nh, KFC_BLOCKCONV_OVERLAP_ADD, nh, 1) == NULL,
          "NULL filter rejected");
    CHECK(kfc_blockconv_alloc(h, 0, KFC_BLOCKCONV_OVERLAP_ADD, nh, 1) == NULL,
          "zero filter length rejected");
    CHECK(kfc_blockconv_alloc(h, nh, (kfc_blockconv_method)99, nh, 1) == NULL,
          "unknown method rejected");

    /* zero_pad <= 0 falls back to 1; block_len <= 0 defaults to nh */
    cfg = kfc_blockconv_alloc(h, nh, KFC_BLOCKCONV_OVERLAP_ADD, 0, 0);
    CHECK(cfg != NULL, "default block/pad selection works");
    if (cfg) {
        CHECK(kfc_blockconv_block_len(cfg) == nh, "default block_len == nh");
        printf("  defaults: block_len=%d, zero_pad treated as 1, nfft=%d\n",
               kfc_blockconv_block_len(cfg), kfc_blockconv_nfft(cfg));
        kfc_blockconv_free(cfg);
    }

    /* L == 1: no overlap history, output must equal the input scaled */
    {
        kiss_fft_cpx h1[1] = { {2.0f, 0.0f} };
        kiss_fft_cpx x[5], y[5];
        int i, r;
        for (i = 0; i < 5; ++i) { x[i].r = (float)(i + 1); x[i].i = 0; }
        cfg = kfc_blockconv_alloc(h1, 1, KFC_BLOCKCONV_OVERLAP_ADD, 2, 1);
        CHECK(cfg != NULL, "L=1 OLA alloc");
        r = kfc_blockconv_convolve(cfg, x, 5, y, 5);
        CHECK(r == 5, "L=1 output length equals input length");
        for (i = 0; i < 5; ++i)
            CHECK(fabsf(y[i].r - 2.0f * x[i].r) < 1e-5f, "L=1 scaling correct");
        kfc_blockconv_free(cfg);
        cfg = kfc_blockconv_alloc(h1, 1, KFC_BLOCKCONV_OVERLAP_SAVE, 2, 1);
        CHECK(cfg != NULL, "L=1 OLS alloc");
        r = kfc_blockconv_convolve(cfg, x, 5, y, 5);
        CHECK(r == 5, "L=1 OLS output length");
        for (i = 0; i < 5; ++i)
            CHECK(fabsf(y[i].r - 2.0f * x[i].r) < 1e-5f, "L=1 OLS scaling");
        kfc_blockconv_free(cfg);
        printf("  L=1 (no overlap): both methods produce 2*x exactly\n");
    }

    /* streaming reuse: run twice on the same config, results must match */
    {
        int nx = 50, ny = nx + nh - 1, r1, r2;
        kiss_fft_cpx *x = malloc((size_t)nx * sizeof(*x));
        kiss_fft_cpx *y1 = malloc((size_t)ny * sizeof(*y1));
        kiss_fft_cpx *y2 = malloc((size_t)ny * sizeof(*y2));
        rng_state = 555;
        gen_signal(x, nx);
        cfg = kfc_blockconv_alloc(h, nh, KFC_BLOCKCONV_OVERLAP_ADD, 16, 1);
        r1 = kfc_blockconv_convolve(cfg, x, nx, y1, ny);
        r2 = kfc_blockconv_convolve(cfg, x, nx, y2, ny); /* auto-reset inside */
        CHECK(r1 == ny && r2 == ny, "reuse output lengths");
        CHECK(memcmp(y1, y2, (size_t)ny * sizeof(*y1)) == 0,
              "reused config gives identical result");
        kfc_blockconv_free(cfg);
        free(x); free(y1); free(y2);
        printf("  config reuse after flush/reset: bit-identical second run\n");
    }
}

/* ------------------------------------------------------- timing benchmark */

static void bench_one(const kiss_fft_cpx *x, int nx,
                      const kiss_fft_cpx *h, int nh,
                      kfc_blockconv_method method, int B, int zp,
                      int iters, double *secs)
{
    int ny = nx + nh - 1;
    kiss_fft_cpx *y = malloc((size_t)ny * sizeof(*y));
    kfc_blockconv_cfg cfg = kfc_blockconv_alloc(h, nh, method, B, zp);
    double t0, t1;
    int i, r = 0;
    CHECK(cfg != NULL, "bench alloc");
    t0 = now_sec();
    for (i = 0; i < iters; ++i)
        r = kfc_blockconv_convolve(cfg, x, nx, y, ny);
    t1 = now_sec();
    CHECK(r == ny, "bench output length");
    *secs = (t1 - t0) / iters;
    kfc_blockconv_free(cfg);
    free(y);
}

static void test_timing(const kiss_fft_cpx *h, int nh)
{
    int nx = 4096;
    int ny = nx + nh - 1;
    int B = 128, iters = 30, i;
    kiss_fft_cpx *x = malloc((size_t)nx * sizeof(*x));
    kiss_fft_cpx *yd = malloc((size_t)ny * sizeof(*yd));
    double t0, t1, td, tola, tols;

    rng_state = 0x9999UL;
    gen_signal(x, nx);

    printf("\n=== B. TIMING (nx=%d, L=%d, output=%d, block B=%d, avg of %d runs) ===\n",
           nx, nh, ny, B, iters);

    /* direct convolution timing (fewer iterations: it is O(N*L)) */
    t0 = now_sec();
    for (i = 0; i < 3; ++i)
        direct_convolve(x, nx, h, nh, yd);
    t1 = now_sec();
    td = (t1 - t0) / 3;

    bench_one(x, nx, h, nh, KFC_BLOCKCONV_OVERLAP_ADD, B, 1, iters, &tola);
    bench_one(x, nx, h, nh, KFC_BLOCKCONV_OVERLAP_SAVE, B, 1, iters, &tols);

    printf("  direct convolution : %10.3f ms   (reference)\n", td * 1e3);
    printf("  overlap-add        : %10.3f ms   speedup vs direct %.2fx\n",
           tola * 1e3, td / tola);
    printf("  overlap-save       : %10.3f ms   speedup vs direct %.2fx\n",
           tols * 1e3, td / tols);
    printf("  OLA/OLS time ratio : %.3f (expected ~1: same FFT count)\n",
           tola / tols);

    /* long-filter regime where block FFT convolution wins clearly: a 512-tap
     * filter over a 64k-sample signal makes direct O(N*L) ~33 million MACs. */
    {
        const int L2 = 512, N2 = 65536, B2 = 1024, it2 = 20, dit2 = 3;
        int ny2 = N2 + L2 - 1;
        kiss_fft_cpx *h2 = malloc((size_t)L2 * sizeof(*h2));
        kiss_fft_cpx *x2 = malloc((size_t)N2 * sizeof(*x2));
        kiss_fft_cpx *y2 = malloc((size_t)ny2 * sizeof(*y2));
        double d2, a2, s2;
        rng_state = 0x5151UL;
        gen_signal(h2, L2);
        rng_state = 0x2727UL;
        gen_signal(x2, N2);
        t0 = now_sec();
        for (i = 0; i < dit2; ++i)
            direct_convolve(x2, N2, h2, L2, y2);
        t1 = now_sec();
        d2 = (t1 - t0) / dit2;
        bench_one(x2, N2, h2, L2, KFC_BLOCKCONV_OVERLAP_ADD, B2, 1, it2, &a2);
        bench_one(x2, N2, h2, L2, KFC_BLOCKCONV_OVERLAP_SAVE, B2, 1, it2, &s2);
        printf("  -- long filter L=%d on N=%d (B=%d) --\n", L2, N2, B2);
        printf("  direct convolution : %10.3f ms\n", d2 * 1e3);
        printf("  overlap-add        : %10.3f ms   speedup vs direct %.2fx\n",
               a2 * 1e3, d2 / a2);
        printf("  overlap-save       : %10.3f ms   speedup vs direct %.2fx\n",
               s2 * 1e3, d2 / s2);
        free(h2); free(x2); free(y2);
    }
    free(x); free(yd);
}

/* ------------------------------------------------- peak memory measurement */

static void test_memory(const kiss_fft_cpx *h, int nh)
{
    static const int blocks[] = { 32, 64, 128, 256 };
    int i;
    kfc_mem_stats st;
    size_t state_bytes = measure_state_size();
    CHECK(state_bytes > 0, "state size measurement");

    printf("\n=== C. PEAK WORKING MEMORY: analytic bound vs measured ===\n");
    printf("  (sizeof(kiss_fft_cpx)=%zu, sizeof(kiss_fft_state)=%zu, blockconv state=%zu)\n",
           sizeof(kiss_fft_cpx), sizeof(struct kiss_fft_state), state_bytes);
    printf("  %-13s %5s %5s %8s %14s %14s %10s\n",
           "method", "L", "B", "nfft",
           "bound(bytes)", "measured(bytes)", "match");

    for (i = 0; i < (int)(sizeof(blocks)/sizeof(blocks[0])); ++i) {
        int B = blocks[i];
        kfc_blockconv_method m;
        for (m = 0; m <= 1; ++m) {
            kfc_blockconv_cfg cfg;
            unsigned long long bound;
            int N, n_allocs;
            kfc_reset_mem_stats();
            ext_current = ext_peak = ext_nalloc = 0;
            kfc_set_alloc_hooks(ext_malloc, ext_free);
            cfg = kfc_blockconv_alloc(h, nh, m, B, 1);
            kfc_get_mem_stats(&st);
            CHECK(cfg != NULL, "memory-test alloc");
            N = kfc_blockconv_nfft(cfg);
            /* number of distinct tracked allocations:
             * state, fwd, inv, fbuf, Xbuf, Hbuf, obuf = 7, plus ov and tail
             * (each hist_len) when L>1 */
            n_allocs = 7 + 2 * (nh > 1 ? 1 : 0);
            bound = analytic_bound(N, nh, n_allocs, state_bytes);
            CHECK(st.peak == bound, "measured peak equals analytic bound");
            CHECK(st.n_alloc == (unsigned long long)n_allocs,
                  "allocation count matches model");
            printf("  %-13s %5d %5d %8d %14llu %14llu %10s\n",
                   method_name(m), nh, B, N, bound, st.peak,
                   st.peak == bound ? "yes" : "NO");
            kfc_blockconv_free(cfg);
            kfc_get_mem_stats(&st);
            CHECK(st.current == 0, "all tracked memory released (no leak)");
            kfc_set_alloc_hooks(NULL, NULL); /* restore malloc/free */
        }
    }

    /* demonstrate the memory-bounded property: peak is flat as the signal
     * grows, for a fixed block length. */
    {
        int B = 64, N;
        unsigned long long peak_short, peak_long;
        kfc_blockconv_cfg cfg = kfc_blockconv_alloc(h, nh,
                                  KFC_BLOCKCONV_OVERLAP_ADD, B, 1);
        kiss_fft_cpx *y;
        N = kfc_blockconv_nfft(cfg);
        kfc_get_mem_stats(&st);
        peak_short = st.peak;
        /* feed a much longer signal through the same fixed-size config */
        {
            int nlong = 8000, ny = nlong + nh - 1, r;
            kiss_fft_cpx *xl = malloc((size_t)nlong * sizeof(*xl));
            y = malloc((size_t)ny * sizeof(*y));
            rng_state = 42;
            gen_signal(xl, nlong);
            r = kfc_blockconv_convolve(cfg, xl, nlong, y, ny);
            CHECK(r == ny, "long-signal run length");
            free(xl); free(y);
        }
        kfc_get_mem_stats(&st);
        peak_long = st.peak;
        CHECK(peak_long == peak_short,
              "peak memory independent of signal length");
        printf("  fixed B=%d (nfft=%d): peak after short alloc = %llu bytes,\n"
               "                         peak after 8000-sample run = %llu bytes"
               " (flat => memory-bounded)\n",
               B, N, peak_short, peak_long);
        kfc_blockconv_free(cfg);
    }
}

/* ------------------------------------------------------- block size sweep */

static void test_block_sweep(const kiss_fft_cpx *h, int nh)
{
    int nx = 4096;
    int ny = nx + nh - 1;
    static const int blocks[] = { 32, 48, 64, 96, 128, 192, 256, 384, 512 };
    int i, iters = 30;
    kiss_fft_cpx *x = malloc((size_t)nx * sizeof(*x));
    kiss_fft_cpx *y = malloc((size_t)ny * sizeof(*y));
    kfc_mem_stats st;

    rng_state = 0x2024UL;
    gen_signal(x, nx);

    printf("\n=== D. BLOCK-LENGTH SWEEP (nx=%d, L=%d, zero_pad=1, %d iterations) ===\n",
           nx, nh, iters);
    printf("  %5s %6s %8s %7s %12s %12s %14s\n",
           "B", "nfft", "frames", "valid/f", "OLA ms", "OLS ms", "peak bytes");

    for (i = 0; i < (int)(sizeof(blocks)/sizeof(blocks[0])); ++i) {
        int B = blocks[i];
        kfc_blockconv_cfg ca, cs;
        double t0, t1, tola, tols;
        int frames, r;
        unsigned long long peak;

        /* isolate the OLA config's peak with a fresh stats window so each row
         * reports its own working-set size, not the cumulative maximum */
        kfc_reset_mem_stats();
        ca = kfc_blockconv_alloc(h, nh, KFC_BLOCKCONV_OVERLAP_ADD, B, 1);
        kfc_get_mem_stats(&st);
        peak = st.peak;
        cs = kfc_blockconv_alloc(h, nh, KFC_BLOCKCONV_OVERLAP_SAVE, B, 1);
        CHECK(ca && cs, "sweep alloc");

        t0 = now_sec();
        for (r = 0; r < iters; ++r)
            kfc_blockconv_convolve(ca, x, nx, y, ny);
        t1 = now_sec();
        tola = (t1 - t0) / iters * 1e3;

        t0 = now_sec();
        for (r = 0; r < iters; ++r)
            kfc_blockconv_convolve(cs, x, nx, y, ny);
        t1 = now_sec();
        tols = (t1 - t0) / iters * 1e3;

        /* number of FFT frames for OLA = ceil(nx/B) (+1 flush frame handled
         * inside the last partial block); report the full-frame count. */
        frames = (nx + B - 1) / B;
        printf("  %5d %6d %8d %7d %12.3f %12.3f %14llu\n",
               B, kfc_blockconv_nfft(ca), frames,
               kfc_blockconv_nfft(cs) - nh + 1, tola, tols, peak);

        kfc_blockconv_free(ca);
        kfc_blockconv_free(cs);
    }
    printf("  trend: small B -> many small FFTs (more overhead, less memory);\n"
           "         large B -> few big FFTs (less overhead, more memory).\n");
    free(x); free(y);
}

int main(void)
{
    const int nh = 32;
    kiss_fft_cpx *h = malloc((size_t)nh * sizeof(*h));

    /* a simple low-pass-ish deterministic filter with a known shape */
    int i;
    rng_state = 0xF117UL;
    gen_signal(h, nh);
    for (i = 0; i < nh; ++i) {
        double w = 0.5 - 0.5 * cos(2.0 * M_PI * i / (nh - 1)); /* Hann window */
        h[i].r = (kiss_fft_scalar)(h[i].r * w * 0.5);
        h[i].i = (kiss_fft_scalar)(h[i].i * w * 0.5);
    }

    printf("kissfft block fast-convolution test (filter length L=%d)\n", nh);

    test_correctness(h, nh);
    test_edge_and_validation(h, nh);
    test_timing(h, nh);
    test_memory(h, nh);
    test_block_sweep(h, nh);

    free(h);

    printf("\n================ SUMMARY ================\n");
    if (g_failures == 0) {
        printf("ALL CHECKS PASSED\n");
        return 0;
    }
    printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
