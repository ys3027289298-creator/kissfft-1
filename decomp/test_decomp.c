/*
 * test_decomp.c — acceptance test / benchmark for kiss_decomp.h.
 *
 * For each requested length (180, 1000, 1024, 997, 1, 0):
 *   - prints the decomposition chosen by the decomposer,
 *   - compares the result against a direct O(N^2) DFT (double precision),
 *   - times the decomposed transform vs the direct DFT,
 *   - documents edge behavior for N = 0 and N = 1.
 *
 * Extra checks:
 *   - N = 1024: decomposer output must be bit-identical to a raw kiss_fft
 *     call (the power-of-two path is untouched library code);
 *   - N = 997: demonstrates the configurable strategies — MIXED_ONLY fails
 *     gracefully, ZEROPAD reproduces the legacy pad-to-1024 behavior.
 *
 * Build (from repo root, no build-system changes):
 *   cc -O2 -Wall -Wextra -I. kiss_fft.c decomp/test_decomp.c -lm -o decomp/test_decomp
 */
#include "kiss_decomp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* deterministic pseudo-random input in [-1,1) */
static void fill_input(kiss_fft_cpx *x, int n)
{
    unsigned long long s = 0x9e3779b97f4a7c15ULL;
    int k;
    for (k = 0; k < n; ++k) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x[k].r = (kiss_fft_scalar)((double)(s >> 33) / 2147483648.0 - 1.0);
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x[k].i = (kiss_fft_scalar)((double)(s >> 33) / 2147483648.0 - 1.0);
    }
}

/* double-precision O(n^2) reference DFT (kiss_fft sign convention) */
static void direct_dft(const kiss_fft_cpx *in, kiss_fft_cpx *out, int n,
                       int inverse)
{
    double s = inverse ? 1.0 : -1.0;
    int k, j;
    for (k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        for (j = 0; j < n; ++j) {
            double ph = s * 2.0 * M_PI * (double)k * (double)j / (double)n;
            double c = cos(ph), sn = sin(ph);
            sr += (double)in[j].r * c - (double)in[j].i * sn;
            si += (double)in[j].r * sn + (double)in[j].i * c;
        }
        out[k].r = (kiss_fft_scalar)sr;
        out[k].i = (kiss_fft_scalar)si;
    }
}

static int g_failures = 0;

static void check(int ok, const char *what)
{
    printf("    %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

/* time `reps` executions, return microseconds per execution */
static double time_exec(const kiss_decomp_plan *pl, const kiss_fft_cpx *in,
                        kiss_fft_cpx *out, int reps)
{
    double t0 = now_sec();
    int r;
    for (r = 0; r < reps; ++r)
        kiss_decomp_exec(pl, in, out);
    return (now_sec() - t0) * 1e6 / (double)reps;
}

static void test_length(int n)
{
    char desc[256];
    kiss_fft_cpx *in, *out, *ref;
    kiss_decomp_plan *pl;
    double max_abs = 0.0, max_ref = 0.0, rel, us_fft, us_dft, t0;
    int k, reps;

    printf("N = %d\n", n);

    pl = kiss_decomp_create(n, 0, KISS_DECOMP_AUTO);
    if (n <= 0) {
        check(pl == NULL, "edge: plan rejected for non-positive length");
        printf("\n");
        return;
    }
    if (!pl) {
        printf("    unexpected: AUTO strategy returned no plan\n");
        ++g_failures;
        printf("\n");
        return;
    }

    kiss_decomp_describe(pl, desc, sizeof(desc));
    printf("    decomposition : %s\n", desc);
    printf("    output length : %d\n", pl->nout);

    in = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
    out = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)pl->nout);
    ref = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)pl->nout);
    fill_input(in, n);

    kiss_decomp_exec(pl, in, out);

    /* reference: direct DFT of the transform the strategy actually computes
     * (for ZEROPAD that is the nout-point DFT of the zero-padded input) */
    if (pl->nout != n) {
        kiss_fft_cpx *padded =
            (kiss_fft_cpx *)calloc((size_t)pl->nout, sizeof(kiss_fft_cpx));
        memcpy(padded, in, sizeof(kiss_fft_cpx) * (size_t)n);
        t0 = now_sec();
        direct_dft(padded, ref, pl->nout, 0);
        us_dft = (now_sec() - t0) * 1e6;
        free(padded);
    } else {
        t0 = now_sec();
        direct_dft(in, ref, n, 0);
        us_dft = (now_sec() - t0) * 1e6;
    }

    for (k = 0; k < pl->nout; ++k) {
        double dr = fabs((double)out[k].r - (double)ref[k].r);
        double di = fabs((double)out[k].i - (double)ref[k].i);
        double mag = hypot((double)ref[k].r, (double)ref[k].i);
        if (dr > max_abs)
            max_abs = dr;
        if (di > max_abs)
            max_abs = di;
        if (mag > max_ref)
            max_ref = mag;
    }
    rel = max_ref > 0.0 ? max_abs / max_ref : max_abs;

    /* timing: enough reps for a stable number */
    reps = (int)(0.2 / (time_exec(pl, in, out, 3) * 1e-6 * 3.0));
    if (reps < 10)
        reps = 10;
    us_fft = time_exec(pl, in, out, reps);

    printf("    max abs err   : %.3e   max |X| = %.3f   rel err = %.3e\n",
           max_abs, max_ref, rel);
    printf("    time          : %.2f us/transform (%d reps)   direct DFT: %.1f "
           "us   speedup: %.1fx\n",
           us_fft, reps, us_dft, us_dft / us_fft);
    check(rel < 1e-3, "accuracy vs direct DFT (rel err < 1e-3)");

    if (n == 1)
        check(out[0].r == in[0].r && out[0].i == in[0].i,
              "edge: N=1 is the identity transform");

    if (n == 1024) { /* power-of-two path must be bit-identical */
        kiss_fft_cfg raw = kiss_fft_alloc(n, 0, NULL, NULL);
        kiss_fft_cpx *rawout =
            (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
        kiss_fft(raw, in, rawout);
        check(memcmp(out, rawout, sizeof(kiss_fft_cpx) * (size_t)n) == 0,
              "pow2 path bit-identical to raw kiss_fft");
        kiss_fft_free(raw);
        free(rawout);
    }

    if (n == 997) { /* demonstrate the configurable strategies */
        kiss_decomp_plan *pm = kiss_decomp_create(n, 0, KISS_DECOMP_MIXED_ONLY);
        kiss_decomp_plan *pz = kiss_decomp_create(n, 0, KISS_DECOMP_ZEROPAD);
        check(pm == NULL, "strategy MIXED_ONLY rejects 997 (prime > 7)");
        if (pz) {
            kiss_fft_cpx *zout = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) *
                                                        (size_t)pz->nout);
            kiss_fft_cpx *zref = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) *
                                                        (size_t)pz->nout);
            kiss_fft_cpx *padded =
                (kiss_fft_cpx *)calloc((size_t)pz->nout, sizeof(kiss_fft_cpx));
            double zerr = 0.0;
            kiss_decomp_describe(pz, desc, sizeof(desc));
            printf("    legacy zeropad: %s\n", desc);
            kiss_decomp_exec(pz, in, zout);
            memcpy(padded, in, sizeof(kiss_fft_cpx) * (size_t)n);
            direct_dft(padded, zref, pz->nout, 0);
            for (k = 0; k < pz->nout; ++k) {
                double dr = fabs((double)zout[k].r - (double)zref[k].r);
                double di = fabs((double)zout[k].i - (double)zref[k].i);
                if (dr > zerr)
                    zerr = dr;
                if (di > zerr)
                    zerr = di;
            }
            printf("    legacy zeropad: max abs err vs %d-point direct DFT = "
                   "%.3e\n",
                   pz->nout, zerr);
            check(pz->nout == 1024 && zerr < 1e-2,
                  "strategy ZEROPAD reproduces legacy pad-to-1024 behavior");
            free(zout);
            free(zref);
            free(padded);
            kiss_decomp_free(pz);
        } else {
            check(0, "strategy ZEROPAD plan created");
        }
    }

    kiss_decomp_free(pl);
    free(in);
    free(out);
    free(ref);
    printf("\n");
}

int main(void)
{
    static const int lengths[] = {180, 1000, 1024, 997, 1, 0};
    size_t i;
    printf("=== kiss_decomp acceptance test (scalar type: %s) ===\n\n",
           sizeof(kiss_fft_scalar) == sizeof(float) ? "float" : "other");
    for (i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i)
        test_length(lengths[i]);
    kiss_fft_cleanup();
    printf("=== %s (%d failure%s) ===\n", g_failures ? "FAILED" : "ALL PASSED",
           g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
