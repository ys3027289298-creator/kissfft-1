/*
 * Unit tests for the phase vocoder C API.
 *
 * Covers: parameter validation, duration contract at alpha 0.5/1/2,
 * alpha=1 reconstruction accuracy, streaming (chunked) vs one-shot
 * equivalence, and the expected-output-length helper.
 *
 * Exits non-zero on any failure.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "phase_vocoder.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NFFT 2048
#define HOP_ANA (NFFT / 8)
#define FS 48000
#define DUR_S 1.0
#define NSAMP (FS * DUR_S)

static int failures = 0;

#define CHECK(cond, msg) do {                                        \
    if (!(cond)) {                                                   \
        printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);      \
        failures++;                                                  \
    } else {                                                         \
        printf("ok:   %s\n", msg);                                   \
    }                                                                \
} while (0)

static void gen_signal(float *x, long n)
{
    /* 440 Hz tone plus a slow amplitude envelope, deterministic. */
    for (long i = 0; i < n; ++i) {
        double t = (double)i / FS;
        double env = 0.5 + 0.5 * sin(2.0 * M_PI * 2.0 * t);
        x[i] = (float)(0.4 * env * sin(2.0 * M_PI * 440.0 * t));
    }
    /* short fade edges */
    int k = FS / 100;
    for (int i = 0; i < k; ++i) {
        x[i] *= (float)i / k;
        x[n - 1 - i] *= (float)i / k;
    }
}

static long run_stream(const float *in, long n, double alpha,
                       float *out, long chunk)
{
    pv_state *pv = pv_alloc(NFFT, HOP_ANA, alpha);
    if (pv == NULL)
        return -1;
    long produced = 0;
    for (long off = 0; off < n; off += chunk) {
        long c = chunk;
        if (off + c > n)
            c = n - off;
        long got = pv_process(pv, in + off, (size_t)c, out + produced);
        if (got < 0) {
            pv_free(pv);
            return -1;
        }
        produced += got;
    }
    produced += pv_flush(pv, out + produced);
    pv_free(pv);
    return produced;
}

static void test_invalid_params(void)
{
    CHECK(pv_alloc(0, HOP_ANA, 1.0) == NULL, "reject nfft=0");
    CHECK(pv_alloc(2047, HOP_ANA, 1.0) == NULL, "reject odd nfft");
    CHECK(pv_alloc(NFFT, 0, 1.0) == NULL, "reject hop=0");
    CHECK(pv_alloc(NFFT, NFFT, 1.0) == NULL, "reject hop > nfft/2");
    CHECK(pv_alloc(NFFT, HOP_ANA, 0.0) == NULL, "reject alpha=0");
    CHECK(pv_alloc(NFFT, HOP_ANA, -1.0) == NULL, "reject negative alpha");
    CHECK(pv_alloc(NFFT, HOP_ANA, NAN) == NULL, "reject NaN alpha");

    pv_state *pv = pv_alloc(NFFT, HOP_ANA, 1.0);
    CHECK(pv != NULL, "alloc valid state");
    float tmp[16];
    CHECK(pv_process(pv, NULL, 8, tmp) == -1, "null input rejected");
    CHECK(pv_process(pv, tmp, 8, NULL) == -1, "null output rejected");
    pv_free(pv);
}

static void test_duration_and_identity(float *in)
{
    double alphas[3] = {0.5, 1.0, 2.0};
    for (int a = 0; a < 3; ++a) {
        double alpha = alphas[a];
        long cap = (long)(NSAMP * alpha) + 2 * NFFT;
        float *out = (float *)malloc((size_t)cap * sizeof(float));

        pv_state *pv = pv_alloc(NFFT, HOP_ANA, alpha);
        long expect = pv_expected_output_len(pv, NSAMP);
        long got = run_stream(in, NSAMP, alpha, out, 4096);

        char buf[128];
        snprintf(buf, sizeof(buf),
                 "alpha=%.1f output length %ld ~= target %.1f (helper %ld)",
                 alpha, got, alpha * NSAMP, expect);
        CHECK(got == expect &&
              fabs((double)got - alpha * NSAMP) <= 1.0, buf);

        if (alpha == 1.0) {
            /* Interior reconstruction: compare away from COLA edges. */
            long edge = NFFT;
            double se = 0.0, sx = 0.0;
            for (long i = edge; i < NSAMP - edge; ++i) {
                double e = in[i] - out[i];
                se += e * e;
                sx += (double)in[i] * in[i];
            }
            double snr = 10.0 * log10(sx / (se + 1e-30));
            snprintf(buf, sizeof(buf),
                     "alpha=1 interior reconstruction SNR %.1f dB > 60 dB",
                     snr);
            CHECK(snr > 60.0, buf);
        }
        free(out);
    }
}

static void test_streaming_equivalence(float *in)
{
    long cap = 2 * NSAMP + 2 * NFFT;
    float *one = (float *)malloc((size_t)cap * sizeof(float));
    float *chunked = (float *)malloc((size_t)cap * sizeof(float));

    long n1 = run_stream(in, NSAMP, 1.5, one, NSAMP);      /* one shot */
    long n2 = run_stream(in, NSAMP, 1.5, chunked, 333);    /* odd chunks */

    CHECK(n1 == n2, "streaming and one-shot produce equal length");
    if (n1 == n2) {
        double maxdiff = 0.0;
        for (long i = 0; i < n1; ++i) {
            double d = fabs((double)one[i] - chunked[i]);
            if (d > maxdiff)
                maxdiff = d;
        }
        char buf[96];
        snprintf(buf, sizeof(buf),
                 "chunked vs one-shot bit-identical (max diff %.2e)",
                 maxdiff);
        CHECK(maxdiff == 0.0, buf);
    }
    free(one);
    free(chunked);
}

int main(void)
{
    float *in = (float *)malloc(NSAMP * sizeof(float));
    gen_signal(in, NSAMP);

    test_invalid_params();
    test_duration_and_identity(in);
    test_streaming_equivalence(in);

    free(in);
    if (failures == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    }
    printf("\n%d TEST(S) FAILED\n", failures);
    return 1;
}
