/*
 *  Regression and performance test for the cepstral pitch detector.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 *
 *  Checks:
 *   1. Harmonic frames at 110/220/440/880 Hz peak within one quefrency bin
 *      of the true period.
 *   2. Silence is reported explicitly without touching the transforms.
 *   3. Audible but aperiodic noise is reported as unvoiced.
 *   4. The lifter order is configurable at runtime.
 *   5. A 4096-point single-frame detect() runs within the timing budget.
 */

#include "kiss_cepstrum.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <tgmath.h>

#ifndef NFFT
#define NFFT 4096
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 16000
#endif

/* Timing budget for one 4096-point frame, in milliseconds. */
#ifndef TIME_BUDGET_MS
#define TIME_BUDGET_MS 0.3
#endif

#define NUM_PITCHES 4
#define NUM_TIMING_RUNS 1000

static const double test_pitches[NUM_PITCHES] = { 110, 220, 440, 880 };

static int failures;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL: ");                                                \
            printf(__VA_ARGS__);                                             \
            printf("\n");                                                    \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

/* Harmonic series with decaying amplitudes: the kind of signal whose
 * log spectrum shows the evenly spaced ripples the cepstrum detects. */
static void synth_harmonic(kiss_fft_scalar *frame, int n, double f0)
{
    int k, h;
    for (k = 0; k < n; ++k) {
        double t = (double)k / SAMPLE_RATE;
        double v = 0.0;
        for (h = 1; h * f0 < SAMPLE_RATE / 2.0; ++h)
            v += sin(2.0 * 3.14159265358979323846 * h * f0 * t) / h;
        frame[k] = (kiss_fft_scalar)(0.25 * v);
    }
}

static void synth_noise(kiss_fft_scalar *frame, int n, double rms)
{
    int k;
    for (k = 0; k < n; ++k) {
        int r = rand();
        double u = (double)r / RAND_MAX - 0.5;
        frame[k] = (kiss_fft_scalar)(2.0 * rms * u);
    }
}

static double elapsed_ms(const struct timespec *t0, const struct timespec *t1)
{
    return (t1->tv_sec - t0->tv_sec) * 1000.0 +
           (t1->tv_nsec - t0->tv_nsec) / 1.0e6;
}

int main(void)
{
    kiss_cepstrum_config cfg;
    kiss_cepstrum_cfg st;
    kiss_cepstrum_pitch result;
    kiss_fft_scalar *frame =
        (kiss_fft_scalar *)malloc((size_t)NFFT * sizeof(kiss_fft_scalar));
    kiss_fft_scalar *cepstrum =
        (kiss_fft_scalar *)malloc((size_t)NFFT * sizeof(kiss_fft_scalar));
    int i, rc;

    if (frame == NULL || cepstrum == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    srand(12345);

    kiss_cepstrum_default_config(&cfg, NFFT, SAMPLE_RATE);
    st = kiss_cepstrum_alloc(&cfg);
    CHECK(st != NULL, "alloc with default config failed");
    if (st == NULL)
        goto out;

    /* ---- 1. pitch accuracy over the four test frequencies ---- */
    for (i = 0; i < NUM_PITCHES; ++i) {
        double f0 = test_pitches[i];
        int expected_bin = (int)lround((double)SAMPLE_RATE / f0);
        int err;

        synth_harmonic(frame, NFFT, f0);
        rc = kiss_cepstrum_detect(st, frame, &result);
        err = abs(result.quefrency_bin - expected_bin);

        printf("f0=%5.0f Hz: rc=%d bin=%d (expected %d, error %d) "
               "pitch=%.2f Hz rms=%.4f peak=%.4f\n",
               f0, rc, result.quefrency_bin, expected_bin, err,
               (double)result.pitch_hz, (double)result.rms,
               (double)result.peak);
        CHECK(rc == KISS_CEPSTRUM_VOICED,
              "f0=%.0f Hz not detected as voiced", f0);
        CHECK(err <= 1,
              "f0=%.0f Hz peak off by %d bins, expected %d got %d",
              f0, err, expected_bin, result.quefrency_bin);
    }

    /* ---- 2. silence handling: exact zero frame ---- */
    for (i = 0; i < NFFT; ++i)
        frame[i] = 0;
    rc = kiss_cepstrum_detect(st, frame, &result);
    printf("zero frame: rc=%d rms=%.6f\n", rc, (double)result.rms);
    CHECK(rc == KISS_CEPSTRUM_SILENCE, "zero frame not flagged as silence");
    CHECK(result.voiced == 0 && result.pitch_hz == 0,
          "silent frame produced a pitch");

    /* near-silence below the configured RMS floor */
    synth_noise(frame, NFFT, (double)cfg.silence_rms * 0.1);
    rc = kiss_cepstrum_detect(st, frame, &result);
    printf("near-silent frame: rc=%d rms=%.6f\n",
           rc, (double)result.rms);
    CHECK(rc == KISS_CEPSTRUM_SILENCE,
          "sub-threshold frame not flagged as silence");

    /* ---- 3. unvoiced: white noise well above the silence floor ---- */
    synth_noise(frame, NFFT, 0.1);
    rc = kiss_cepstrum_detect(st, frame, &result);
    printf("white noise: rc=%d rms=%.4f peak=%.4f\n",
           rc, (double)result.rms, (double)result.peak);
    CHECK(rc == KISS_CEPSTRUM_UNVOICED,
          "white noise reported as voiced (peak %.4f)",
          (double)result.peak);

    /* ---- 4. lifter order is configurable and actually applied ---- */
    rc = kiss_cepstrum_set_lifter_order(st, 2);
    CHECK(rc == 2, "lifter order 2 not accepted (got %d)", rc);
    synth_harmonic(frame, NFFT, 220.0);
    (void)kiss_cepstrum_compute(st, frame, cepstrum);
    CHECK(cepstrum[0] == 0 && cepstrum[1] == 0,
          "lifter did not zero low quefrencies");
    rc = kiss_cepstrum_detect(st, frame, &result);
    CHECK(rc == KISS_CEPSTRUM_VOICED &&
              abs(result.quefrency_bin -
                  (int)lround((double)SAMPLE_RATE / 220.0)) <= 1,
          "pitch lost after lifter reconfiguration");

    /* disabling the lifter restores the DC cepstrum coefficient */
    rc = kiss_cepstrum_set_lifter_order(st, 0);
    CHECK(rc == 0, "lifter could not be disabled");
    (void)kiss_cepstrum_compute(st, frame, cepstrum);
    CHECK(cepstrum[0] != 0, "lifter still active after being disabled");

    /* out-of-range orders are rejected or clamped, never crash */
    CHECK(kiss_cepstrum_set_lifter_order(st, -1) == -1,
          "negative lifter order accepted");
    CHECK(kiss_cepstrum_set_lifter_order(st, NFFT) == -1,
          "oversized lifter order accepted");

    /* invalid configs must be rejected at alloc time */
    {
        kiss_cepstrum_config bad = cfg;
        bad.nfft = 4095; /* odd */
        CHECK(kiss_cepstrum_alloc(&bad) == NULL, "odd nfft accepted");
        bad = cfg;
        bad.max_pitch_hz = bad.min_pitch_hz;
        CHECK(kiss_cepstrum_alloc(&bad) == NULL, "empty pitch range accepted");
    }

    /* ---- 5. single-frame timing on 4096 points ---- */
    synth_harmonic(frame, NFFT, 220.0);
    {
        struct timespec t0, t1;
        static double samples[NUM_TIMING_RUNS];
        double raw_worst = 0.0, p99, avg = 0.0;
        int j;

        /* Warm up: the first calls pay allocator/cache cold-start costs. */
        for (i = 0; i < 50; ++i)
            rc = kiss_cepstrum_detect(st, frame, &result);

        for (i = 0; i < NUM_TIMING_RUNS; ++i) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            rc = kiss_cepstrum_detect(st, frame, &result);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            samples[i] = elapsed_ms(&t0, &t1);
            avg += samples[i];
            if (samples[i] > raw_worst)
                raw_worst = samples[i];
        }
        avg /= NUM_TIMING_RUNS;

        /* Simple insertion sort; NUM_TIMING_RUNS is small. The budget is
         * checked against the 99th percentile so a single scheduler
         * hiccup cannot mask the algorithm's actual cost. */
        for (i = 1; i < NUM_TIMING_RUNS; ++i) {
            double v = samples[i];
            for (j = i; j > 0 && samples[j - 1] > v; --j)
                samples[j] = samples[j - 1];
            samples[j] = v;
        }
        p99 = samples[NUM_TIMING_RUNS * 99 / 100];

        printf("timing over %d x %d-point frames: avg %.4f ms, "
               "p99 %.4f ms, raw worst %.4f ms (budget %.2f ms)\n",
               NUM_TIMING_RUNS, NFFT, avg, p99, raw_worst, TIME_BUDGET_MS);
        CHECK(p99 <= TIME_BUDGET_MS,
              "p99 frame time %.4f ms exceeds %.2f ms",
              p99, (double)TIME_BUDGET_MS);
    }

    KISS_FFT_FREE(st);
out:
    free(frame);
    free(cepstrum);

    if (failures == 0) {
        printf("all cepstrum pitch tests passed\n");
        return 0;
    }
    printf("%d cepstrum pitch test(s) failed\n", failures);
    return 1;
}
