/*
 *  Copyright (c) 2003-2026, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Self-contained test for the kiss_tuner pitch-estimation pipeline
 * (spectral peak -> parabolic interpolation -> median smoothing ->
 * octave correction). Generates its own small synthetic samples and
 * runs everything in one go.
 *
 * Usage: tunertest [median_window]   (default 5)
 *
 * "before" = naive peak-bin picking on the same windowed spectrum
 *            (no interpolation, no smoothing, no octave correction)
 * "after"  = full kiss_tuner pipeline
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kiss_fftr.h"
#include "kiss_tuner.h"

#define FS 8000
#define NFFT 1024
#define NFRAMES 16

static unsigned int s_rand_state = 12345;

static float frand(void) /* deterministic uniform noise in [-1,1] */
{
    s_rand_state = s_rand_state * 1664525u + 1013904223u;
    return ((s_rand_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

/* "before" estimate: plain peak-bin pick, no interpolation/smoothing */
static float naive_peak_freq(kiss_fftr_cfg fftr, const kiss_fft_scalar *frame,
                             kiss_fft_scalar *windowed, kiss_fft_cpx *spectrum,
                             const float *hann)
{
    int i, kpeak;
    float mag, best;
    double fmin_bin = ceil(50.0 * NFFT / FS);
    double fmax_bin = floor(2000.0 * NFFT / FS);
    int kmin = (int) fmin_bin;
    int kmax = (int) fmax_bin;

    for (i = 0; i < NFFT; ++i)
        windowed[i] = (kiss_fft_scalar) (frame[i] * hann[i]);
    kiss_fftr(fftr, windowed, spectrum);

    kpeak = kmin;
    best = 0.0f;
    for (i = kmin; i <= kmax; ++i) {
        mag = (float) hypot((double) spectrum[i].r, (double) spectrum[i].i);
        if (mag > best) {
            best = mag;
            kpeak = i;
        }
    }
    return (float) kpeak * FS / NFFT;
}

static int is_octave_error(float est, float truth)
{
    float r = est / truth;
    return (r > 1.9f && r < 2.1f) || (r > 0.475f && r < 0.525f);
}

typedef struct {
    double sum_abs_err;
    float max_abs_err;
    int octave_errors;
    int n;
} err_stats;

static void stats_update(err_stats *s, float est, float truth)
{
    float e = fabsf(est - truth);
    s->sum_abs_err += e;
    if (e > s->max_abs_err)
        s->max_abs_err = e;
    if (is_octave_error(est, truth))
        ++s->octave_errors;
    ++s->n;
}

static void stats_print(const char *tag, const err_stats *s)
{
    printf("    %-7s mean|err|=%.4f Hz  max|err|=%.4f Hz  octave_errors=%d/%d\n",
           tag, s->sum_abs_err / s->n, s->max_abs_err, s->octave_errors, s->n);
}

/*
 * Run one voiced signal through the before/after estimators.
 * If corrupt_frame >= 0, that frame is replaced by a loud noise burst
 * (simulates a transient that throws off a single raw estimate).
 * Returns number of pipeline octave errors.
 */
static int run_signal(const char *name, float true_f0, int harmonic_mode, int corrupt_frame, int median_window)
{
    kiss_tuner_cfg tuner;
    kiss_fftr_cfg fftr;
    kiss_fft_scalar frame[NFFT];
    kiss_fft_scalar windowed[NFFT];
    kiss_fft_cpx spectrum[NFFT / 2 + 1];
    float hann[NFFT];
    err_stats before = {0.0, 0.0f, 0, 0};
    err_stats after = {0.0, 0.0f, 0, 0};
    int i, n;

    tuner = kiss_tuner_alloc(NFFT, FS, median_window, NULL, NULL);
    fftr = kiss_fftr_alloc(NFFT, 0, NULL, NULL);
    if (!tuner || !fftr) {
        fprintf(stderr, "alloc failed\n");
        exit(2);
    }
    for (i = 0; i < NFFT; ++i)
        hann[i] = (float) (0.5 - 0.5 * cos(2.0 * 3.14159265358979323846 * i / (NFFT - 1)));

    printf("  signal \"%s\" true_f0=%.1f Hz\n", name, true_f0);
    for (n = 0; n < NFRAMES; ++n) {
        float f_before, f_after;
        kiss_tuner_status status;
        for (i = 0; i < NFFT; ++i) {
            double t = (double) (n * NFFT + i) / FS;
            double ph = 2.0 * 3.14159265358979323846 * true_f0 * t;
            double v;
            if (n == corrupt_frame)
                v = 0.5 * frand(); /* loud transient noise burst */
            else if (harmonic_mode)
                /* octave trap: 2nd harmonic louder than the fundamental */
                v = 0.6 * sin(ph) + 1.0 * sin(2.0 * ph);
            else
                v = 0.8 * sin(ph);
            v += 0.005 * frand();
            frame[i] = (kiss_fft_scalar) v;
        }
        f_before = naive_peak_freq(fftr, frame, windowed, spectrum, hann);
        status = kiss_tuner_process(tuner, frame, &f_after);
        if (status != KISS_TUNER_OK) {
            fprintf(stderr, "    frame %d: unexpected status %d on voiced frame\n", n, (int) status);
            exit(2);
        }
        printf("    frame %2d: before=%8.3f Hz  after=%8.3f Hz\n", n, f_before, f_after);
        stats_update(&before, f_before, true_f0);
        stats_update(&after, f_after, true_f0);
    }
    stats_print("before:", &before);
    stats_print("after:", &after);

    kiss_tuner_free(tuner);
    free(fftr);
    return after.octave_errors;
}

static int run_silence(int median_window)
{
    kiss_tuner_cfg tuner = kiss_tuner_alloc(NFFT, FS, median_window, NULL, NULL);
    kiss_fft_scalar frame[NFFT];
    int n, bad = 0;

    printf("  signal \"silence\" (expect KISS_TUNER_SILENCE, f0=0)\n");
    memset(frame, 0, sizeof(frame));
    for (n = 0; n < 4; ++n) {
        float f0 = -1.0f;
        kiss_tuner_status status = kiss_tuner_process(tuner, frame, &f0);
        printf("    frame %2d: status=KISS_TUNER_SILENCE(%d) f0=%.1f\n", n, (int) status, f0);
        if (status != KISS_TUNER_SILENCE || f0 != 0.0f)
            ++bad;
    }
    kiss_tuner_free(tuner);
    return bad;
}

static int run_failure_paths(void)
{
    int bad = 0;
    printf("  failure paths\n");
    if (kiss_tuner_alloc(NFFT - 1, FS, 5, NULL, NULL) != NULL) { /* odd nfft */
        printf("    FAIL: odd nfft accepted\n");
        ++bad;
    } else {
        printf("    odd nfft rejected (NULL) as expected\n");
    }
    if (kiss_tuner_alloc(NFFT, 0, 5, NULL, NULL) != NULL) { /* bad sample rate */
        printf("    FAIL: sample_rate=0 accepted\n");
        ++bad;
    } else {
        printf("    sample_rate=0 rejected (NULL) as expected\n");
    }
    if (kiss_tuner_process(NULL, NULL, NULL) != KISS_TUNER_SILENCE) {
        printf("    FAIL: NULL args not handled\n");
        ++bad;
    } else {
        printf("    NULL args handled (KISS_TUNER_SILENCE) as expected\n");
    }
    return bad;
}

int main(int argc, char **argv)
{
    int median_window = 5;
    int failures = 0;

    if (argc > 1)
        median_window = atoi(argv[1]);
    printf("kiss_tuner pipeline test: nfft=%d fs=%d median_window=%d (clamped to odd in [1,63])\n",
           NFFT, FS, median_window);

    printf("[1] pure tones\n");
    failures += run_signal("sine-220", 220.0f, 0, -1, median_window);
    failures += run_signal("sine-440", 440.0f, 0, -1, median_window);
    failures += run_signal("sine-880", 880.0f, 0, -1, median_window);

    printf("[2] octave trap (fundamental 220 Hz, stronger 2nd harmonic)\n");
    failures += run_signal("octave-trap-220", 220.0f, 1, -1, median_window);

    printf("[3] transient noise burst on frame 8 of a 440 Hz tone (median smoothing)\n");
    {
        /* With a median window > 1 the corrupted frame's raw estimate is
         * rejected; with window 1 it passes straight through. Only count
         * it as a failure when smoothing was actually requested. */
        int transient_errors = run_signal("sine-440-transient", 440.0f, 0, 8, median_window);
        if (median_window > 1)
            failures += transient_errors;
    }

    printf("[4] silence frames\n");
    failures += run_silence(median_window);

    printf("[5] failure paths\n");
    failures += run_failure_paths();

    if (failures) {
        printf("RESULT: FAIL (%d problem(s))\n", failures);
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
