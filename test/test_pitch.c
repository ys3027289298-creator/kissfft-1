/*
 *  Copyright (c) 2003-2026, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/* Tests for the spectral peak detector and harmonic tracker:
 *  1. a 440 Hz tone is reported within one cent (in fact far inside it)
 *  2. harmonic tones report their fundamental, not a harmonic
 *  3. an established f0 does not jump an octave when the fundamental is weak
 *     or missing from the spectrum, while a real octave change is followed
 *  4. every detected peak carries a harmonic number and a stable track id
 *  5. noise and silence are not reported as a pitched frame
 *  6. the caller-supplied-buffer allocation path and the setters behave
 *  7. a frame's result is the same whether the caller asks for peaks or not
 *
 * Fixed-point builds are exercised too, but with a looser accuracy bound:
 * the spectrum is quantised there, so sub-cent interpolation is impossible.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kiss_pitch.h"

#define PI 3.14159265358979323846264338327950288

enum { NFFT = 4096 };
static const double SAMPLE_RATE = 44100.0;

static int failures = 0;

static void fail(const char *what)
{
    printf("FAIL: %s\n", what);
    ++failures;
}

#ifndef USE_SIMD
#define KISS_PITCH_TEST_SCALAR 1
#endif

#ifdef KISS_PITCH_TEST_SCALAR
#if defined(FIXED_POINT)
static const double ACCURACY_CENTS = 12.0; /* quantised spectrum: about a bin */
#else
static const double ACCURACY_CENTS = 1.0;
#endif

static double cents(double f, double ref)
{
    if (f <= 0.0 || ref <= 0.0)
        return 1e9;
    return 1200.0 * log(f / ref) / log(2.0);
}

/* Sum of cosines with the given amplitudes; phase varies per frame and
 * partial so consecutive frames are never accidentally identical. */
static void harmonic_frame(kiss_fft_scalar *x, double f0, const double *amps, int n,
                           int frame_no)
{
    int i, h;
    double scale = 1.0;

    /* fixed-point input has to stay well inside its range */
#if defined(FIXED_POINT)
    scale = (double)(((kiss_fft_scalar)1 << (sizeof(kiss_fft_scalar) * 8 - 3)));
#endif
    for (i = 0; i < NFFT; ++i) {
        double t = i / SAMPLE_RATE;
        double v = 0.0;
        for (h = 0; h < n; ++h) {
            double phase = 0.37 * frame_no * (h + 1) + 0.23 * h;
            v += amps[h] * cos(2.0 * PI * f0 * (h + 1) * t + phase);
        }
        x[i] = (kiss_fft_scalar)(scale * v / (double)n);
    }
}

static void sine_frame(kiss_fft_scalar *x, double f, int frame_no)
{
    double amps[1];
    amps[0] = 1.0;
    harmonic_frame(x, f, amps, 1, frame_no);
}

static void silent_frame(kiss_fft_scalar *x)
{
    memset(x, 0, sizeof(kiss_fft_scalar) * NFFT);
}
/* Process one frame and return the voiced result. */
static int run_frame(kiss_pitch_cfg cfg, const kiss_fft_scalar *x,
                     kiss_pitch_frame *out, kiss_pitch_peak *peaks, int max_peaks)
{
    memset(out, 0, sizeof(*out));
    return kiss_pitch_process(cfg, x, out, peaks, max_peaks);
}

/* ---- 1. absolute accuracy ------------------------------------------------ */

static void test_single_tone_accuracy(void)
{
    static const double tones[] = { 110.0, 220.0, 440.0, 880.0, 1000.0, 1760.0 };
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    size_t t;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    for (t = 0; t < sizeof(tones) / sizeof(tones[0]); ++t) {
        int frame;
        double worst = 0.0;

        kiss_pitch_reset(cfg);
        for (frame = 0; frame < 8; ++frame) {
            double err;

            sine_frame(x, tones[t], frame);
            if (!run_frame(cfg, x, &out, NULL, 0)) {
                printf("(%g Hz frame %d not voiced, confidence %g)\n",
                       tones[t], frame, out.confidence);
                fail("a clean sine should be voiced");
                break;
            }
            err = fabs(cents(out.f0_hz, tones[t]));
            if (err > worst)
                worst = err;
        }
        if (worst > ACCURACY_CENTS) {
            printf("(%g Hz: worst error %.4f cents, limit %.1f)\n",
                   tones[t], worst, ACCURACY_CENTS);
            fail("tone frequency outside the accuracy bound");
        }
    }

    free(x);
    kiss_pitch_free(cfg);
}

/* 440 Hz specifically, as required: within one cent even when the frame
 * phase changes, which moves the peak between bins. */
static void test_440_exact(void)
{
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    int frame;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    kiss_pitch_reset(cfg);
    for (frame = 0; frame < 40; ++frame) {
        double err;
        /* frame 0 starts at phase 0, later frames at a different phase */
        sine_frame(x, 440.0, frame);
        if (!run_frame(cfg, x, &out, NULL, 0)) {
            fail("440 Hz frame not voiced");
            break;
        }
        err = fabs(cents(out.f0_hz, 440.0));
        if (err > ACCURACY_CENTS) {
            printf("(440 Hz frame %d: error %.5f cents)\n", frame, err);
            fail("440 Hz outside one cent");
            break;
        }
    }

    free(x);
    kiss_pitch_free(cfg);
}

/* ---- 2. harmonic tones report the fundamental ---------------------------- */

static void test_harmonic_tones(void)
{
    static const double fund_amps[] = { 1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.08 };
    static const double missing_fund[] = { 0.0, 0.9, 0.7, 0.5, 0.4, 0.3 };
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    int frame;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    for (frame = 0; frame < 6; ++frame) {
        harmonic_frame(x, 220.0, fund_amps, 8, frame);
        if (!run_frame(cfg, x, &out, NULL, 0) ||
            fabs(cents(out.f0_hz, 220.0)) > ACCURACY_CENTS) {
            printf("(harmonic 220 Hz: got %g Hz)\n", out.f0_hz);
            fail("harmonic tone did not report f0 = 220 Hz");
            break;
        }
    }

    /* the fundamental missing from the spectrum: the partials still imply it */
    for (frame = 0; frame < 6; ++frame) {
        harmonic_frame(x, 220.0, missing_fund, 6, frame);
        if (!run_frame(cfg, x, &out, NULL, 0) ||
            fabs(cents(out.f0_hz, 220.0)) > ACCURACY_CENTS) {
            printf("(missing fundamental 220 Hz: got %g Hz)\n", out.f0_hz);
            fail("missing fundamental did not report f0 = 220 Hz");
            break;
        }
    }

    free(x);
    kiss_pitch_free(cfg);
}

/* ---- 3. octave stability ------------------------------------------------- */

static void test_octave_stability(void)
{
    static const double normal[] = { 1.0, 0.8, 0.6, 0.5, 0.4, 0.3 };
    static const double weak_fund[] = { 0.02, 1.0, 0.1, 0.9, 0.1, 0.8 };
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    int frame, jumped = 0;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    /* establish 440 Hz on a well-behaved harmonic tone ... */
    for (frame = 0; frame < 8; ++frame) {
        harmonic_frame(x, 440.0, normal, 6, frame);
        if (!run_frame(cfg, x, &out, NULL, 0) ||
            fabs(cents(out.f0_hz, 440.0)) > ACCURACY_CENTS) {
            fail("could not establish 440 Hz");
            break;
        }
    }
    /* ... then hide the fundamental. A naive peak picker would report 880. */
    for (frame = 0; frame < 10; ++frame) {
        harmonic_frame(x, 440.0, weak_fund, 6, frame);
        if (run_frame(cfg, x, &out, NULL, 0) &&
            fabs(cents(out.f0_hz, 440.0)) > 100.0) {
            ++jumped;
        }
    }
    if (jumped) {
        printf("(%d of 10 weak-fundamental frames jumped an octave)\n", jumped);
        fail("established pitch jumped an octave");
    }

    free(x);
    kiss_pitch_free(cfg);
}

static void test_real_octave_change_is_followed(void)
{
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    int frame, held = 0;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    for (frame = 0; frame < 5; ++frame) {
        sine_frame(x, 440.0, frame);
        (void)run_frame(cfg, x, &out, NULL, 0);
    }
    /* a sustained jump to 880 must be followed within a few frames */
    for (frame = 5; frame < 15; ++frame) {
        sine_frame(x, 880.0, frame);
        if (run_frame(cfg, x, &out, NULL, 0) &&
            fabs(cents(out.f0_hz, 880.0)) <= ACCURACY_CENTS)
            ++held;
    }
    if (held == 0) {
        printf("(could not follow 440 -> 880, ended at %g Hz)\n", out.f0_hz);
        fail("a real octave change was not followed");
    }

    free(x);
    kiss_pitch_free(cfg);
}

/* A single frame that happens to favour the octave above must not move the
 * track. This is the "does not flip octaves" guarantee on its own. */
static void test_single_ambiguous_frame_does_not_flip(void)
{
    static const double fundamental_only[] = { 1.0 };
    static const double octave_favoured[] = { 0.02, 1.0, 0.1, 0.9 };
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    int frame;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    for (frame = 0; frame < 5; ++frame) {
        harmonic_frame(x, 440.0, fundamental_only, 1, frame);
        (void)run_frame(cfg, x, &out, NULL, 0);
    }
    harmonic_frame(x, 440.0, octave_favoured, 4, 0);
    if (!run_frame(cfg, x, &out, NULL, 0) ||
        fabs(cents(out.f0_hz, 440.0)) > ACCURACY_CENTS) {
        printf("(one ambiguous frame moved f0 to %g Hz)\n", out.f0_hz);
        fail("one ambiguous frame flipped the octave");
    }

    free(x);
    kiss_pitch_free(cfg);
}

/* ---- 4. peaks, harmonic numbers and tracks ------------------------------- */

static void test_peak_annotations(void)
{
    static const double amps[] = { 1.0, 0.8, 0.6, 0.4 };
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    kiss_pitch_peak peaks[16];
    int i, n_harmonics;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    harmonic_frame(x, 440.0, amps, 4, 0);
    if (!run_frame(cfg, x, &out, peaks, 16)) {
        fail("harmonic frame not voiced");
        free(x);
        kiss_pitch_free(cfg);
        return;
    }
    if (out.n_peaks < 4) {
        printf("(only %d peaks detected)\n", out.n_peaks);
        fail("expected at least four peaks");
    }
    n_harmonics = 0;
    for (i = 0; i < out.n_peaks; ++i) {
        if (peaks[i].harmonic > 0) {
            ++n_harmonics;
            if (fabs(cents(peaks[i].freq_hz, 440.0 * peaks[i].harmonic)) > ACCURACY_CENTS) {
                printf("(peak %d at %g Hz labelled harmonic %d)\n",
                       i, peaks[i].freq_hz, peaks[i].harmonic);
                fail("harmonic label does not match the peak frequency");
            }
        }
    }
    if (n_harmonics < 4) {
        printf("(only %d peaks labelled with a harmonic number)\n", n_harmonics);
        fail("expected every partial to carry a harmonic number");
    }
    /* the highest partial must be labelled too: its ratio to f0 sits right on
     * an integer and must not be truncated away */
    {
        int highest = 0;
        for (i = 0; i < out.n_peaks; ++i)
            if (peaks[i].harmonic > highest)
                highest = peaks[i].harmonic;
        if (highest != 4) {
            printf("(highest partial labelled %d, expected 4)\n", highest);
            fail("the highest partial was not assigned to its harmonic");
        }
    }
    for (i = 1; i < out.n_peaks; ++i) {
        if (!(peaks[i].freq_hz > peaks[i - 1].freq_hz))
            fail("peaks are not sorted by increasing frequency");
    }

    /* the same partials must keep their track id and age across frames */
    {
        int ids_before[16], n_before = out.n_peaks;
        for (i = 0; i < n_before; ++i)
            ids_before[i] = peaks[i].track_id;
        harmonic_frame(x, 440.0, amps, 4, 1);
        if (!run_frame(cfg, x, &out, peaks, 16))
            fail("second harmonic frame not voiced");
        if (out.n_peaks != n_before)
            fail("peak count changed between identical frames");
        for (i = 0; i < out.n_peaks && i < n_before; ++i) {
            if (peaks[i].track_id != ids_before[i]) {
                printf("(peak %d changed track %d -> %d)\n",
                       i, ids_before[i], peaks[i].track_id);
                fail("track id not stable across frames");
            }
            if (peaks[i].track_age != 2) {
                printf("(peak %d age %d after two frames)\n", i, peaks[i].track_age);
                fail("track age did not advance");
            }
        }
    }

    free(x);
    kiss_pitch_free(cfg);
}

static void test_max_peaks_truncates(void)
{
    static const double amps[] = { 1.0, 0.8, 0.6, 0.4, 0.3 };
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    kiss_pitch_peak peaks[2];

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    harmonic_frame(x, 440.0, amps, 5, 0);
    memset(peaks, 0, sizeof(peaks));
    (void)run_frame(cfg, x, &out, peaks, 2);
    if (out.n_peaks != 2)
        fail("n_peaks does not report the number written to a full array");

    /* asking for no peaks must not change the frame result */
    {
        kiss_pitch_frame without;
        (void)run_frame(cfg, x, &without, NULL, 0);
        if (without.voiced != out.voiced ||
            fabs(without.f0_hz / out.f0_hz - 1.0) > 1e-12 ||
            without.confidence != out.confidence)
            fail("frame result depends on whether peaks were requested");
    }

    free(x);
    kiss_pitch_free(cfg);
}

/* ---- 5. noise and silence ------------------------------------------------ */

static void test_noise_is_not_pitched(void)
{
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    unsigned seed;
    int voiced_count = 0;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    for (seed = 1; seed <= 20; ++seed) {
        int i;
        unsigned state = seed * 2654435761u + 1u;
        for (i = 0; i < NFFT; ++i) {
            /* cheap deterministic white noise, roughly unit RMS */
            state = state * 1103515245u + 12345u;
            x[i] = (kiss_fft_scalar)(((double)((state >> 8) & 0xffff) / 32768.0 - 1.0) * 0.5);
        }
        kiss_pitch_reset(cfg);
        if (run_frame(cfg, x, &out, NULL, 0))
            ++voiced_count;
    }
    if (voiced_count > 1) {
        printf("(%d of 20 noise frames reported as voiced)\n", voiced_count);
        fail("white noise is being reported as pitched");
    }

    free(x);
    kiss_pitch_free(cfg);
}

static void test_silence_is_not_pitched(void)
{
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    silent_frame(x);
    if (run_frame(cfg, x, &out, NULL, 0)) {
        fail("silence reported as voiced");
    } else {
        if (out.f0_hz != 0.0 || out.confidence != 0.0 || out.n_peaks != 0)
            fail("unvoiced frame did not clear its outputs");
        if (out.rms != 0.0)
            fail("silent frame reports a non-zero RMS");
    }

    free(x);
    kiss_pitch_free(cfg);
}

static void test_silence_survives_gap(void)
{
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;
    int i;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    sine_frame(x, 440.0, 0);
    (void)run_frame(cfg, x, &out, NULL, 0);
    /* long silence must clear the established pitch */
    silent_frame(x);
    for (i = 0; i < 10; ++i)
        (void)run_frame(cfg, x, &out, NULL, 0);
    sine_frame(x, 880.0, 0);
    if (!run_frame(cfg, x, &out, NULL, 0) ||
        fabs(cents(out.f0_hz, 880.0)) > ACCURACY_CENTS) {
        printf("(after silence the new pitch was reported as %g Hz)\n", out.f0_hz);
        fail("the established pitch survived a long silence");
    }

    free(x);
    kiss_pitch_free(cfg);
}

#endif /* KISS_PITCH_TEST_SCALAR */

/* ---- 6. allocation and configuration ------------------------------------- */

static void test_alloc_and_setters(void)
{
    kiss_pitch_cfg cfg;
    size_t len = 0, too_small = 0;
    void *buf;

    /* size query then caller-supplied buffer */
    cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, &len);
    if (cfg != NULL) {
        fail("size query allocated without lenmem == NULL");
        kiss_pitch_free(cfg);
    }
    if (len == 0) {
        fail("size query did not report a size");
        return;
    }
    buf = malloc(len);
    if (buf == NULL) {
        fail("out of memory");
        return;
    }
    {
        size_t have = len;
        cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, buf, &have);
        if (cfg != (kiss_pitch_cfg)buf)
            fail("caller-supplied buffer was not used");
        if (have != len)
            fail("caller-supplied path changed the reported size");
        /* the caller owns the buffer in this path; do not free it here */
    }
    /* a buffer that is one byte too small must be refused, not overflowed */
    if (len > 1) {
        too_small = len - 1;
        cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, buf, &too_small);
        if (cfg != NULL)
            fail("kiss_pitch_alloc accepted an undersized buffer");
        if (too_small != len)
            fail("size query did not report the size needed");
    }
    free(buf);

    /* invalid arguments */
    if (kiss_pitch_alloc(0, SAMPLE_RATE, NULL, NULL) != NULL)
        fail("kiss_pitch_alloc accepted nfft == 0");
    if (kiss_pitch_alloc(NFFT + 1, SAMPLE_RATE, NULL, NULL) != NULL)
        fail("kiss_pitch_alloc accepted an odd nfft");
    if (kiss_pitch_alloc(NFFT, 0.0, NULL, NULL) != NULL)
        fail("kiss_pitch_alloc accepted a zero sample rate");

    cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }

    /* setters validate their arguments and otherwise take effect */
    if (kiss_pitch_set_range(cfg, 100.0, 90.0))
        fail("set_range accepted fmax <= fmin");
    if (kiss_pitch_set_range(cfg, -1.0, 1000.0))
        fail("set_range accepted a negative fmin");
    if (kiss_pitch_set_range(cfg, 0.5, 1000.0))
        fail("set_range accepted a range outside the FFT bins");
    if (!kiss_pitch_set_range(cfg, 200.0, 800.0))
        fail("set_range rejected a valid range");
    if (kiss_pitch_set_max_harmonics(cfg, 0))
        fail("set_max_harmonics accepted 0");
    if (kiss_pitch_set_max_harmonics(cfg, 65))
        fail("set_max_harmonics accepted an out-of-range value");
    if (!kiss_pitch_set_max_harmonics(cfg, 8))
        fail("set_max_harmonics rejected a valid value");
    kiss_pitch_set_detection(cfg, -80.0, 0.3);
    kiss_pitch_reset(cfg);
    kiss_pitch_free(cfg);
}

#ifdef KISS_PITCH_TEST_SCALAR
/* The narrowed search range must be honoured: a 1000 Hz tone is outside it. */
static void test_range_is_honoured(void)
{
    kiss_pitch_cfg cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    kiss_fft_scalar *x;
    kiss_pitch_frame out;

    if (cfg == NULL) {
        fail("kiss_pitch_alloc returned NULL");
        return;
    }
    x = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * NFFT);
    if (x == NULL) {
        fail("out of memory");
        kiss_pitch_free(cfg);
        return;
    }

    if (!kiss_pitch_set_range(cfg, 200.0, 800.0))
        fail("set_range rejected a valid range");
    sine_frame(x, 1000.0, 0);
    if (run_frame(cfg, x, &out, NULL, 0) &&
        fabs(cents(out.f0_hz, 1000.0)) <= ACCURACY_CENTS)
        fail("a tone outside the search range was reported anyway");

    free(x);
    kiss_pitch_free(cfg);
}

#endif /* KISS_PITCH_TEST_SCALAR */

/* ---- 7. build availability ----------------------------------------------- */

static void test_available(void)
{
    kiss_pitch_cfg cfg;
#if defined(USE_SIMD)
    if (kiss_pitch_available())
        fail("kiss_pitch_available() is true in a SIMD build");
    cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    if (cfg != NULL)
        fail("kiss_pitch_alloc succeeded in a SIMD build");
#else
    if (!kiss_pitch_available())
        fail("kiss_pitch_available() is false in a scalar build");
    cfg = kiss_pitch_alloc(NFFT, SAMPLE_RATE, NULL, NULL);
    if (cfg == NULL)
        fail("kiss_pitch_alloc failed in a scalar build");
    kiss_pitch_free(cfg);
#endif
}

int main(void)
{
    /* the analysis needs even frames; keep a clear message if the build
     * cannot do it at all */
    if (!kiss_pitch_available()) {
        printf("SKIP: spectral analysis is not available in this build\n");
        return 0;
    }

    test_available();
    test_alloc_and_setters();
#ifdef KISS_PITCH_TEST_SCALAR
    test_single_tone_accuracy();
    test_440_exact();
    test_harmonic_tones();
    test_octave_stability();
    test_real_octave_change_is_followed();
    test_single_ambiguous_frame_does_not_flip();
    test_peak_annotations();
    test_max_peaks_truncates();
    test_noise_is_not_pitched();
    test_silence_is_not_pitched();
    test_silence_survives_gap();
    test_range_is_honoured();
#endif

    printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
