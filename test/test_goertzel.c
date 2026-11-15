/*
 *  Self-contained test for the keypress tone detector
 *  (framing + multi-Goertzel + peak decision, see kiss_goertzel.h).
 *
 *  Builds a small synthetic 8 kHz stream:
 *    silence -> white noise -> DTMF '1' (697+1209 Hz)
 *    -> white noise -> DTMF '9' (852+1477 Hz) -> silence
 *
 *  Checks:
 *    1. tone frames report the right frequencies with numeric confidence
 *    2. noise and silence frames are rejected
 *    3. Goertzel power matches the corresponding full-FFT (kiss_fftr)
 *       bin within the documented error bound
 *    4. invalid parameters are reported as errors
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kiss_goertzel.h"
#include "kiss_fftr.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define FS 8000.0f
#define NFR 256            /* frame length */
#define HOP 128            /* hop */
#define NSAMP 16000        /* 2 s stream */
#define MAXFRAMES ((NSAMP - NFR) / HOP + 1)

static kiss_fft_scalar sig[NSAMP];
static kiss_goertzel_frame_result res[MAXFRAMES];

static int nfail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { ++nfail; printf("FAIL: "); printf(__VA_ARGS__); \
        printf("  [%s:%d]\n", __FILE__, __LINE__); } \
} while (0)

/* deterministic white noise, no external deps */
static unsigned lcg = 12345u;
static double frand(void)
{
    lcg = lcg * 1664525u + 1013904223u;
    return ((double)(lcg >> 8) / (double)(1u << 24)) * 2.0 - 1.0;
}

static void add_sine(int from, int to, double freq, double amp)
{
    int i;
    for (i = from; i < to; ++i)
        sig[i] += (kiss_fft_scalar)(amp * sin(2.0 * M_PI * freq * (double)i / FS));
}

static void add_noise(int from, int to, double amp)
{
    int i;
    for (i = from; i < to; ++i)
        sig[i] += (kiss_fft_scalar)(amp * frand());
}

static void build_signal(void)
{
    memset(sig, 0, sizeof(sig));
    /* 0     .. 2000 : silence                                        */
    add_noise(2000, 4000, 0.4);            /* broadband noise         */
    add_sine(4000, 8000, 697.0, 0.4);      /* DTMF '1'                */
    add_sine(4000, 8000, 1209.0, 0.4);
    add_noise(8000, 10000, 0.4);           /* broadband noise         */
    add_sine(10000, 14000, 852.0, 0.4);    /* DTMF '9'                */
    add_sine(10000, 14000, 1477.0, 0.4);
    /* 14000 .. 16000: silence                                        */
}

/* segment a sample offset belongs to: 0 silence, 1 noise, 2 '1', 3 '9' */
static int segment_of(int off)
{
    if (off < 2000 || off >= 14000) return 0;
    if (off < 4000) return 1;
    if (off < 8000) return 2;
    if (off < 10000) return 1;
    return 3;
}

static int has_det(const kiss_goertzel_frame_result *r, float freq)
{
    int i;
    for (i = 0; i < r->n_det; ++i)
        if (fabsf(r->det_freq[i] - freq) < 0.5f)
            return 1;
    return 0;
}

static double conf_of(const kiss_goertzel_frame_result *r, float freq)
{
    int i;
    for (i = 0; i < r->n_det; ++i)
        if (fabsf(r->det_freq[i] - freq) < 0.5f)
            return r->det_conf[i];
    return 0.0;
}

static void test_detection(void)
{
    static const float dtmf[] = { 697, 770, 852, 941, 1209, 1336, 1477 };
    kiss_goertzel_cfg cfg;
    int nframes, i;
    int n_tone1 = 0, n_tone9 = 0, n_sil = 0, n_noise = 0;

    cfg = kiss_goertzel_alloc(NFR, HOP, FS, dtmf, 7, 0.2f, -50.0f);
    CHECK(cfg != NULL, "alloc with valid params failed");
    if (!cfg)
        return;

    nframes = kiss_goertzel_process(cfg, sig, NSAMP, res, MAXFRAMES);
    CHECK(nframes == MAXFRAMES, "expected %d frames, got %d", MAXFRAMES, nframes);

    printf("--- per-frame detections (non-empty frames only) ---\n");
    for (i = 0; i < nframes; ++i) {
        int seg = segment_of(res[i].sample_offset);
        int interior = res[i].sample_offset % HOP == 0 &&
                       segment_of(res[i].sample_offset) ==
                       segment_of(res[i].sample_offset + NFR - 1);

        if (res[i].n_det > 0 || res[i].is_silence) {
            int j;
            printf("frame %3d @%5d: ", res[i].frame_index, res[i].sample_offset);
            if (res[i].is_silence)
                printf("silence");
            for (j = 0; j < res[i].n_det; ++j)
                printf(" %7.1fHz conf=%.3f", res[i].det_freq[j], res[i].det_conf[j]);
            printf("\n");
        }

        if (!interior)
            continue; /* frames straddling a boundary: no assertion */

        if (seg == 0) {
            CHECK(res[i].is_silence && res[i].n_det == 0,
                  "silence frame @%d not rejected", res[i].sample_offset);
            ++n_sil;
        } else if (seg == 1) {
            CHECK(res[i].n_det == 0,
                  "noise frame @%d produced %d false detections",
                  res[i].sample_offset, res[i].n_det);
            ++n_noise;
        } else if (seg == 2) {
            CHECK(res[i].n_det == 2 && has_det(&res[i], 697) &&
                  has_det(&res[i], 1209),
                  "DTMF '1' frame @%d: n_det=%d", res[i].sample_offset, res[i].n_det);
            CHECK(conf_of(&res[i], 697) > 0.35 && conf_of(&res[i], 1209) > 0.35,
                  "DTMF '1' confidence too low @%d", res[i].sample_offset);
            ++n_tone1;
        } else {
            CHECK(res[i].n_det == 2 && has_det(&res[i], 852) &&
                  has_det(&res[i], 1477),
                  "DTMF '9' frame @%d: n_det=%d", res[i].sample_offset, res[i].n_det);
            ++n_tone9;
        }
    }
    printf("interior frames: tone1=%d tone9=%d noise=%d silence=%d\n",
           n_tone1, n_tone9, n_noise, n_sil);
    CHECK(n_tone1 > 20 && n_tone9 > 20, "too few tone frames classified");
    CHECK(n_noise > 10 && n_sil > 10, "too few noise/silence frames classified");

    kiss_goertzel_free(cfg);
}

static void test_error_bound(void)
{
    /* on-bin target freqs (bin spacing 8000/256 = 31.25 Hz) so the
       Goertzel coefficient and the FFT bin are the same DFT coefficient */
    static const float onbin[] = { 687.5f, 843.75f, 1218.75f, 1468.75f };
    static const int bins[] = { 22, 27, 39, 47 };
    kiss_goertzel_cfg cfg;
    kiss_fftr_cfg fcfg;
    kiss_fft_cpx fout[NFR / 2 + 1];
    int nframes, i, t;
    double max_meas = 0.0, max_bound = 0.0;

    cfg = kiss_goertzel_alloc(NFR, HOP, FS, onbin, 4, 0.2f, -50.0f);
    fcfg = kiss_fftr_alloc(NFR, 0, NULL, NULL);
    CHECK(cfg != NULL && fcfg != NULL, "alloc failed in error-bound test");
    if (!cfg || !fcfg)
        return;

    nframes = kiss_goertzel_process(cfg, sig, NSAMP, res, MAXFRAMES);
    for (i = 0; i < nframes; ++i) {
        const kiss_fft_scalar *frame = sig + res[i].sample_offset;
        double sum_abs = 0.0, bound, delta;
        int j;
        for (j = 0; j < NFR; ++j)
            sum_abs += fabs((double)frame[j]);
        bound = kiss_goertzel_error_bound(cfg, sum_abs);
        CHECK(bound >= 0.0, "error bound must be non-negative");
        if (sum_abs > 0.0)
            CHECK(bound > 0.0, "error bound must be positive for non-zero frames");

        kiss_fftr(fcfg, frame, fout);
        for (t = 0; t < 4; ++t) {
            double p_g = kiss_goertzel_tone_power(cfg, frame, t);
            double p_f = (double)fout[bins[t]].r * fout[bins[t]].r +
                         (double)fout[bins[t]].i * fout[bins[t]].i;
            double pbound = bound * (2.0 * sqrt(p_f) + bound);
            delta = fabs(p_g - p_f);
            if (delta > max_meas) max_meas = delta;
            if (pbound > max_bound) max_bound = pbound;
            CHECK(delta <= pbound,
                  "frame %d tone %d: |dP|=%.3e > bound=%.3e", i, t, delta, pbound);
        }
    }
    printf("--- Goertzel vs full-FFT bin (power) ---\n");
    printf("frames=%d tones/frame=4  max measured |dP| = %.6e\n", nframes, max_meas);
    printf("documented upper bound  max |dP| = %.6e  (margin x%.1f)\n",
           max_bound, max_meas > 0 ? max_bound / max_meas : -1.0);

    kiss_goertzel_free(cfg);
    kiss_fftr_free(fcfg);
}

static void test_invalid_params(void)
{
    static const float ok[] = { 697, 1209 };
    static const float bad_zero[] = { 0.0f, 1209 };
    static const float bad_nyq[] = { 697, 4000 };   /* == fs/2 */
    static const float bad_many[17] = { 0 };
    kiss_goertzel_cfg c;

    CHECK(kiss_goertzel_alloc(0, HOP, FS, ok, 2, 0.2f, -50.0f) == NULL,
          "frame_len=0 accepted");
    CHECK(kiss_goertzel_alloc(4, HOP, FS, ok, 2, 0.2f, -50.0f) == NULL,
          "frame_len=4 accepted");
    CHECK(kiss_goertzel_alloc(NFR, 0, FS, ok, 2, 0.2f, -50.0f) == NULL,
          "hop=0 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, 0.0f, ok, 2, 0.2f, -50.0f) == NULL,
          "sample_rate=0 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, -FS, ok, 2, 0.2f, -50.0f) == NULL,
          "negative sample_rate accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, NULL, 2, 0.2f, -50.0f) == NULL,
          "freqs=NULL accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, ok, 0, 0.2f, -50.0f) == NULL,
          "nfreqs=0 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, bad_many, 17, 0.2f, -50.0f) == NULL,
          "nfreqs>MAX accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, bad_zero, 2, 0.2f, -50.0f) == NULL,
          "freq=0 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, bad_nyq, 2, 0.2f, -50.0f) == NULL,
          "freq>=fs/2 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, ok, 2, 0.0f, -50.0f) == NULL,
          "threshold=0 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, ok, 2, 1.5f, -50.0f) == NULL,
          "threshold>1 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, ok, 2, 0.2f, 0.0f) == NULL,
          "silence_db=0 accepted");
    CHECK(kiss_goertzel_alloc(NFR, HOP, FS, ok, 2, 0.2f, -500.0f) == NULL,
          "silence_db=-500 accepted");

    c = kiss_goertzel_alloc(NFR, HOP, FS, ok, 2, 0.2f, -50.0f);
    CHECK(c != NULL, "valid alloc rejected");
    if (c) {
        CHECK(kiss_goertzel_process(NULL, sig, NSAMP, res, MAXFRAMES) < 0,
              "process(NULL cfg) accepted");
        CHECK(kiss_goertzel_process(c, NULL, NSAMP, res, MAXFRAMES) < 0,
              "process(NULL samples) accepted");
        CHECK(kiss_goertzel_process(c, sig, NSAMP, NULL, MAXFRAMES) < 0,
              "process(NULL results) accepted");
        CHECK(kiss_goertzel_process(c, sig, -1, res, MAXFRAMES) < 0,
              "process(nsamples<0) accepted");
        CHECK(kiss_goertzel_tone_power(c, sig, 5) < 0.0,
              "tone_power(bad index) accepted");
        kiss_goertzel_free(c);
    }
    printf("--- invalid-parameter checks done ---\n");
}

int main(void)
{
    build_signal();
    test_detection();
    test_error_bound();
    test_invalid_params();

    if (nfail == 0) {
        printf("test_goertzel: ALL TESTS PASSED\n");
        return 0;
    }
    printf("test_goertzel: %d FAILURES\n", nfail);
    return 1;
}
