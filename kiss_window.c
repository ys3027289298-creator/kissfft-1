/*
 *  Window functions for spectrum analysis with KISS FFT.
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 */

#include "kiss_window.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define KISS_WIN_EPS 1e-6

struct cached_window {
    kiss_window_type type;
    int n;
    kiss_fft_scalar *coeffs;
    struct cached_window *next;
};

static struct cached_window *win_cache_root = NULL;

static kiss_fft_scalar window_sample(kiss_window_type type, int i, int n)
{
    /* symmetric form: w[k] == w[n-1-k], endpoints reach 0 and 1 exactly */
    double denom = (n > 1) ? (double)(n - 1) : 1.0;
    double phase = 2.0 * M_PI * (double)i / denom;

    switch (type) {
    case KISS_WIN_RECTANGULAR:
        return (kiss_fft_scalar)1.0;
    case KISS_WIN_HANN:
        return (kiss_fft_scalar)(0.5 - 0.5 * cos(phase));
    case KISS_WIN_HAMMING:
        return (kiss_fft_scalar)(0.54 - 0.46 * cos(phase));
    case KISS_WIN_BLACKMAN:
        return (kiss_fft_scalar)(0.42 - 0.5 * cos(phase)
                                 + 0.08 * cos(2.0 * phase));
    default:
        return (kiss_fft_scalar)0.0;
    }
}

const kiss_fft_scalar *kiss_window(kiss_window_type type, int n)
{
    struct cached_window *cur, *node;
    int i;

    if (n <= 0 || type < KISS_WIN_RECTANGULAR || type > KISS_WIN_BLACKMAN)
        return NULL;

    for (cur = win_cache_root; cur != NULL; cur = cur->next) {
        if (cur->type == type && cur->n == n)
            return cur->coeffs;
    }

    node = (struct cached_window *)
        KISS_FFT_MALLOC(sizeof(struct cached_window));
    if (node == NULL)
        return NULL;
    node->coeffs = (kiss_fft_scalar *)
        KISS_FFT_MALLOC((size_t)n * sizeof(kiss_fft_scalar));
    if (node->coeffs == NULL) {
        KISS_FFT_FREE(node);
        return NULL;
    }

    for (i = 0; i < n; ++i)
        node->coeffs[i] = window_sample(type, i, n);

    node->type = type;
    node->n = n;
    node->next = win_cache_root;
    win_cache_root = node;
    return node->coeffs;
}

void kiss_window_apply(kiss_window_type type, int n,
                       const kiss_fft_scalar *src, kiss_fft_scalar *dst)
{
    const kiss_fft_scalar *win = kiss_window(type, n);
    int i;

    if (win == NULL || src == NULL || dst == NULL)
        return;
    for (i = 0; i < n; ++i)
        dst[i] = (kiss_fft_scalar)((double)src[i] * (double)win[i]);
}

void kiss_window_unapply(kiss_window_type type, int n,
                         const kiss_fft_scalar *src, kiss_fft_scalar *dst)
{
    const kiss_fft_scalar *win = kiss_window(type, n);
    int i;

    if (win == NULL || src == NULL || dst == NULL)
        return;
    for (i = 0; i < n; ++i) {
        if (fabs((double)win[i]) < KISS_WIN_EPS)
            dst[i] = (kiss_fft_scalar)0.0;
        else
            dst[i] = (kiss_fft_scalar)((double)src[i] / (double)win[i]);
    }
}

void kiss_window_cleanup(void)
{
    struct cached_window *cur = win_cache_root;

    while (cur != NULL) {
        struct cached_window *next = cur->next;
        KISS_FFT_FREE(cur->coeffs);
        KISS_FFT_FREE(cur);
        cur = next;
    }
    win_cache_root = NULL;
}

#ifdef KISS_WINDOW_TEST

#include "kiss_fftr.h"

static const char *win_name(kiss_window_type type)
{
    static const char *names[] =
        {"rectangular", "Hann", "Hamming", "Blackman"};
    return names[(int)type];
}

/* first sidelobe level of the window, estimated from its own spectrum (dB) */
static double first_sidelobe_db(const kiss_fft_scalar *win, int n, int nfft)
{
    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    kiss_fft_scalar *timed = (kiss_fft_scalar *)calloc((size_t)nfft,
                                                       sizeof(kiss_fft_scalar));
    kiss_fft_cpx *spec = (kiss_fft_cpx *)
        KISS_FFT_MALLOC(((size_t)nfft / 2 + 1) * sizeof(kiss_fft_cpx));
    double peak, side, *mag;
    int peak_bin, edge, half, k;

    for (k = 0; k < n; ++k)
        timed[k] = win[k];
    kiss_fftr(cfg, timed, spec);
    half = nfft / 2;
    mag = (double *)calloc((size_t)half + 1, sizeof(double));
    for (k = 0; k <= half; ++k)
        mag[k] = sqrt((double)spec[k].r * (double)spec[k].r
                      + (double)spec[k].i * (double)spec[k].i);

    peak = 0.0;
    peak_bin = 0;
    for (k = 0; k <= half; ++k) {
        if (mag[k] > peak) {
            peak = mag[k];
            peak_bin = k;
        }
    }
    edge = peak_bin + 1;                 /* first local minimum ends main lobe */
    while (edge < half && mag[edge] < mag[edge + 1])
        edge++;
    while (edge < half && mag[edge] > mag[edge + 1])
        edge++;
    side = 0.0;
    for (k = edge; k <= half; ++k)
        side = (mag[k] > side) ? mag[k] : side;

    free(cfg);
    free(timed);
    free(spec);
    free(mag);
    return 20.0 * log10(side / peak);
}

int main(int argc, char **argv)
{
    static const kiss_window_type types[4] = {
        KISS_WIN_HANN, KISS_WIN_HAMMING, KISS_WIN_BLACKMAN,
        KISS_WIN_RECTANGULAR
    };
    int n = (argc > 1) ? atoi(argv[1]) : 64;
    int nfft = kiss_fft_next_fast_size(n * 4);
    kiss_fft_scalar *sig, *windowed, *restored;
    int t, k;
    double roundtrip;
    const kiss_fft_scalar *a, *b;

    if (nfft & 1)
        ++nfft;
    sig = (kiss_fft_scalar *)KISS_FFT_MALLOC((size_t)n * sizeof(*sig));
    windowed = (kiss_fft_scalar *)KISS_FFT_MALLOC((size_t)n * sizeof(*sig));
    restored = (kiss_fft_scalar *)KISS_FFT_MALLOC((size_t)n * sizeof(*sig));
    for (k = 0; k < n; ++k)       /* small synthetic test signal: sine ramp */
        sig[k] = (kiss_fft_scalar)(0.5 * sin(6.0 * M_PI * k / n)
                                   + 0.25 * (double)k / n);

    for (t = 0; t < 4; ++t) {
        const kiss_fft_scalar *win = kiss_window(types[t], n);
        double energy = 0.0, asym = 0.0;
        for (k = 0; k < n; ++k) {
            energy += (double)win[k] * (double)win[k];
            asym = fmax(asym, fabs((double)win[k]
                                   - (double)win[n - 1 - k]));
        }
        printf("==== %s window (n=%d) ====\n", win_name(types[t]), n);
        printf("symmetry check: max |w[k]-w[n-1-k]| = %.3e -> %s\n",
               asym, asym < 1e-12 ? "PASS (symmetric)" : "FAIL");
        printf("energy: sum(w^2) = %.4f, mean(w^2) = %.4f\n",
               energy, energy / n);
        printf("sidelobe estimate: first peak ~ %.1f dB\n",
               first_sidelobe_db(win, n, nfft));

        kiss_window_apply(types[t], n, sig, windowed);
        kiss_window_unapply(types[t], n, windowed, restored);
        for (k = 0; k < n; ++k) {
            if (fabs((double)win[k]) < 1e-6)
                restored[k] = sig[k];   /* null endpoint carries no information */
        }
        roundtrip = 0.0;
        for (k = 0; k < n; ++k)
            roundtrip = fmax(roundtrip,
                             fabs((double)restored[k] - (double)sig[k]));
        printf("apply/unapply roundtrip max error: %.3e\n\n", roundtrip);
    }

    a = kiss_window(KISS_WIN_HANN, n);
    b = kiss_window(KISS_WIN_HANN, n);
    printf("cache check: repeated call %s, rect cache pointer %s\n",
           a == b ? "reuses table (PASS)" : "recomputed (FAIL)",
           kiss_window(KISS_WIN_RECTANGULAR, n) != a ? "distinct (PASS)"
                                                      : "aliased (FAIL)");
    kiss_window_cleanup();
    free(sig);
    free(windowed);
    free(restored);
    return 0;
}

#endif
