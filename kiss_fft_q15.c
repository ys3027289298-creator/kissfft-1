/*
 *  16-bit fixed-point (Q15) radix-2 FFT, see kiss_fft_q15.h.
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_fft_q15.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Q15_ONE 32767

struct kiss_fft_q15_state {
    int nfft;
    int inverse;
    int mode;
    int shift;
    unsigned long overflows;
    int owned;
    kiss_fft_q15_cpx twiddles[1]; /* nfft/2 entries, flexible tail */
};

/* Clamp an int32 accumulator to the int16 range and count saturations. */
static int16_t q15_sat(int32_t x, struct kiss_fft_q15_state *st)
{
    if (x > INT16_MAX) {
        st->overflows++;
        return INT16_MAX;
    }
    if (x < INT16_MIN) {
        st->overflows++;
        return INT16_MIN;
    }
    return (int16_t)x;
}

/* Rounding Q15 multiply: (a*b + 2^14) >> 15. Never overflows int16. */
static int16_t q15_mul(int16_t a, int16_t b)
{
    int32_t p = (int32_t)a * b + (1 << 14);
    return (int16_t)(p >> 15);
}

/* Complex Q15 multiply; the cross-sum/difference is saturated. */
static kiss_fft_q15_cpx q15_cmul(kiss_fft_q15_cpx a, kiss_fft_q15_cpx w,
                                 struct kiss_fft_q15_state *st)
{
    kiss_fft_q15_cpx m;
    m.r = q15_sat((int32_t)q15_mul(a.r, w.r) - q15_mul(a.i, w.i), st);
    m.i = q15_sat((int32_t)q15_mul(a.r, w.i) + q15_mul(a.i, w.r), st);
    return m;
}

static int is_pow2(int n)
{
    return n >= 2 && (n & (n - 1)) == 0;
}

/* Round to nearest int without depending on C99 lround(). */
static long q15_round(double x)
{
    return (long)(x >= 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
}

/* Arithmetic right shift by one bit (halving), well defined for negatives. */
static int16_t half16(int16_t x)
{
    return (int16_t)(((int32_t)x) >> 1);
}

kiss_fft_q15_cfg kiss_fft_q15_alloc(int nfft, int inverse_fft, int mode,
                                    void *mem, size_t *lenmem)
{
    struct kiss_fft_q15_state *st;
    size_t need;
    int p;

    if (!is_pow2(nfft))
        return NULL;

    need = sizeof(struct kiss_fft_q15_state)
         + (size_t)(nfft / 2 - 1) * sizeof(kiss_fft_q15_cpx);
    if (lenmem != NULL && (mem == NULL || *lenmem < need)) {
        *lenmem = need;
        if (mem == NULL)
            return NULL;
    }

    st = (struct kiss_fft_q15_state *)
         (mem != NULL ? mem : malloc(need));
    if (st == NULL)
        return NULL;

    st->nfft = nfft;
    st->inverse = inverse_fft;
    st->mode = mode ? KISS_FFT_Q15_UNSCALED : KISS_FFT_Q15_SCALED;
    st->shift = 0;
    st->overflows = 0;
    st->owned = (mem == NULL);

    /* Generate the Q15 twiddle table at the requested length:
     * forward W^p = exp(-j 2pi p/nfft), inverse uses the conjugate. */
    for (p = 0; p < nfft / 2; p++) {
        double phase = (inverse_fft ? 1.0 : -1.0) * 2.0 * M_PI * p / nfft;
        long c = q15_round(Q15_ONE * cos(phase));
        long s = q15_round(Q15_ONE * sin(phase));
        st->twiddles[p].r = (int16_t)(c > INT16_MAX ? INT16_MAX : c);
        st->twiddles[p].i = (int16_t)(s < INT16_MIN ? INT16_MIN : s);
    }
    return st;
}

void kiss_fft_q15_free(kiss_fft_q15_cfg cfg)
{
    if (cfg != NULL && cfg->owned)
        free(cfg);
}

unsigned long kiss_fft_q15_overflows(const kiss_fft_q15_cfg cfg)
{
    return cfg->overflows;
}

int kiss_fft_q15_scale_shift(const kiss_fft_q15_cfg cfg)
{
    return cfg->shift;
}

void kiss_fft_q15_reset_overflows(kiss_fft_q15_cfg cfg)
{
    cfg->overflows = 0;
}

void kiss_fft_q15(kiss_fft_q15_cfg cfg,
                  const kiss_fft_q15_cpx *fin, kiss_fft_q15_cpx *fout)
{
    const int nfft = cfg->nfft;
    int i, j, bit, m, half, step, k;

    if (fin != fout)
        memcpy(fout, fin, (size_t)nfft * sizeof(*fin));

    /* Bit-reversal permutation. */
    for (i = 1, j = 0; i < nfft; i++) {
        kiss_fft_q15_cpx tmp;
        bit = nfft >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            tmp = fout[i];
            fout[i] = fout[j];
            fout[j] = tmp;
        }
    }

    /* Radix-2 DIT stages. In scaled mode a stage whose values can no
     * longer double is halved first (block-float scaling), so the
     * saturated adds below only act as a safety net / overflow counter. */
    for (m = 2; m <= nfft; m <<= 1) {
        if (cfg->mode == KISS_FFT_Q15_SCALED) {
            int need_scale = 0;
            for (i = 0; i < nfft; i++) {
                if (abs(fout[i].r) >= 16384 || abs(fout[i].i) >= 16384) {
                    need_scale = 1;
                    break;
                }
            }
            if (need_scale) {
                for (i = 0; i < nfft; i++) {
                    fout[i].r = half16(fout[i].r);
                    fout[i].i = half16(fout[i].i);
                }
                cfg->shift++;
            }
        }
        half = m >> 1;
        step = nfft / m;
        for (k = 0; k < half; k++) {
            kiss_fft_q15_cpx w = cfg->twiddles[k * step];
            for (j = k; j < nfft; j += m) {
                kiss_fft_q15_cpx u = fout[j];
                kiss_fft_q15_cpx t = q15_cmul(fout[j + half], w, cfg);
                fout[j].r = q15_sat((int32_t)u.r + t.r, cfg);
                fout[j].i = q15_sat((int32_t)u.i + t.i, cfg);
                fout[j + half].r = q15_sat((int32_t)u.r - t.r, cfg);
                fout[j + half].i = q15_sat((int32_t)u.i - t.i, cfg);
            }
        }
    }
}

double kiss_fft_q15_snr(const kiss_fft_q15_cpx *q15,
                        const double *ref_re, const double *ref_im,
                        int nfft, double *max_err)
{
    double sigpow = 0.0, noisepow = 0.0, worst = 0.0;
    int k;

    for (k = 0; k < nfft; k++) {
        double dr = (double)q15[k].r - ref_re[k];
        double di = (double)q15[k].i - ref_im[k];
        double err = sqrt(dr * dr + di * di);
        if (err > worst)
            worst = err;
        sigpow += ref_re[k] * ref_re[k] + ref_im[k] * ref_im[k];
        noisepow += dr * dr + di * di;
    }
    if (max_err != NULL)
        *max_err = worst;
    return 10.0 * log10(sigpow / (noisepow + 1e-30));
}

#ifdef KISS_FFT_Q15_SELFTEST
/* Same SNR/max_err metric for already-double candidate spectra. */
static double kiss_fft_q15_snr_dbl(const double *qre, const double *qim,
                                   const double *ref_re, const double *ref_im,
                                   int nfft, double *max_err)
{
    double sigpow = 0.0, noisepow = 0.0, worst = 0.0;
    int k;

    for (k = 0; k < nfft; k++) {
        double dr = qre[k] - ref_re[k];
        double di = qim[k] - ref_im[k];
        double err = sqrt(dr * dr + di * di);
        if (err > worst)
            worst = err;
        sigpow += ref_re[k] * ref_re[k] + ref_im[k] * ref_im[k];
        noisepow += dr * dr + di * di;
    }
    if (max_err != NULL)
        *max_err = worst;
    return 10.0 * log10(sigpow / (noisepow + 1e-30));
}

#include <stdio.h>

/* Naive floating-point DFT reference, same int16 sample scale as input. */
static void float_dft(const kiss_fft_q15_cpx *in, double *re, double *im, int n)
{
    int k, t;
    for (k = 0; k < n; k++) {
        double sr = 0.0, si = 0.0;
        for (t = 0; t < n; t++) {
            double ph = -2.0 * M_PI * k * t / n;
            sr += in[t].r * cos(ph) - in[t].i * sin(ph);
            si += in[t].r * sin(ph) + in[t].i * cos(ph);
        }
        re[k] = sr;
        im[k] = si;
    }
}

static int16_t clip16(double v)
{
    if (v > INT16_MAX) return INT16_MAX;
    if (v < INT16_MIN) return INT16_MIN;
    return (int16_t)q15_round(v);
}

/* Two-tone real signal: bins 7 and 23, amplitudes A and A/4. */
static void make_signal(kiss_fft_q15_cpx *x, int n, int amp)
{
    int t;
    for (t = 0; t < n; t++) {
        double v = amp * cos(2.0 * M_PI * 7 * t / n)
                 + (amp / 4.0) * cos(2.0 * M_PI * 23 * t / n);
        x[t].r = clip16(v);
        x[t].i = 0;
    }
}

static void run_case(const char *name, int n, int amp, int mode)
{
    kiss_fft_q15_cpx *in = malloc(sizeof(*in) * n);
    kiss_fft_q15_cpx *out = malloc(sizeof(*out) * n);
    double *re = malloc(sizeof(double) * n), *im = malloc(sizeof(double) * n);
    kiss_fft_q15_cfg cfg = kiss_fft_q15_alloc(n, 0, mode, NULL, NULL);
    double snr, maxerr, gain;
    double *qre = malloc(sizeof(double) * n), *qim = malloc(sizeof(double) * n);

    make_signal(in, n, amp);
    float_dft(in, re, im, n);
    kiss_fft_q15(cfg, in, out);
    gain = (mode == KISS_FFT_Q15_SCALED)
         ? (double)(1 << kiss_fft_q15_scale_shift(cfg)) : 1.0;
    for (int t = 0; t < n; t++) { /* undo block-float scaling in doubles */
        qre[t] = out[t].r * gain;
        qim[t] = out[t].i * gain;
    }
    snr = kiss_fft_q15_snr_dbl(qre, qim, re, im, n, &maxerr);
    printf("--- %s (N=%d, amp=%d, %s) ---\n", name, n, amp,
           mode == KISS_FFT_Q15_SCALED ? "scaled" : "unscaled");
    printf("SNR=%.2f dB  max_err=%.2f (int16 LSB)  overflows=%lu  shifts=%d\n\n",
           snr, maxerr, kiss_fft_q15_overflows(cfg),
           kiss_fft_q15_scale_shift(cfg));
    kiss_fft_q15_free(cfg);
    free(in); free(out); free(re); free(im); free(qre); free(qim);
}

int main(void)
{
    printf("Q15 FFT self-test: SNR vs float DFT, max error in int16 LSB\n\n");
    run_case("case1 256-point", 256, 2000, KISS_FFT_Q15_SCALED);
    run_case("case2 1024-point", 1024, 2000, KISS_FFT_Q15_SCALED);
    run_case("case3 full-scale", 256, 32767, KISS_FFT_Q15_UNSCALED);
    run_case("case4 small-signal", 256, 16, KISS_FFT_Q15_SCALED);
    return 0;
}
#endif
