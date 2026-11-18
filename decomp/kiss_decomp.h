/*
 * kiss_decomp.h — arbitrary-length FFT length decomposer on top of the
 * public KISS FFT API (kiss_fft.h). Header-only; no changes to the library.
 *
 * Length strategy for a requested transform length N:
 *
 *   mixed radix : if N = 2^a * 3^b * 5^c * 7^d, run kiss_fft directly on N
 *                 (radix 2/3/4/5 use the optimized butterflies, radix 7 the
 *                 generic one). The power-of-two path is the library's own,
 *                 bit-identical to calling kiss_fft() yourself.
 *   Bluestein   : otherwise, exact N-point DFT via chirp-z convolution of
 *                 length M = kiss_fft_next_fast_size(2N-1).
 *   zero-pad    : legacy behavior — pad the input with zeros to the next
 *                 power of two M and compute the M-point DFT. NOTE: this is
 *                 NOT the N-point DFT; it is the N-point spectrum resampled
 *                 onto an M-bin grid (the output has M bins, not N).
 *
 * Edge lengths: N <= 0 is rejected (plan == NULL); N == 1 is the identity.
 *
 * Only kiss_fft.h public symbols are used; this file introduces no new
 * dependencies and no changes to existing library code.
 */
#ifndef KISS_DECOMP_H
#define KISS_DECOMP_H

#include "kiss_fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KISS_DECOMP_AUTO = 0,   /* mixed radix when possible, else Bluestein   */
    KISS_DECOMP_MIXED_ONLY, /* fail (NULL plan) if N has a prime factor > 7 */
    KISS_DECOMP_BLUESTEIN,  /* force Bluestein (exact N-point DFT)          */
    KISS_DECOMP_ZEROPAD     /* force legacy zero-pad to next power of two   */
} kiss_decomp_strategy;

typedef struct kiss_decomp_plan {
    int n;                     /* requested length (input length)           */
    int nout;                  /* output length (== n except ZEROPAD)       */
    int nfft;                  /* internal transform length                 */
    int inverse;               /* 0 = forward, 1 = inverse                  */
    kiss_decomp_strategy strategy; /* strategy actually in effect          */
    int factors[32];           /* prime factors of n (mixed) or of M        */
    int nfactors;
    kiss_fft_cfg cfg;          /* mixed / zero-pad path                     */
    kiss_fft_cfg cfg_fwd;      /* Bluestein: forward M-point cfg            */
    kiss_fft_cfg cfg_inv;      /* Bluestein: inverse M-point cfg            */
    kiss_fft_cpx *chirp;       /* Bluestein: exp(s*i*pi*k^2/n), k = 0..n-1  */
    kiss_fft_cpx *bfft;        /* Bluestein: FFT of the b sequence (len M)  */
    kiss_fft_cpx *scratch_a;   /* work buffers (len nfft)                   */
    kiss_fft_cpx *scratch_b;
} kiss_decomp_plan;

/* factor n over {2,3,5,7}; stores all prime factors (incl. any leftover
 * factor > 7) in fac[] and returns the unfactored remainder (1 == smooth) */
static int kiss_decomp_factor(int n, int *fac, int *nfactors)
{
    static const int radices[4] = {2, 3, 5, 7};
    int nf = 0, i, rem = n;
    for (i = 0; i < 4; ++i) {
        while (rem > 1 && rem % radices[i] == 0) {
            if (nf < 32) fac[nf] = radices[i];
            ++nf;
            rem /= radices[i];
        }
    }
    if (rem > 1) {
        int r = rem, p;
        for (p = 11; (long)p * p <= r; p += 2) { /* report big prime factors */
            while (r % p == 0) {
                if (nf < 32) fac[nf] = p;
                ++nf;
                r /= p;
            }
        }
        if (r > 1) {
            if (nf < 32) fac[nf] = r;
            ++nf;
        }
    }
    *nfactors = nf;
    return rem;
}

static int kiss_decomp_next_pow2(int n)
{
    int m = 1;
    while (m < n)
        m <<= 1;
    return m;
}

/* Returns NULL for n <= 0, for allocation failure, or when the requested
 * strategy cannot handle n (e.g. KISS_DECOMP_MIXED_ONLY on a non-smooth n). */
static kiss_decomp_plan *kiss_decomp_create(int n, int inverse,
                                            kiss_decomp_strategy strategy)
{
    kiss_decomp_plan *pl;
    int rem, smooth;

    if (n <= 0)
        return NULL;
    pl = (kiss_decomp_plan *)calloc(1, sizeof(*pl));
    if (!pl)
        return NULL;
    pl->n = n;
    pl->nfft = n;
    pl->nout = n;
    pl->inverse = inverse ? 1 : 0;

    if (n == 1) { /* identity transform, any strategy degenerates */
        pl->strategy = KISS_DECOMP_MIXED_ONLY;
        pl->factors[0] = 1;
        pl->nfactors = 1;
        return pl;
    }

    rem = kiss_decomp_factor(n, pl->factors, &pl->nfactors);
    smooth = (rem == 1);

    if (strategy == KISS_DECOMP_AUTO)
        strategy = smooth ? KISS_DECOMP_MIXED_ONLY : KISS_DECOMP_BLUESTEIN;
    pl->strategy = strategy;

    switch (strategy) {
    case KISS_DECOMP_MIXED_ONLY:
        if (!smooth) {
            free(pl);
            return NULL;
        }
        pl->cfg = kiss_fft_alloc(n, pl->inverse, NULL, NULL);
        if (!pl->cfg) {
            free(pl);
            return NULL;
        }
        break;

    case KISS_DECOMP_ZEROPAD:
        pl->nfft = kiss_decomp_next_pow2(n);
        pl->nout = pl->nfft; /* caller gets the padded spectrum */
        kiss_decomp_factor(pl->nfft, pl->factors, &pl->nfactors);
        pl->cfg = kiss_fft_alloc(pl->nfft, pl->inverse, NULL, NULL);
        pl->scratch_a =
            (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)pl->nfft);
        if (!pl->cfg || !pl->scratch_a) {
            kiss_decomp_plan *tmp = pl;
            free(tmp->scratch_a);
            free(tmp);
            return NULL;
        }
        break;

    case KISS_DECOMP_BLUESTEIN: {
        int m = kiss_fft_next_fast_size(2 * n - 1);
        int k;
        double s = pl->inverse ? 1.0 : -1.0;
        pl->nfft = m;
        kiss_decomp_factor(m, pl->factors, &pl->nfactors);
        pl->cfg_fwd = kiss_fft_alloc(m, 0, NULL, NULL);
        pl->cfg_inv = kiss_fft_alloc(m, 1, NULL, NULL);
        pl->chirp = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)n);
        pl->bfft = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)m);
        pl->scratch_a =
            (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)m);
        pl->scratch_b =
            (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (size_t)m);
        if (!pl->cfg_fwd || !pl->cfg_inv || !pl->chirp || !pl->bfft ||
            !pl->scratch_a || !pl->scratch_b) {
            kiss_fft_free(pl->cfg_fwd);
            kiss_fft_free(pl->cfg_inv);
            free(pl->chirp);
            free(pl->bfft);
            free(pl->scratch_a);
            free(pl->scratch_b);
            free(pl);
            return NULL;
        }
        /* X[k] = e^{s i pi k^2/n} * sum_j (x[j] e^{s i pi j^2/n})
         *                              * e^{-s i pi (k-j)^2/n}            */
        for (k = 0; k < n; ++k) {
            double phase = s * M_PI * ((double)k * (double)k) / (double)n;
            pl->chirp[k].r = (kiss_fft_scalar)cos(phase);
            pl->chirp[k].i = (kiss_fft_scalar)sin(phase);
        }
        /* b[m] = e^{-s i pi m^2/n} on the circular grid 0..m-1 (b is even) */
        pl->scratch_b[0].r = 1.0f;
        pl->scratch_b[0].i = 0.0f;
        for (k = 1; k < m; ++k)
            pl->scratch_b[k].r = pl->scratch_b[k].i = 0.0f;
        for (k = 1; k < n; ++k) {
            double phase = -s * M_PI * ((double)k * (double)k) / (double)n;
            kiss_fft_cpx v;
            v.r = (kiss_fft_scalar)cos(phase);
            v.i = (kiss_fft_scalar)sin(phase);
            pl->scratch_b[k] = v;
            pl->scratch_b[m - k] = v;
        }
        kiss_fft(pl->cfg_fwd, pl->scratch_b, pl->bfft);
        break;
    }
    default: /* KISS_DECOMP_AUTO was resolved above; unreachable */
        break;
    }
    return pl;
}

static void kiss_decomp_exec(const kiss_decomp_plan *pl,
                             const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
    int k;
    if (pl->n == 1) {
        fout[0] = fin[0];
        return;
    }
    switch (pl->strategy) {
    case KISS_DECOMP_MIXED_ONLY:
        kiss_fft(pl->cfg, fin, fout);
        break;

    case KISS_DECOMP_ZEROPAD:
        memcpy(pl->scratch_a, fin, sizeof(kiss_fft_cpx) * (size_t)pl->n);
        for (k = pl->n; k < pl->nfft; ++k)
            pl->scratch_a[k].r = pl->scratch_a[k].i = 0.0f;
        kiss_fft(pl->cfg, pl->scratch_a, fout);
        break;

    case KISS_DECOMP_BLUESTEIN: {
        const int m = pl->nfft;
        for (k = 0; k < pl->n; ++k) { /* a[j] = x[j] * chirp[j] */
            pl->scratch_a[k].r =
                fin[k].r * pl->chirp[k].r - fin[k].i * pl->chirp[k].i;
            pl->scratch_a[k].i =
                fin[k].r * pl->chirp[k].i + fin[k].i * pl->chirp[k].r;
        }
        for (; k < m; ++k)
            pl->scratch_a[k].r = pl->scratch_a[k].i = 0.0f;
        kiss_fft(pl->cfg_fwd, pl->scratch_a, pl->scratch_b);
        for (k = 0; k < m; ++k) { /* pointwise multiply by FFT(b) */
            kiss_fft_cpx t;
            t.r = pl->scratch_b[k].r * pl->bfft[k].r -
                  pl->scratch_b[k].i * pl->bfft[k].i;
            t.i = pl->scratch_b[k].r * pl->bfft[k].i +
                  pl->scratch_b[k].i * pl->bfft[k].r;
            pl->scratch_b[k] = t;
        }
        kiss_fft(pl->cfg_inv, pl->scratch_b, pl->scratch_a);
        /* kiss_fft inverse is unnormalized: divide by m, apply chirp */
        for (k = 0; k < pl->n; ++k) {
            kiss_fft_cpx c;
            c.r = pl->scratch_a[k].r / (kiss_fft_scalar)m;
            c.i = pl->scratch_a[k].i / (kiss_fft_scalar)m;
            fout[k].r = c.r * pl->chirp[k].r - c.i * pl->chirp[k].i;
            fout[k].i = c.r * pl->chirp[k].i + c.i * pl->chirp[k].r;
        }
        break;
    }
    default: /* KISS_DECOMP_AUTO never reaches a plan */
        break;
    }
}

static void kiss_decomp_free(kiss_decomp_plan *pl)
{
    if (!pl)
        return;
    kiss_fft_free(pl->cfg);
    kiss_fft_free(pl->cfg_fwd);
    kiss_fft_free(pl->cfg_inv);
    free(pl->chirp);
    free(pl->bfft);
    free(pl->scratch_a);
    free(pl->scratch_b);
    free(pl);
}

/* human-readable factorization, e.g. "2^2 * 3^2 * 5" */
static void kiss_decomp_fmt_factors(char *buf, size_t len, const int *fac,
                                    int nfactors)
{
    size_t off = 0;
    int i = 0;
    buf[0] = '\0';
    if (nfactors <= 0) {
        snprintf(buf, len, "1");
        return;
    }
    while (i < nfactors) {
        int p = fac[i], e = 0, j = i;
        while (j < nfactors && fac[j] == p) {
            ++e;
            ++j;
        }
        if (e == 1)
            off += (size_t)snprintf(buf + off, len - off, "%s%d",
                                    off ? " * " : "", p);
        else
            off += (size_t)snprintf(buf + off, len - off, "%s%d^%d",
                                    off ? " * " : "", p, e);
        i = j;
    }
}

/* describes the plan: strategy actually used and the factorization */
static void kiss_decomp_describe(const kiss_decomp_plan *pl, char *buf,
                                 size_t len)
{
    char f[160];
    kiss_decomp_fmt_factors(f, sizeof(f), pl->factors, pl->nfactors);
    if (pl->n == 1) {
        snprintf(buf, len, "1 (identity)");
        return;
    }
    switch (pl->strategy) {
    case KISS_DECOMP_MIXED_ONLY:
        snprintf(buf, len, "%d = %s [mixed-radix 2/3/5/7, native kiss_fft]",
                 pl->n, f);
        break;
    case KISS_DECOMP_ZEROPAD:
        snprintf(buf, len,
                 "%d -> zero-pad to %d = %s [legacy; NOT the %d-point DFT]",
                 pl->n, pl->nfft, f, pl->n);
        break;
    case KISS_DECOMP_BLUESTEIN:
        snprintf(buf, len, "%d -> Bluestein, conv len %d = %s [exact]", pl->n,
                 pl->nfft, f);
        break;
    default:
        snprintf(buf, len, "%d (auto)", pl->n);
        break;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* KISS_DECOMP_H */
