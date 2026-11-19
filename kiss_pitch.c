/*
 *  Copyright (c) 2003-2026, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_pitch.h"
#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"

#include <math.h>
#include <string.h>

/* Buffers of different types share one allocation; rounding every size up to
 * this union keeps each buffer aligned for its own type (including double,
 * which KISS_FFT_PACK_SIZE_UP does not cover). */
union kiss_pitch_pack_align {
    void *p;
    double d;
    kiss_fft_cpx c;
};
#define KISS_PITCH_PACK_SIZE_UP(size) \
    (((size) + sizeof(union kiss_pitch_pack_align) - 1) / sizeof(union kiss_pitch_pack_align) * sizeof(union kiss_pitch_pack_align))

#define KISS_PITCH_TWOPI 6.283185307179586476925286766559

#define KISS_PITCH_DEFAULT_FMIN 50.0
#define KISS_PITCH_DEFAULT_FMAX 2000.0
#define KISS_PITCH_DEFAULT_HARMONICS 16
#define KISS_PITCH_MAX_HARMONICS 64
#define KISS_PITCH_DEFAULT_FLOOR_DB (-60.0)
#define KISS_PITCH_DEFAULT_MIN_CONFIDENCE 0.5


/* a partial belongs to a candidate f0 when it is within this fraction of h*f0 */
#define KISS_PITCH_MATCH_TOL 0.03
/* confidence counts power within this (narrower) fraction of each harmonic */
#define KISS_PITCH_CONF_TOL 0.012
/* two peaks closer than this many bins merge into the stronger one */
#define KISS_PITCH_MIN_PEAK_SEPARATION 3
/* candidate f0s closer than this are treated as the same candidate */
#define KISS_PITCH_CANDIDATE_MERGE 0.01
/* how many unvoiced frames an established f0 survives */
#define KISS_PITCH_MAX_GAP 2
/* a competing octave must beat the established f0 by this score ratio ... */
#define KISS_PITCH_SWITCH_MARGIN 1.25
/* ... for this many consecutive frames before the analysis follows it */
#define KISS_PITCH_SWITCH_FRAMES 3
/* ... while explaining this much of the established pitch's partial energy */
#define KISS_PITCH_SWITCH_COVERAGE_SLACK 0.02
/* peaks are matched to previous-frame tracks within this fraction of the
 * predicted frequency, or two bins, whichever is larger */
#define KISS_PITCH_TRACK_TOL 0.03
/* candidates within this ratio of the established f0 count as "same pitch" */
#define KISS_PITCH_SAME_RATIO 0.06
/* a candidate with a peak sitting at f0 wins ties against one without */
#define KISS_PITCH_PRESENCE_BONUS 1.05
/* refinement stops when f0 moves by less than this fraction */
#define KISS_PITCH_REFINE_EPS 1e-9

struct kiss_pitch_peak_internal {
    double freq;
    double mag; /* relative to the strongest in-band peak */
    int bin;
    int track;  /* index into the track list, -1 before linking */
};

struct kiss_pitch_candidate {
    double f0;      /* refined fundamental */
    double covered; /* amplitude of the partials explained by this f0 */
    double score;   /* covered amplitude, discounted by the harmonic count */
    double rank;    /* score with a tie-break bonus */
    int harmonics;  /* highest harmonic number explained */
    int n_assigned; /* number of partials explained */
    int present;    /* a peak sits at f0 itself */
};

struct kiss_pitch_track {
    double freq;
    int id;    /* stable identifier handed out to the caller */
    int frame; /* frame in which the track was last matched */
    int age;   /* consecutive frames the track has been alive */
};

struct kiss_pitch_state {
    int nfft;
    int n_bins; /* nfft/2 + 1 */
    double sample_rate;
    double bin_hz;
    double fmin;
    double fmax;
    double floor_db;
    double min_confidence;
    int max_harmonics;
    int n_cand_slots;
    int track_cap;

    int frame;     /* frames processed so far */
    int gap;       /* unvoiced frames since the last voiced one */
    int challenge; /* consecutive frames another pitch has won on score */
    double last_f0;
    int n_tracks;
    int next_track_id;

    kiss_fftr_cfg fftr;

    /* Every buffer lives in the same allocation as the state itself. */
    kiss_fft_scalar *windowed; /* nfft */
    kiss_fft_cpx *spec;        /* n_bins */
    double *window;            /* nfft */
    double *mag;               /* n_bins */
    unsigned char *is_peak;    /* n_bins */
    struct kiss_pitch_peak_internal *peaks; /* n_bins worst case */
    struct kiss_pitch_candidate *cands;     /* n_cand_slots */
    int *assign;               /* max_harmonics + 1, scratch */
    int *chosen_assign;        /* max_harmonics + 1 */
    int *peak_harmonic;        /* n_bins */
    double *harmonic_err;      /* max_harmonics + 1 */
    double scratch_covered;    /* amplitude covered by the last evaluation */
    struct kiss_pitch_track *tracks; /* track_cap */
};

/* ---- availability -------------------------------------------------------- */

#if defined(USE_SIMD)
int kiss_pitch_available(void)
{
    return 0;
}
#else
int kiss_pitch_available(void)
{
    return 1;
}
#endif

/* ---- allocation ---------------------------------------------------------- */

static size_t pitch_track_cap(int n_bins)
{
    int max_peaks = n_bins / (KISS_PITCH_MIN_PEAK_SEPARATION + 1) + 1;
    return (size_t)(KISS_PITCH_MAX_GAP + 1) * (size_t)max_peaks;
}

static size_t pitch_memneeded(int nfft, int n_bins, size_t fftr_size,
                              int n_cand_slots, int track_cap)
{
    return KISS_PITCH_PACK_SIZE_UP(sizeof(struct kiss_pitch_state))
        + KISS_PITCH_PACK_SIZE_UP(fftr_size)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(kiss_fft_scalar) * (size_t)nfft)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(kiss_fft_cpx) * (size_t)n_bins)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(double) * (size_t)nfft)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(double) * (size_t)n_bins)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(unsigned char) * (size_t)n_bins)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(struct kiss_pitch_peak_internal) * (size_t)n_bins)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(struct kiss_pitch_candidate) * (size_t)n_cand_slots)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(int) * (size_t)(KISS_PITCH_MAX_HARMONICS + 1))
        + KISS_PITCH_PACK_SIZE_UP(sizeof(int) * (size_t)(KISS_PITCH_MAX_HARMONICS + 1))
        + KISS_PITCH_PACK_SIZE_UP(sizeof(int) * (size_t)n_bins)
        + KISS_PITCH_PACK_SIZE_UP(sizeof(double) * (size_t)(KISS_PITCH_MAX_HARMONICS + 1))
        + KISS_PITCH_PACK_SIZE_UP(sizeof(struct kiss_pitch_track) * (size_t)track_cap);
}

kiss_pitch_cfg kiss_pitch_alloc(int nfft, double sample_rate, void *mem, size_t *lenmem)
{
    kiss_pitch_cfg st = NULL;
    size_t fftr_size = 0, memneeded;
    int n_bins, n_cand_slots, track_cap;
    double bin_hz;
    char *ptr;

    KISS_FFT_ALIGN_CHECK(mem)

    if (!kiss_pitch_available())
        return NULL;
    if (nfft < 8 || (nfft & 1) || !(sample_rate > 0.0))
        return NULL;
    /* keep every derived size well inside size_t */
    if ((size_t)nfft > (size_t)-1 / 64)
        return NULL;

    n_bins = nfft / 2 + 1;
    bin_hz = sample_rate / nfft;

    (void)kiss_fftr_alloc(nfft, 0, NULL, &fftr_size);
    if (fftr_size == 0)
        return NULL;

    /* Candidate f0s are de-duplicated on a logarithmic grid with
     * KISS_PITCH_CANDIDATE_MERGE spacing. The widest legal search range runs
     * from two bins to n_bins-2 bins, which is the most slots ever needed. */
    {
        double max_ratio = (double)(n_bins - 2) / 2.0;
        if (max_ratio < 1.0)
            max_ratio = 1.0;
        n_cand_slots = (int)(log(max_ratio) / log(1.0 + KISS_PITCH_CANDIDATE_MERGE)) + 3;
        if (n_cand_slots < 4)
            n_cand_slots = 4;
    }
    track_cap = (int)pitch_track_cap(n_bins);

    memneeded = pitch_memneeded(nfft, n_bins, fftr_size, n_cand_slots, track_cap);
    if (memneeded < fftr_size) /* size_t overflow */
        return NULL;

    if (lenmem == NULL) {
        st = (kiss_pitch_cfg) KISS_FFT_MALLOC(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_pitch_cfg) mem;
        *lenmem = memneeded;
    }
    if (st == NULL)
        return NULL;

    ptr = (char *) st;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(struct kiss_pitch_state));
    st->fftr = (kiss_fftr_cfg)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(fftr_size);
    st->windowed = (kiss_fft_scalar *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(kiss_fft_scalar) * (size_t)nfft);
    st->spec = (kiss_fft_cpx *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(kiss_fft_cpx) * (size_t)n_bins);
    st->window = (double *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(double) * (size_t)nfft);
    st->mag = (double *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(double) * (size_t)n_bins);
    st->is_peak = (unsigned char *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(unsigned char) * (size_t)n_bins);
    st->peaks = (struct kiss_pitch_peak_internal *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(struct kiss_pitch_peak_internal) * (size_t)n_bins);
    st->cands = (struct kiss_pitch_candidate *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(struct kiss_pitch_candidate) * (size_t)n_cand_slots);
    st->assign = (int *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(int) * (size_t)(KISS_PITCH_MAX_HARMONICS + 1));
    st->chosen_assign = (int *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(int) * (size_t)(KISS_PITCH_MAX_HARMONICS + 1));
    st->peak_harmonic = (int *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(int) * (size_t)n_bins);
    st->harmonic_err = (double *)(void *)ptr;
    ptr += KISS_PITCH_PACK_SIZE_UP(sizeof(double) * (size_t)(KISS_PITCH_MAX_HARMONICS + 1));
    st->tracks = (struct kiss_pitch_track *)(void *)ptr;

    (void)kiss_fftr_alloc(nfft, 0, st->fftr, &fftr_size);
    if (st->fftr == NULL) {
        if (lenmem == NULL)
            KISS_FFT_FREE(st);
        return NULL;
    }

    st->nfft = nfft;
    st->n_bins = n_bins;
    st->sample_rate = sample_rate;
    st->bin_hz = bin_hz;
    st->n_cand_slots = n_cand_slots;
    st->track_cap = track_cap;
    st->max_harmonics = KISS_PITCH_DEFAULT_HARMONICS;
    st->floor_db = KISS_PITCH_DEFAULT_FLOOR_DB;
    st->min_confidence = KISS_PITCH_DEFAULT_MIN_CONFIDENCE;

    /* clamp the default search range to the frequencies this FFT can see */
    st->fmin = KISS_PITCH_DEFAULT_FMIN;
    if (st->fmin < 2.0 * bin_hz)
        st->fmin = 2.0 * bin_hz;
    st->fmax = KISS_PITCH_DEFAULT_FMAX;
    if (st->fmax > (n_bins - 2) * bin_hz)
        st->fmax = (n_bins - 2) * bin_hz;
    if (!(st->fmax > st->fmin)) {
        KISS_FFT_ERROR("kiss_pitch_alloc: nfft too small for the default frequency range");
        if (lenmem == NULL) /* only release memory this call allocated */
            KISS_FFT_FREE(st);
        return NULL;
    }

    {
        int i;
        for (i = 0; i < nfft; ++i)
            st->window[i] = 0.5 - 0.5 * cos(KISS_PITCH_TWOPI * i / (double)nfft);
    }

    kiss_pitch_reset(st);
    return st;
}

void kiss_pitch_reset(kiss_pitch_cfg cfg)
{
    if (cfg == NULL)
        return;
    cfg->scratch_covered = 0.0;
    cfg->frame = 0;
    cfg->gap = KISS_PITCH_MAX_GAP + 1;
    cfg->challenge = 0;
    cfg->last_f0 = 0.0;
    cfg->n_tracks = 0;
    cfg->next_track_id = 1;
}

int kiss_pitch_set_range(kiss_pitch_cfg cfg, double fmin, double fmax)
{
    if (cfg == NULL || !(fmin > 0.0) || !(fmax > fmin))
        return 0;
    /* keep two bins of headroom so the interpolation neighbours exist */
    if (fmin < 2.0 * cfg->bin_hz || fmax > (cfg->n_bins - 2) * cfg->bin_hz)
        return 0;
    cfg->fmin = fmin;
    cfg->fmax = fmax;
    return 1;
}

int kiss_pitch_set_max_harmonics(kiss_pitch_cfg cfg, int max_harmonics)
{
    if (cfg == NULL || max_harmonics < 1 || max_harmonics > KISS_PITCH_MAX_HARMONICS)
        return 0;
    cfg->max_harmonics = max_harmonics;
    return 1;
}

void kiss_pitch_set_detection(kiss_pitch_cfg cfg, double floor_db, double min_confidence)
{
    if (cfg == NULL)
        return;
    if (floor_db <= 0.0)
        cfg->floor_db = floor_db;
    if (min_confidence >= 0.0 && min_confidence <= 1.0)
        cfg->min_confidence = min_confidence;
}

static void clear_output(kiss_pitch_frame *out, double rms)
{
    out->voiced = 0;
    out->f0_hz = 0.0;
    out->confidence = 0.0;
    out->rms = rms;
    out->n_peaks = 0;
}

#if !defined(USE_SIMD)

/* Cast helpers: converting straight from floor()/ceil() to int trips
 * -Wbad-function-cast, so go through a double on purpose. */
static int pitch_int_floor(double v)
{
    int out;
    v = floor(v);
    out = (int)v;
    return out;
}

static int pitch_int_ceil(double v)
{
    int out;
    v = ceil(v);
    out = (int)v;
    return out;
}

/* ---- peak detection ------------------------------------------------------ */

/* Fill st->peaks with the in-band local maxima above the floor, sorted by
 * increasing frequency, and return how many there are. */
static int detect_peaks(kiss_pitch_cfg st)
{
    const int nb = st->n_bins;
    int kmin, kmax, k, n = 0;
    double max_mag = 0.0, floor_mag;

    kmin = pitch_int_ceil(st->fmin / st->bin_hz);
    kmax = pitch_int_floor(st->fmax / st->bin_hz);
    if (kmin < 1)
        kmin = 1;
    if (kmax > nb - 2)
        kmax = nb - 2;
    if (kmin > kmax)
        return 0;

    for (k = kmin; k <= kmax; ++k) {
        st->is_peak[k] = 0;
        if (st->mag[k] > max_mag)
            max_mag = st->mag[k];
    }
    if (!(max_mag > 0.0))
        return 0;
    floor_mag = max_mag * pow(10.0, st->floor_db / 20.0);

    /* Keep a local maximum only if nothing stronger lies within
     * KISS_PITCH_MIN_PEAK_SEPARATION bins, so a broad lobe yields one peak. */
    for (k = kmin; k <= kmax; ++k) {
        int j, lo, hi, ok;

        if (!(st->mag[k] >= floor_mag))
            continue;
        if (!(st->mag[k] > st->mag[k - 1] && st->mag[k] >= st->mag[k + 1]))
            continue;
        lo = k - KISS_PITCH_MIN_PEAK_SEPARATION;
        hi = k + KISS_PITCH_MIN_PEAK_SEPARATION;
        if (lo < kmin)
            lo = kmin;
        if (hi > kmax)
            hi = kmax;
        ok = 1;
        for (j = lo; j <= hi; ++j) {
            if (j == k)
                continue;
            /* ties go to the lower bin, so exactly one of two equal maxima
             * survives and the peak list stays sorted and separated */
            if (st->mag[j] > st->mag[k] || (st->mag[j] == st->mag[k] && j < k)) {
                ok = 0;
                break;
            }
        }
        if (ok)
            st->is_peak[k] = 1;
    }

    /* Parabolic interpolation through the three bins around each maximum.
     * For a Hann window this recovers the true frequency of a clean tone to
     * a small fraction of a cent, because the main lobe is very nearly
     * parabolic in magnitude; the correction is clamped to half a bin so a
     * noisy or overlapping lobe cannot move the peak far from its maximum. */
    for (k = kmin + 1; k <= kmax - 1; ++k) {
        double a, b, c, denom, delta;

        if (!st->is_peak[k])
            continue;
        a = st->mag[k - 1];
        b = st->mag[k];
        c = st->mag[k + 1];
        denom = a + 2.0 * b + c;
        delta = (denom > 0.0) ? 2.0 * (c - a) / denom : 0.0;
        if (delta > 0.5)
            delta = 0.5;
        else if (delta < -0.5)
            delta = -0.5;

        st->peaks[n].freq = (k + delta) * st->bin_hz;
        st->peaks[n].mag = b / max_mag;
        st->peaks[n].bin = k;
        st->peaks[n].track = -1;
        ++n;
    }
    return n;
}

/* ---- harmonic grouping --------------------------------------------------- */

/* Put the closest peak into every harmonic slot of f0. Returns the number of
 * slots filled, the number of slots considered and whether f0 itself got a
 * peak. */
static int assign_partials(kiss_pitch_cfg st, int n_peaks, double f0,
                           int *assign, int *H_out, int *present_out)
{
    int h, i, H, filled = 0;

    H = 0;
    *present_out = 0;
    for (h = 0; h <= st->max_harmonics; ++h)
        assign[h] = -1;
    if (!(f0 > 0.0))
        return 0;

    /* a partial sitting exactly on h*f0 can come out a hair below the
     * integer, so nudge the ratio up before truncating */
    H = pitch_int_floor(st->peaks[n_peaks - 1].freq / f0 + 1e-6);
    if (H > st->max_harmonics)
        H = st->max_harmonics;
    if (H < 1)
        return 0;
    for (h = 1; h <= H; ++h)
        st->harmonic_err[h] = -1.0;

    for (i = 0; i < n_peaks; ++i) {
        double f = st->peaks[i].freq;
        double err;

        h = pitch_int_floor(f / f0 + 0.5);
        if (h < 1 || h > H)
            continue;
        err = fabs(f - h * f0);
        if (!(err <= KISS_PITCH_MATCH_TOL * h * f0))
            continue;
        if (assign[h] < 0 || err < st->harmonic_err[h]) {
            assign[h] = i;
            st->harmonic_err[h] = err;
        }
    }
    for (h = 1; h <= H; ++h) {
        if (assign[h] >= 0)
            ++filled;
    }
    *H_out = H;
    *present_out = (assign[1] >= 0);
    return filled;
}

/* Evaluate one candidate f0: assign partials, refine f0 by weighted least
 * squares over the assigned partials, and repeat a few times. */
static int evaluate_candidate(kiss_pitch_cfg st, int n_peaks, double f0_in,
                              double *f0_out, double *score_out,
                              int *H_out, int *present_out, int *n_assigned_out)
{
    double f0 = f0_in;
    int iter, filled = 0, H = 0, present = 0;

    for (iter = 0; iter < 3; ++iter) {
        double num = 0.0, den = 0.0, refined;
        int h;

        filled = assign_partials(st, n_peaks, f0, st->assign, &H, &present);
        if (filled == 0)
            return 0;
        for (h = 1; h <= H; ++h) {
            int pi = st->assign[h];
            if (pi < 0)
                continue;
            num += st->peaks[pi].mag * h * st->peaks[pi].freq;
            den += st->peaks[pi].mag * h * h;
        }
        if (iter == 2 || !(den > 0.0))
            break;
        refined = num / den;
        if (!(refined > 0.0))
            break;
        if (fabs(refined - f0) <= KISS_PITCH_REFINE_EPS * f0) {
            f0 = refined;
            break;
        }
        f0 = refined;
    }

    /* one more assignment so the reported partials match the reported f0 */
    {
        int h;
        double sum = 0.0;
        filled = assign_partials(st, n_peaks, f0, st->assign, &H, &present);
        if (filled == 0)
            return 0;
        for (h = 1; h <= H; ++h) {
            if (st->assign[h] >= 0)
                sum += st->peaks[st->assign[h]].mag;
        }
        *f0_out = f0;
        *H_out = H;
        *present_out = present;
        *n_assigned_out = filled;
        st->scratch_covered = sum;
        /* Covered partial amplitude, discounted by the harmonic count: a
         * lower octave only wins if it explains appreciably more spectrum. */
        *score_out = sum / sqrt((double)H);
    }
    return 1;
}

/* Share of the in-band spectral power that falls on the harmonic comb of f0. */
static double harmonic_energy_fraction(kiss_pitch_cfg st, double f0)
{
    int k, h, H, kmin, kmax;
    double total = 0.0, got = 0.0;

    kmin = pitch_int_ceil(st->fmin / st->bin_hz);
    kmax = pitch_int_floor(st->fmax / st->bin_hz);
    if (kmin < 1)
        kmin = 1;
    if (kmax > st->n_bins - 1)
        kmax = st->n_bins - 1;

    for (k = kmin; k <= kmax; ++k)
        total += st->mag[k] * st->mag[k];
    if (!(total > 0.0))
        return 0.0;

    H = pitch_int_floor(st->fmax / f0);
    if (H > st->max_harmonics)
        H = st->max_harmonics;
    for (h = 1; h <= H; ++h) {
        double centre = h * f0;
        double half = KISS_PITCH_CONF_TOL * centre;
        int lo, hi;

        /* a narrow tone can be thinner than a bin; always take the bins
         * around the harmonic so low notes are not penalised */
        if (half < 1.5 * st->bin_hz)
            half = 1.5 * st->bin_hz;
        lo = pitch_int_ceil((centre - half) / st->bin_hz);
        hi = pitch_int_floor((centre + half) / st->bin_hz);
        if (lo < kmin)
            lo = kmin;
        if (hi > kmax)
            hi = kmax;
        for (k = lo; k <= hi; ++k)
            got += st->mag[k] * st->mag[k];
    }
    if (got > total)
        return 1.0;
    return got / total;
}

/* Produce candidate fundamentals from every peak divided by 1..max_harmonics,
 * de-duplicated on logarithmic slots. Returns the number of candidates. */
static int generate_candidates(kiss_pitch_cfg st, int n_peaks)
{
    const double log_step = log(1.0 + KISS_PITCH_CANDIDATE_MERGE);
    int i, h, n = 0;

    for (i = 0; i < st->n_cand_slots; ++i)
        st->cands[i].score = -1.0;

    for (i = 0; i < n_peaks; ++i) {
        for (h = 1; h <= st->max_harmonics; ++h) {
            double f0 = st->peaks[i].freq / h;
            int slot;

            if (f0 < st->fmin)
                break; /* higher h only lowers f0 */
            if (f0 > st->fmax)
                continue;
            slot = pitch_int_floor(log(f0 / st->fmin) / log_step);
            if (slot < 0)
                slot = 0;
            if (slot >= st->n_cand_slots)
                slot = st->n_cand_slots - 1;
            if (st->cands[slot].score < 0.0) {
                st->cands[slot].score = 0.0;
                st->cands[slot].f0 = f0;
                ++n;
            }
        }
    }
    return n;
}

/* Index of the best usable candidate, optionally restricted to those within
 * KISS_PITCH_SAME_RATIO of `around`. Returns -1 when there is none. */
static int pick_best(kiss_pitch_cfg st, int same_ratio_only, double around)
{
    int i, best = -1;

    for (i = 0; i < st->n_cand_slots; ++i) {
        struct kiss_pitch_candidate *c = &st->cands[i];

        if (c->score <= 0.0)
            continue;
        if (same_ratio_only && !(fabs(c->f0 / around - 1.0) <= KISS_PITCH_SAME_RATIO))
            continue;
        if (best < 0 || c->rank > st->cands[best].rank)
            best = i;
    }
    return best;
}

/* ---- peak tracking ------------------------------------------------------- */

static void link_tracks(kiss_pitch_cfg st, int n_peaks, double f0)
{
    double scale = 1.0;
    int i, ti, kept = 0, n_tracks = st->n_tracks;

    if (st->last_f0 > 0.0 && f0 > 0.0)
        scale = f0 / st->last_f0;

    /* expire stale tracks and age the survivors */
    for (ti = 0; ti < n_tracks; ++ti) {
        struct kiss_pitch_track tr = st->tracks[ti];
        if (st->frame - tr.frame <= KISS_PITCH_MAX_GAP) {
            tr.age += st->frame - tr.frame - 1;
            st->tracks[kept++] = tr;
        }
    }
    n_tracks = kept;

    /* oldest tracks pick first, so long-lived partials keep their id */
    for (i = 0; i < n_tracks; ++i) {
        int oldest = i, j;
        for (j = i + 1; j < n_tracks; ++j) {
            if (st->tracks[j].age > st->tracks[oldest].age)
                oldest = j;
        }
        if (oldest != i) {
            struct kiss_pitch_track tmp = st->tracks[i];
            st->tracks[i] = st->tracks[oldest];
            st->tracks[oldest] = tmp;
        }
    }

    for (ti = 0; ti < n_tracks; ++ti) {
        double predicted = st->tracks[ti].freq * scale;
        double limit = KISS_PITCH_TRACK_TOL * predicted;
        int best = -1;
        double best_dist = 0.0;

        if (limit < 2.0 * st->bin_hz)
            limit = 2.0 * st->bin_hz;
        for (i = 0; i < n_peaks; ++i) {
            double dist;
            if (st->peaks[i].track >= 0)
                continue;
            dist = fabs(st->peaks[i].freq - predicted);
            if (dist <= limit && (best < 0 || dist < best_dist)) {
                best = i;
                best_dist = dist;
            }
        }
        if (best >= 0) {
            st->peaks[best].track = ti;
            st->tracks[ti].freq = st->peaks[best].freq;
            st->tracks[ti].frame = st->frame;
            st->tracks[ti].age += 1;
        }
    }

    for (i = 0; i < n_peaks; ++i) {
        if (st->peaks[i].track >= 0)
            continue;
        if (n_tracks >= st->track_cap)
            break;
        st->tracks[n_tracks].freq = st->peaks[i].freq;
        st->tracks[n_tracks].frame = st->frame;
        st->tracks[n_tracks].age = 1;
        if (st->next_track_id <= 0) /* ids only need to be unique while a track lives */
            st->next_track_id = 1;
        st->tracks[n_tracks].id = st->next_track_id++;
        st->peaks[i].track = n_tracks;
        ++n_tracks;
    }
    st->n_tracks = n_tracks;
}

/* ---- per-frame analysis -------------------------------------------------- */

static void mark_unvoiced(kiss_pitch_cfg st)
{
    ++st->gap;
    st->challenge = 0;
    if (st->gap > KISS_PITCH_MAX_GAP)
        st->last_f0 = 0.0;
}

int kiss_pitch_process(kiss_pitch_cfg st, const kiss_fft_scalar *frame,
                       kiss_pitch_frame *out, kiss_pitch_peak *peaks, int max_peaks)
{
    double rms = 0.0, f0 = 0.0, confidence = 0.0;
    int i, h, n_peaks, best = -1, chosen = -1, voiced = 0;
    int H = 0, present = 0;

    if (out == NULL)
        return 0;
    if (st == NULL || frame == NULL) {
        clear_output(out, 0.0);
        return 0;
    }
    if (peaks == NULL)
        max_peaks = 0;
    if (max_peaks < 0)
        max_peaks = 0;

    ++st->frame;

    for (i = 0; i < st->nfft; ++i) {
        double v = (double)frame[i];
        rms += v * v;
    }
    rms = sqrt(rms / st->nfft);

    if (rms == 0.0) {
        clear_output(out, rms);
        link_tracks(st, 0, 0.0);
        mark_unvoiced(st);
        return 0;
    }

    for (i = 0; i < st->nfft; ++i)
        st->windowed[i] = (kiss_fft_scalar) ((double)frame[i] * st->window[i]);
    kiss_fftr(st->fftr, st->windowed, st->spec);
    for (i = 0; i < st->n_bins; ++i)
        st->mag[i] = hypot((double)st->spec[i].r, (double)st->spec[i].i);

    n_peaks = detect_peaks(st);
    if (n_peaks == 0) {
        clear_output(out, rms);
        link_tracks(st, 0, 0.0);
        mark_unvoiced(st);
        return 0;
    }

    (void)generate_candidates(st, n_peaks);

    for (i = 0; i < st->n_cand_slots; ++i) {
        struct kiss_pitch_candidate *c = &st->cands[i];
        int cH, cpresent, cna;
        double cf0, cscore;

        if (c->score < 0.0)
            continue;
        if (!evaluate_candidate(st, n_peaks, c->f0, &cf0, &cscore,
                                &cH, &cpresent, &cna)) {
            c->score = -1.0;
            continue;
        }
        c->f0 = cf0;
        c->covered = st->scratch_covered;
        c->score = cscore;
        c->harmonics = cH;
        c->present = cpresent;
        c->n_assigned = cna;
        c->rank = cscore * (cpresent ? KISS_PITCH_PRESENCE_BONUS : 1.0);
    }

    best = pick_best(st, 0, 0.0);
    if (best < 0) {
        clear_output(out, rms);
        link_tracks(st, n_peaks, 0.0);
        mark_unvoiced(st);
        return 0;
    }

    /* Octave guard: keep the established pitch unless a competitor is
     * appreciably better for several consecutive frames. A steady tone then
     * cannot flip octaves on a frame that merely happens to favour another
     * interpretation, while a real register change is still followed. */
    chosen = best;
    if (st->last_f0 > 0.0 && st->gap <= KISS_PITCH_MAX_GAP) {
        int incumbent = pick_best(st, 1, st->last_f0);
        if (incumbent >= 0) {
            if (fabs(st->cands[best].f0 / st->last_f0 - 1.0) <= KISS_PITCH_SAME_RATIO) {
                st->challenge = 0;
            } else if (st->cands[best].score >
                           KISS_PITCH_SWITCH_MARGIN * st->cands[incumbent].score &&
                       /* the newcomer must explain at least as much partial
                        * energy; an octave-down reading always does, because
                        * every partial it claims is also a partial upstairs */
                       st->cands[best].covered >=
                           st->cands[incumbent].covered * (1.0 - KISS_PITCH_SWITCH_COVERAGE_SLACK)) {
                ++st->challenge;
                if (st->challenge >= KISS_PITCH_SWITCH_FRAMES) {
                    st->challenge = 0;
                } else {
                    chosen = incumbent;
                }
            } else {
                st->challenge = 0;
                chosen = incumbent;
            }
        } else {
            st->challenge = 0;
        }
    } else {
        st->challenge = 0;
    }

    f0 = st->cands[chosen].f0;
    H = st->cands[chosen].harmonics;
    confidence = harmonic_energy_fraction(st, f0);
    voiced = (confidence >= st->min_confidence);

    /* rebuild the assignment of the chosen candidate for the output */
    for (i = 0; i < n_peaks; ++i)
        st->peak_harmonic[i] = 0;
    if (voiced) {
        if (!assign_partials(st, n_peaks, f0, st->chosen_assign, &H, &present)) {
            voiced = 0;
        } else {
            for (h = 1; h <= H; ++h) {
                if (st->chosen_assign[h] >= 0)
                    st->peak_harmonic[st->chosen_assign[h]] = h;
            }
        }
    }

    link_tracks(st, n_peaks, voiced ? f0 : 0.0);

    clear_output(out, rms);
    out->n_peaks = n_peaks < max_peaks ? n_peaks : max_peaks;
    for (i = 0; i < out->n_peaks; ++i) {
        int ti = st->peaks[i].track;
        peaks[i].freq_hz = st->peaks[i].freq;
        peaks[i].magnitude = st->peaks[i].mag;
        peaks[i].bin = st->peaks[i].bin;
        peaks[i].harmonic = voiced ? st->peak_harmonic[i] : 0;
        peaks[i].track_id = ti >= 0 ? st->tracks[ti].id : 0;
        peaks[i].track_age = ti >= 0 ? st->tracks[ti].age : 0;
    }

    if (voiced) {
        out->voiced = 1;
        out->f0_hz = f0;
        out->confidence = confidence;
        st->last_f0 = f0;
        st->gap = 0;
    } else {
        mark_unvoiced(st);
    }
    return voiced;
}

#else /* USE_SIMD */

int kiss_pitch_process(kiss_pitch_cfg cfg, const kiss_fft_scalar *frame,
                       kiss_pitch_frame *out, kiss_pitch_peak *peaks, int max_peaks)
{
    (void)cfg;
    (void)frame;
    (void)peaks;
    (void)max_peaks;
    if (out != NULL)
        clear_output(out, 0.0);
    return 0;
}

#endif /* USE_SIMD */
