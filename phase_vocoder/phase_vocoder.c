/*
 * Phase vocoder for time-scale modification, built on kissfft.
 *
 * Reference: J. Laroche and M. Dolson, "New phase-vocoder techniques for
 * real-time pitch shifting, chorusing, harmonizing, and other exotic audio
 * modifications", J. Audio Eng. Soc. 47(11), 1999.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "phase_vocoder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kiss_fft.h"
#include "kiss_fftr.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TWO_PI (2.0 * M_PI)

struct pv_state {
    int nfft;            /* FFT / window length N                    */
    int nbin;            /* number of positive-frequency bins N/2+1  */
    int hop_ana;         /* fixed analysis hop H_a                   */
    int hop_syn;         /* synthesis hop H_s = round(alpha * H_a)   */
    double alpha;        /* stretch factor; H_s / H_a                */

    double *window;      /* Hann analysis/synthesis window, length N */
    double norm_gain;    /* COLA normalization constant              */

    kiss_fftr_cfg fwd;   /* forward real FFT state                   */
    kiss_fftr_cfg inv;   /* inverse real FFT state                   */

    float *ana_buf;      /* N windowed input samples                 */
    kiss_fft_cpx *spec;  /* N/2+1 synthesis spectrum                 */
    float *ifft_buf;     /* N inverse-FFT output samples             */

    double *prev_phase;  /* analysis phase of previous frame         */
    double *syn_phase;   /* accumulated synthesis phase              */
    double *smooth_mag;  /* EMA-smoothed magnitudes for peak picking  */

    /* contiguous input buffer; grows to hold the full stream so every
       frame keeps its nfft-long look-back prefix available */
    float *inbuf;
    size_t inbuf_len;
    size_t inbuf_cap;

    /* overlap-add tail and its fill level */
    float *ola;
    int ola_len;
    /* write offset of the next synthesis frame inside `ola`; advances by
       hop_syn for every frame and is reset when the tail is drained */
    int ola_write;

    /* fractional input read position in absolute sample coordinates */
    double read_pos;
    long total_in;       /* total samples ever received              */
    long total_out;      /* total samples ever produced              */
    int first_frame;
    int flushed;
};

/* Wrap a phase difference into (-pi, pi]. */
static double princarg(double phase)
{
    phase = fmod(phase + M_PI, TWO_PI);
    if (phase < 0.0)
        phase += TWO_PI;
    return phase - M_PI;
}

pv_state *pv_alloc(int nfft, int hop_ana, double alpha)
{
    if (nfft <= 0 || (nfft & 1) != 0)
        return NULL;
    if (hop_ana <= 0 || hop_ana > nfft / 2)
        return NULL;
    if (!(alpha > 0.0) || !isfinite(alpha))
        return NULL;

    /* Synthesis hop tracks alpha. Keeping it integer makes the OLA lattice
       exact; at alpha == 1 both hops coincide for perfect reconstruction. */
    int hop_syn = (int)lround(alpha * (double)hop_ana);
    if (hop_syn <= 0)
        return NULL;

    pv_state *pv = (pv_state *)calloc(1, sizeof(pv_state));
    if (pv == NULL)
        return NULL;

    pv->nfft = nfft;
    pv->nbin = nfft / 2 + 1;
    pv->hop_ana = hop_ana;
    pv->hop_syn = hop_syn;
    pv->alpha = alpha;
    pv->first_frame = 1;

    int nbin = pv->nbin;
    size_t nf = (size_t)nfft;

    pv->window = (double *)malloc(nf * sizeof(double));
    pv->ana_buf = (float *)malloc(nf * sizeof(float));
    pv->ifft_buf = (float *)malloc(nf * sizeof(float));
    pv->spec = (kiss_fft_cpx *)malloc((size_t)nbin * sizeof(kiss_fft_cpx));
    pv->prev_phase = (double *)calloc((size_t)nbin, sizeof(double));
    pv->syn_phase = (double *)calloc((size_t)nbin, sizeof(double));
    pv->smooth_mag = (double *)calloc((size_t)nbin, sizeof(double));
    /* one synthesis frame plus a few hops of queue headroom; the OLA tail
       is drained every frame in both process and flush, so this is bounded */
    pv->ola = (float *)calloc((size_t)(nfft + 4 * hop_syn), sizeof(float));
    pv->inbuf_cap = (size_t)(2 * nfft);
    pv->inbuf = (float *)malloc(pv->inbuf_cap * sizeof(float));

    if (pv->window == NULL || pv->ana_buf == NULL || pv->ifft_buf == NULL ||
        pv->spec == NULL || pv->prev_phase == NULL || pv->syn_phase == NULL ||
        pv->smooth_mag == NULL ||
        pv->ola == NULL || pv->inbuf == NULL) {
        pv_free(pv);
        return NULL;
    }

    /* Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n/N)). The OLA
       normalization uses the synthesis hop. */
    double w2_sum = 0.0;
    for (int n = 0; n < nfft; ++n) {
        double w = 0.5 * (1.0 - cos(TWO_PI * n / nfft));
        pv->window[n] = w;
        if (n % hop_syn == 0)
            w2_sum += w * w;
    }
    /* kiss_fftri returns N times the time signal; scaling each frame by
       1 / (N * w2_sum) makes the overlap-add gain exactly 1 in steady
       state. Requires N / hop_syn >= 2 (>= 75% overlap recommended). */
    pv->norm_gain = 1.0 / (nfft * w2_sum);

    pv->fwd = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    pv->inv = kiss_fftr_alloc(nfft, 1, NULL, NULL);
    if (pv->fwd == NULL || pv->inv == NULL) {
        pv_free(pv);
        return NULL;
    }

    return pv;
}

void pv_free(pv_state *pv)
{
    if (pv == NULL)
        return;
    free(pv->window);
    free(pv->ana_buf);
    free(pv->ifft_buf);
    free(pv->spec);
    free(pv->prev_phase);
    free(pv->syn_phase);
    free(pv->smooth_mag);
    free(pv->ola);
    free(pv->inbuf);
    if (pv->fwd != NULL)
        kiss_fftr_free(pv->fwd);
    if (pv->inv != NULL)
        kiss_fftr_free(pv->inv);
    free(pv);
}

/*
 * Window, transform and phase-propagate one analysis frame starting at
 * inbuf offset `start` (zeros past the end of inbuf), then overlap-add.
 *
 * Phase locking (Laroche & Dolson 1999): bins that are local magnitude
 * maxima ("peaks") drive phase propagation; every non-peak bin rotates
 * rigidly with the nearest peak, preserving the analysis phase offset.
 * This keeps the spectral shape coherent and suppresses the phasiness /
 * amplitude modulation of independent per-bin phase propagation.
 */
static void pv_emit_frame(pv_state *pv, size_t start)
{
    const int nfft = pv->nfft;
    const int nbin = pv->nbin;
    const int hop_ana = pv->hop_ana;
    const int hop_syn = pv->hop_syn;

    for (int n = 0; n < nfft; ++n) {
        float x = 0.0f;
        if (start + (size_t)n < pv->inbuf_len)
            x = pv->inbuf[start + n];
        pv->ana_buf[n] = (float)(x * pv->window[n]);
    }
    kiss_fftr(pv->fwd, pv->ana_buf, pv->spec);

    double mag[nbin], phase[nbin];
    for (int k = 0; k < nbin; ++k) {
        mag[k] = hypot(pv->spec[k].r, pv->spec[k].i);
        phase[k] = atan2(pv->spec[k].i, pv->spec[k].r);
    }

    if (pv->first_frame) {
        /* Seed synthesis phases with the analysis phases. */
        for (int k = 0; k < nbin; ++k) {
            pv->prev_phase[k] = phase[k];
            pv->syn_phase[k] = phase[k];
            pv->smooth_mag[k] = mag[k];
        }
        pv->first_frame = 0;
    } else {
        /* Peak detection runs on an EMA-smoothed magnitude so a tone
           landing between two bins cannot flip the peak identity from
           frame to frame; the current-frame phase/magnitude still drive
           synthesis. */
        const double ema = 0.7;
        double smag[nbin];
        for (int k = 0; k < nbin; ++k) {
            pv->smooth_mag[k] = ema * pv->smooth_mag[k] +
                                (1.0 - ema) * mag[k];
            smag[k] = pv->smooth_mag[k];
        }

        /* Mark local magnitude maxima; DC and Nyquist bins never peak. */
        int is_peak[nbin];
        memset(is_peak, 0, sizeof(is_peak));
        for (int k = 1; k < nbin - 1; ++k) {
            if (smag[k] > smag[k - 1] && smag[k] >= smag[k + 1] &&
                smag[k] > 1e-30)
                is_peak[k] = 1;
        }

        /* Rotation applied to every region, indexed by its peak bin. */
        double peak_rot[nbin];
        memset(peak_rot, 0, sizeof(peak_rot));
        for (int p = 1; p < nbin - 1; ++p) {
            if (!is_peak[p])
                continue;
            double omega = TWO_PI * p / nfft;
            double delta = princarg(phase[p] - pv->prev_phase[p]
                                    - omega * hop_ana);
            double inst_freq = omega + delta / hop_ana;
            pv->syn_phase[p] += inst_freq * hop_syn;
            /* Rigid rotation of this peak's whole region: the synthesis
               phase advance minus the expected advance at the peak. */
            peak_rot[p] = princarg(pv->syn_phase[p] - phase[p]);
        }

        /* Nearest peak to the left (sweep) and then to the right. */
        int region[nbin];
        int last_peak = -1;
        for (int k = 0; k < nbin; ++k) {
            if (is_peak[k])
                last_peak = k;
            region[k] = last_peak;
        }
        int next_peak = -1;
        for (int k = nbin - 1; k >= 0; --k) {
            if (is_peak[k]) {
                next_peak = k;
            } else if (region[k] == -1) {
                region[k] = next_peak;
            } else if (next_peak != -1 &&
                       (k - region[k]) > (next_peak - k)) {
                region[k] = next_peak;
            }
        }

        for (int k = 0; k < nbin; ++k) {
            if (is_peak[k])
                continue;
            int p = region[k];
            if (p >= 0) {
                /* Lock to the region peak: keep the analysis phase
                   offset and apply the peak's rigid rotation. */
                pv->syn_phase[k] = phase[k] + peak_rot[p];
            } else {
                /* No peak anywhere (near-silent spectrum): fall back to
                   independent propagation to avoid a frozen phase. */
                double omega = TWO_PI * k / nfft;
                double delta = princarg(phase[k] - pv->prev_phase[k]
                                        - omega * hop_ana);
                double inst_freq = omega + delta / hop_ana;
                pv->syn_phase[k] += inst_freq * hop_syn;
            }
        }

        for (int k = 0; k < nbin; ++k) {
            pv->spec[k].r = (kiss_fft_scalar)(mag[k] * cos(pv->syn_phase[k]));
            pv->spec[k].i = (kiss_fft_scalar)(mag[k] * sin(pv->syn_phase[k]));
            pv->prev_phase[k] = phase[k];
        }
    }

    /* Inverse FFT, window again, overlap-add into the tail at the current
       synthesis write offset (frames advance by hop_syn each). */
    kiss_fftri(pv->inv, pv->spec, pv->ifft_buf);

    float *ola = pv->ola;
    int off = pv->ola_write;
    for (int n = 0; n < nfft; ++n) {
        double y = pv->ifft_buf[n] * pv->window[n] * pv->norm_gain;
        int j = off + n;
        if (j < pv->ola_len)
            ola[j] += (float)y;
        else
            ola[j] = (float)y;
    }
    pv->ola_len = off + nfft;
    pv->ola_write += pv->hop_syn;
}

/* Pop the oldest hop_syn samples out of the OLA tail; returns count. */
static int pv_drain(pv_state *pv, float *out)
{
    int h = pv->hop_syn;
    int take = h;
    if (pv->ola_len < take)
        take = pv->ola_len;
    if (take > 0)
        memcpy(out, pv->ola, (size_t)take * sizeof(float));

    int remain = pv->ola_len - take;
    memmove(pv->ola, pv->ola + take, (size_t)remain * sizeof(float));
    pv->ola_len = remain;
    pv->ola_write -= take;
    if (pv->ola_write < 0)
        pv->ola_write = 0;
    return take;
}

/* Append `nin` samples, growing the rolling input buffer if needed. */
static int pv_append_input(pv_state *pv, const float *in, size_t nin)
{
    size_t need = pv->inbuf_len + nin;
    if (need > pv->inbuf_cap) {
        size_t cap = pv->inbuf_cap;
        while (cap < need)
            cap *= 2;
        float *p = (float *)realloc(pv->inbuf, cap * sizeof(float));
        if (p == NULL)
            return -1;
        pv->inbuf = p;
        pv->inbuf_cap = cap;
    }
    memcpy(pv->inbuf + pv->inbuf_len, in, nin * sizeof(float));
    pv->inbuf_len += nin;
    return 0;
}

long pv_process(pv_state *pv, const float *in, size_t nin, float *out)
{
    if (pv == NULL || (nin > 0 && in == NULL) || out == NULL || pv->flushed)
        return -1;

    if (pv_append_input(pv, in, nin) != 0)
        return -1;
    pv->total_in += (long)nin;

    long produced = 0;

    /* Emit only frames fully backed by the input received so far; frames
       whose window extends past the current end are deferred to flush().
       The input buffer is kept contiguous rather than trimmed, because a
       later frame starting inside the buffer still needs an nfft-long
       look-back prefix. */
    while (1) {
        long start = (long)pv->read_pos;
        if (start < 0 || (size_t)start + (size_t)pv->nfft > pv->inbuf_len)
            break;

        pv_emit_frame(pv, (size_t)start);
        pv->read_pos += pv->hop_ana;

        if (pv->ola_len >= pv->hop_syn) {
            long got = pv_drain(pv, out + produced);
            produced += got;
            pv->total_out += got;
        }
    }
    return produced;
}

long pv_flush(pv_state *pv, float *out)
{
    if (pv == NULL || out == NULL || pv->flushed)
        return -1;
    pv->flushed = 1;

    long produced = 0;
    long target = lround(pv->alpha * (double)pv->total_in);

    /* Emit each remaining frame (zero-padded past the last input sample)
       and immediately drain the synthesis hop, keeping the OLA queue
       bounded. Stop once the ideal target length is reached. */
    while (1) {
        long start = (long)pv->read_pos;
        if (start < 0 || start >= (long)pv->inbuf_len)
            break;

        pv_emit_frame(pv, (size_t)start);
        pv->read_pos += pv->hop_ana;

        if (pv->ola_len >= pv->hop_syn && pv->total_out < target) {
            long room = target - pv->total_out;
            if (room >= pv->hop_syn) {
                long got = pv_drain(pv, out + produced);
                produced += got;
                pv->total_out += got;
            } else {
                memcpy(out + produced, pv->ola,
                       (size_t)room * sizeof(float));
                produced += room;
                pv->total_out += room;
                break;
            }
        }
    }

    /* Drain any remaining tail up to the target length. */
    while (pv->ola_len > 0 && pv->total_out < target) {
        long room = target - pv->total_out;
        if (room >= pv->hop_syn && pv->ola_len >= pv->hop_syn) {
            long got = pv_drain(pv, out + produced);
            produced += got;
            pv->total_out += got;
        } else {
            long got = room < pv->ola_len ? room : pv->ola_len;
            memcpy(out + produced, pv->ola, (size_t)got * sizeof(float));
            produced += got;
            pv->total_out += got;
            break;
        }
    }
    return produced;
}

long pv_expected_output_len(const pv_state *pv, long input_len)
{
    if (pv == NULL || input_len <= 0)
        return 0;
    /* Target duration is exactly alpha times the input; flush truncates
       the zero-padded tail to this length. */
    long target = lround(pv->alpha * (double)input_len);
    return target > 0 ? target : 0;
}
