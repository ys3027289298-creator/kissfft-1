/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kfc.h"

typedef struct cached_fft *kfc_cfg;

struct cached_fft
{
    int nfft;
    int inverse;
    kiss_fft_cfg cfg;
    kfc_cfg next;
};

static kfc_cfg cache_root=NULL;
static int ncached=0;

static kiss_fft_cfg find_cached_fft(int nfft,int inverse)
{
    size_t len = 0;
    kfc_cfg  cur=cache_root;
    kfc_cfg  prev=NULL;
    while ( cur ) {
        if ( cur->nfft == nfft && inverse == cur->inverse )
            break;/*found the right node*/
        prev = cur;
        cur = prev->next;
    }
    if (cur== NULL) {
        /* no cached node found, need to create a new one*/
        kiss_fft_alloc(nfft,inverse,0,&len);
        if (len == 0)
            return NULL; /* invalid nfft */
#ifdef USE_SIMD
        int padding = (16-sizeof(struct cached_fft)) & 15;
        // make sure the cfg aligns on a 16 byte boundary
        len += padding;
#endif
        cur = (kfc_cfg)KISS_FFT_MALLOC((sizeof(struct cached_fft) + len ));
        if (cur == NULL)
            return NULL;
        cur->cfg = (kiss_fft_cfg)(cur+1);
#ifdef USE_SIMD
        cur->cfg = (kiss_fft_cfg) ((char*)(cur+1)+padding);
#endif
        kiss_fft_alloc(nfft,inverse,cur->cfg,&len);
        cur->nfft=nfft;
        cur->inverse=inverse;
        cur->next = NULL;
        if ( prev )
            prev->next = cur;
        else
            cache_root = cur;
        ++ncached;
    }
    return cur->cfg;
}

void kfc_cleanup(void)
{
    kfc_cfg  cur=cache_root;
    kfc_cfg  next=NULL;
    while (cur){
        next = cur->next;
        free(cur);
        cur=next;
    }
    ncached=0;
    cache_root = NULL;
}
void kfc_fft(int nfft, const kiss_fft_cpx * fin,kiss_fft_cpx * fout)
{
    kiss_fft( find_cached_fft(nfft,0),fin,fout );
}

void kfc_ifft(int nfft, const kiss_fft_cpx * fin,kiss_fft_cpx * fout)
{
    kiss_fft( find_cached_fft(nfft,1),fin,fout );
}

/* ===========================================================================
 *  Memory-bounded block (streaming) fast convolution
 *
 *  Everything below is new functionality built strictly on top of the
 *  unchanged kiss_fft public API. It supports two switchable, mathematically
 *  equivalent formulations -- overlap-add (OLA) and overlap-save (OLS) --
 *  whose working memory depends only on the chosen block/FFT length, never on
 *  the total signal length. See kfc_blockconv.h for the contract.
 * ===========================================================================
 */

#include "kfc_blockconv.h"
#include "_kiss_fft_guts.h"
#include <limits.h>

/* ---- overridable allocation hooks with lightweight accounting ----------- */

static void *kfc_def_malloc(size_t n)            { return malloc(n); }
static void  kfc_def_free(void *p)               { free(p); }

static void *(*kfc_hook_malloc)(size_t) = kfc_def_malloc;
static void  (*kfc_hook_free)(void *)   = kfc_def_free;

static kfc_mem_stats kfc_mem = {0, 0, 0, 0};
static int kfc_last_error = 0;

void kfc_set_alloc_hooks(void *(*m)(size_t), void (*f)(void *))
{
    kfc_hook_malloc = m ? m : kfc_def_malloc;
    kfc_hook_free   = f ? f : kfc_def_free;
}

void kfc_get_mem_stats(kfc_mem_stats *out)
{
    if (out)
        *out = kfc_mem;
}

void kfc_reset_mem_stats(void)
{
    kfc_mem.n_alloc = 0;
    kfc_mem.n_free  = 0;
    kfc_mem.current = 0;
    kfc_mem.peak    = 0;
}

/* Each tracked block carries a small header so free() can reconcile the exact
 * byte count, which is what makes the measured peak match the analytic bound. */
typedef struct { size_t bytes; } kfc_hdr;

static void *kfc_alloc_tracked(size_t n)
{
    kfc_hdr *h = (kfc_hdr *)kfc_hook_malloc(sizeof(kfc_hdr) + n);
    if (!h)
        return NULL;
    h->bytes = n;
    ++kfc_mem.n_alloc;
    kfc_mem.current += sizeof(kfc_hdr) + n;
    if (kfc_mem.current > kfc_mem.peak)
        kfc_mem.peak = kfc_mem.current;
    return (void *)(h + 1);
}

static void kfc_free_tracked(void *p)
{
    kfc_hdr *h;
    size_t block;
    if (!p)
        return;
    h = ((kfc_hdr *)p) - 1;
    block = sizeof(kfc_hdr) + h->bytes;
    kfc_hook_free(h);
    ++kfc_mem.n_free;
    if (block <= kfc_mem.current)
        kfc_mem.current -= block;
    else
        kfc_mem.current = 0; /* defensive: never leave a stuck counter */
}

struct kfc_blockconv_state {
    kfc_blockconv_method method;
    int n_filt;          /* L                                                  */
    int block_len;       /* B: fresh signal samples per frame                  */
    int nfft;            /* N: FFT length (a 2,3,5-smooth "fast" size)         */
    int valid;           /* valid output per frame: B (OLA) or N-L+1 (OLS)     */
    int hist_len;        /* overlap width: L-1 (both methods)                  */
    kiss_fft_cfg fwd;    /* forward FFT state                                  */
    kiss_fft_cfg inv;    /* inverse FFT state                                  */
    kiss_fft_cpx *fbuf;  /* N samples: assembled time-domain FFT input         */
    kiss_fft_cpx *Xbuf;  /* N samples: frequency-domain signal block           */
    kiss_fft_cpx *Hbuf;  /* N samples: filter spectrum, pre-scaled by 1/N      */
    kiss_fft_cpx *obuf;  /* N samples: inverse-FFT result / frame output       */
    kiss_fft_cpx *ov;    /* OLA: (L-1) tail-add; OLS: (L-1) input history      */
    kiss_fft_cpx *tail;  /* OLS: sliding window of the last (L-1) real inputs  */
    int n_ov;            /* OLA: tail samples held; OLS: carried-output count  */
    int n_buf;           /* fresh signal samples buffered for the next frame   */
    int n_tail;          /* OLS: valid entries currently in the tail window     */
    int n_emitted;       /* OLS: total output samples produced so far          */
    int n_seen;          /* OLS: total signal samples ingested so far          */
    int frame;           /* number of FFT frames already emitted               */
};

kfc_blockconv_cfg kfc_blockconv_alloc(
        const kiss_fft_cpx *filt, int n_filt,
        kfc_blockconv_method method,
        int block_len, int zero_pad)
{
    struct kfc_blockconv_state *st;
    int N, B, want;
    size_t cfg_fwd_len = 0, cfg_inv_len = 0;

    kfc_last_error = 0;

#ifdef FIXED_POINT
    /* The reference path divides the filter spectrum by N; fixed-point builds
     * instead scale inside the FFT butterflies. Reject them explicitly rather
     * than silently returning a wrong result. */
    kfc_last_error = KFC_BLOCKCONV_UNSUPPORTED;
    return NULL;
#else

    if (filt == NULL || n_filt <= 0) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return NULL;
    }
    if (method != KFC_BLOCKCONV_OVERLAP_ADD &&
        method != KFC_BLOCKCONV_OVERLAP_SAVE) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return NULL;
    }
    if (zero_pad <= 0)
        zero_pad = 1;
    if (block_len <= 0)
        block_len = n_filt;                 /* default: one filter per block */

    B = block_len;

    /*
     * Explicit, documented handling of "block shorter than the filter":
     *  - overlap-add supports it: linear convolution of a B-sample block and
     *    an L-sample filter needs N >= B+L-1, so the FFT simply grows.
     *  - overlap-save cannot: its premise is keeping L-1 history and still
     *    having N-(L-1) >= 1 valid output sample. Refuse the request.
     */
    if (method == KFC_BLOCKCONV_OVERLAP_SAVE && B < n_filt) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return NULL;
    }

    /* N = smallest fast size >= zero_pad * (B + L - 1), with overflow checks */
    want = B + n_filt - 1;
    if (want < B || zero_pad > (INT_MAX - 1) / want) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return NULL;
    }
    want *= zero_pad;
    N = kiss_fft_next_fast_size(want);
    if (N < want || N <= 0) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return NULL;
    }

    st = (struct kfc_blockconv_state *)kfc_alloc_tracked(sizeof(*st));
    if (!st) {
        kfc_last_error = KFC_BLOCKCONV_NOMEM;
        return NULL;
    }
    memset(st, 0, sizeof(*st));
    st->method    = method;
    st->n_filt    = n_filt;
    st->block_len = B;
    st->nfft      = N;
    st->hist_len  = n_filt - 1;
    st->valid     = (method == KFC_BLOCKCONV_OVERLAP_ADD) ? B : (N - n_filt + 1);

    /* size queries first so an unsupported FFT size fails before any work.
     * With mem=NULL and a non-NULL lenmem, kiss_fft_alloc is a size query:
     * it returns NULL but writes the required byte count into the length. */
    (void)kiss_fft_alloc(N, 0, NULL, &cfg_fwd_len);
    (void)kiss_fft_alloc(N, 1, NULL, &cfg_inv_len);
    if (cfg_fwd_len == 0 || cfg_inv_len == 0) {
        kfc_free_tracked(st);
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return NULL;
    }

    st->fwd  = (kiss_fft_cfg)kfc_alloc_tracked(cfg_fwd_len);
    st->inv  = (kiss_fft_cfg)kfc_alloc_tracked(cfg_inv_len);
    st->fbuf = (kiss_fft_cpx *)kfc_alloc_tracked((size_t)N * sizeof(kiss_fft_cpx));
    st->Xbuf = (kiss_fft_cpx *)kfc_alloc_tracked((size_t)N * sizeof(kiss_fft_cpx));
    st->Hbuf = (kiss_fft_cpx *)kfc_alloc_tracked((size_t)N * sizeof(kiss_fft_cpx));
    st->obuf = (kiss_fft_cpx *)kfc_alloc_tracked((size_t)N * sizeof(kiss_fft_cpx));
    st->ov   = st->hist_len > 0
             ? (kiss_fft_cpx *)kfc_alloc_tracked((size_t)st->hist_len * sizeof(kiss_fft_cpx))
             : NULL;
    st->tail = st->hist_len > 0
             ? (kiss_fft_cpx *)kfc_alloc_tracked((size_t)st->hist_len * sizeof(kiss_fft_cpx))
             : NULL;

    if (!st->fwd || !st->inv || !st->fbuf || !st->Xbuf ||
        !st->Hbuf || !st->obuf ||
        (st->hist_len > 0 && (!st->ov || !st->tail))) {
        kfc_blockconv_free(st);
        kfc_last_error = KFC_BLOCKCONV_NOMEM;
        return NULL;
    }

    if (kiss_fft_alloc(N, 0, st->fwd, &cfg_fwd_len) == NULL ||
        kiss_fft_alloc(N, 1, st->inv, &cfg_inv_len) == NULL) {
        kfc_blockconv_free(st);
        kfc_last_error = KFC_BLOCKCONV_NOMEM;
        return NULL;
    }

    /* prepare the filter once: zero-pad to N, transform, pre-scale by 1/N */
    memset(st->fbuf, 0, (size_t)N * sizeof(kiss_fft_cpx));
    memcpy(st->fbuf, filt, (size_t)n_filt * sizeof(kiss_fft_cpx));
    kiss_fft(st->fwd, st->fbuf, st->Hbuf);
    {
        kiss_fft_scalar scale = (kiss_fft_scalar)1 / (kiss_fft_scalar)N;
        int k;
        for (k = 0; k < N; ++k) {
            st->Hbuf[k].r *= scale;
            st->Hbuf[k].i *= scale;
        }
    }

    memset(st->fbuf, 0, (size_t)N * sizeof(kiss_fft_cpx));
    if (st->ov)
        memset(st->ov, 0, (size_t)st->hist_len * sizeof(kiss_fft_cpx));
    st->n_ov = 0;
    st->n_buf = 0;
    st->n_tail = 0;
    st->n_emitted = 0;
    st->n_seen = 0;
    st->frame = 0;
    return st;
#endif /* FIXED_POINT */
}

void kfc_blockconv_reset(kfc_blockconv_cfg cfg)
{
    if (!cfg)
        return;
    cfg->n_ov = 0;
    cfg->n_buf = 0;
    cfg->n_tail = 0;
    cfg->n_emitted = 0;
    cfg->n_seen = 0;
    cfg->frame = 0;
    if (cfg->ov && cfg->hist_len > 0)
        memset(cfg->ov, 0, (size_t)cfg->hist_len * sizeof(kiss_fft_cpx));
    memset(cfg->fbuf, 0, (size_t)cfg->nfft * sizeof(kiss_fft_cpx));
}

void kfc_blockconv_free(kfc_blockconv_cfg cfg)
{
    if (!cfg)
        return;
    /* The kiss_fft states live inside our tracked buffers; they need no
     * destructor of their own, so everything is released through one path. */
    kfc_free_tracked(cfg->fwd);
    kfc_free_tracked(cfg->inv);
    kfc_free_tracked(cfg->fbuf);
    kfc_free_tracked(cfg->Xbuf);
    kfc_free_tracked(cfg->Hbuf);
    kfc_free_tracked(cfg->obuf);
    kfc_free_tracked(cfg->ov);
    kfc_free_tracked(cfg->tail);
    kfc_free_tracked(cfg);
}

int kfc_blockconv_nfft(const kfc_blockconv_cfg cfg)
{ return cfg ? cfg->nfft : 0; }

int kfc_blockconv_block_len(const kfc_blockconv_cfg cfg)
{ return cfg ? cfg->block_len : 0; }

int kfc_blockconv_filter_len(const kfc_blockconv_cfg cfg)
{ return cfg ? cfg->n_filt : 0; }

int kfc_blockconv_last_error(void)
{ return kfc_last_error; }

/* one FFT frame: multiply spectra in place and inverse transform */
static void kfc_emit_frame(struct kfc_blockconv_state *st)
{
    int k;
    kiss_fft_cpx t;
    kiss_fft(st->fwd, st->fbuf, st->Xbuf);
    for (k = 0; k < st->nfft; ++k) {
        C_MUL(t, st->Xbuf[k], st->Hbuf[k]);
        st->Xbuf[k] = t;
    }
    kiss_fft(st->inv, st->Xbuf, st->obuf);
    ++st->frame;
}

int kfc_blockconv_process(
        kfc_blockconv_cfg st,
        const kiss_fft_cpx *in, int n_in,
        kiss_fft_cpx *out, int cap_out)
{
    int produced = 0;

    if (!st || !in || !out || cap_out < 0) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return KFC_BLOCKCONV_INVALID;
    }
    kfc_last_error = 0;

    if (st->method == KFC_BLOCKCONV_OVERLAP_ADD) {
        int pos = 0;
        while (pos < n_in) {
            int take = st->block_len - st->n_buf;
            int k;
            if (take > n_in - pos)
                take = n_in - pos;
            memcpy(st->fbuf + st->n_buf, in + pos,
                   (size_t)take * sizeof(kiss_fft_cpx));
            st->n_buf += take;
            pos += take;

            if (st->n_buf == st->block_len) {
                /* zero the padding region beyond the fresh block */
                for (k = st->block_len; k < st->nfft; ++k) {
                    st->fbuf[k].r = 0;
                    st->fbuf[k].i = 0;
                }
                kfc_emit_frame(st);
                /* fold in the previous frame's (L-1)-sample tail */
                for (k = 0; k < st->n_ov; ++k) {
                    st->obuf[k].r += st->ov[k].r;
                    st->obuf[k].i += st->ov[k].i;
                }
                if (produced + st->block_len > cap_out) {
                    kfc_last_error = KFC_BLOCKCONV_INVALID;
                    return KFC_BLOCKCONV_INVALID;
                }
                memcpy(out + produced, st->obuf,
                       (size_t)st->block_len * sizeof(kiss_fft_cpx));
                produced += st->block_len;
                /* retain the (L-1)-sample tail for the next frame */
                st->n_ov = st->hist_len;
                if (st->n_ov > 0)
                    memcpy(st->ov, st->obuf + st->block_len,
                           (size_t)st->n_ov * sizeof(kiss_fft_cpx));
                st->n_buf = 0;
            }
        }
    } else { /* OVERLAP_SAVE
             * Xbuf[0..n_buf) is a queue of fresh signal samples not yet turned
             * into a frame; ov[0..hist_len) holds the L-1 input samples that
             * precede the queue (zero-primed after reset). Whenever the queue
             * reaches valid = N-(L-1) samples, an N-point frame is assembled in
             * fbuf as [ history | valid fresh samples ], transformed out-of-place
             * (fbuf -> Xbuf -> obuf), and its first L-1 wrap-around outputs are
             * discarded; the `valid` valid outputs are emitted. The frame's last
             * L-1 inputs become the next frame's history. The flush() companion
             * handles the final partial frame and its wrapped tail. */
        int pos = 0;
        while (pos < n_in) {
            int take = st->valid - st->n_buf;
            int k;
            if (take > n_in - pos)
                take = n_in - pos;
            memcpy(st->Xbuf + st->n_buf, in + pos,
                   (size_t)take * sizeof(kiss_fft_cpx));
            /* maintain the sliding window of the last hist_len real inputs */
            for (k = 0; k < take; ++k) {
                if (st->hist_len == 0)
                    break;
                if (st->n_tail == st->hist_len)
                    memmove(st->tail, st->tail + 1,
                            (size_t)(st->hist_len - 1) * sizeof(kiss_fft_cpx));
                else
                    ++st->n_tail;
                st->tail[st->n_tail - 1] = in[pos + k];
            }
            st->n_buf += take;
            st->n_seen += take;
            pos += take;

            if (st->n_buf == st->valid) {
                /* assemble the N-point frame in fbuf (fbuf and Xbuf stay
                 * distinct because kiss_fft is strictly out-of-place) */
                memset(st->fbuf, 0, (size_t)st->nfft * sizeof(kiss_fft_cpx));
                if (st->hist_len > 0)
                    memcpy(st->fbuf, st->ov,
                           (size_t)st->hist_len * sizeof(kiss_fft_cpx));
                memcpy(st->fbuf + st->hist_len, st->Xbuf,
                       (size_t)st->valid * sizeof(kiss_fft_cpx));
                kfc_emit_frame(st);
                if (produced + st->valid > cap_out) {
                    kfc_last_error = KFC_BLOCKCONV_INVALID;
                    return KFC_BLOCKCONV_INVALID;
                }
                /* keep only the last N-(L-1)=valid samples; the head
                 * obuf[0..hist_len) is wrap-around and is dropped here */
                memcpy(out + produced, st->obuf + st->hist_len,
                       (size_t)st->valid * sizeof(kiss_fft_cpx));
                produced += st->valid;
                st->n_emitted += st->valid;
                /* next frame's input history = last L-1 inputs of this frame */
                if (st->hist_len > 0)
                    memcpy(st->ov, st->fbuf + st->nfft - st->hist_len,
                           (size_t)st->hist_len * sizeof(kiss_fft_cpx));
                st->n_buf = 0;
                (void)k;
            }
        }
    }
    return produced;
}

int kfc_blockconv_flush(
        kfc_blockconv_cfg st,
        kiss_fft_cpx *out, int cap_out)
{
    int produced = 0, k;

    if (!st || !out || cap_out < 0) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return KFC_BLOCKCONV_INVALID;
    }
    kfc_last_error = 0;

    if (st->method == KFC_BLOCKCONV_OVERLAP_ADD) {
        if (st->n_buf > 0) {
            int n;
            /* final partial block; explicit zero padding beyond n_buf */
            for (k = st->n_buf; k < st->nfft; ++k) {
                st->fbuf[k].r = 0;
                st->fbuf[k].i = 0;
            }
            kfc_emit_frame(st);
            for (k = 0; k < st->n_ov; ++k) {
                st->obuf[k].r += st->ov[k].r;
                st->obuf[k].i += st->ov[k].i;
            }
            n = st->n_buf + st->hist_len;   /* span of the last frame */
            if (n > cap_out) {
                kfc_last_error = KFC_BLOCKCONV_INVALID;
                return KFC_BLOCKCONV_INVALID;
            }
            memcpy(out, st->obuf, (size_t)n * sizeof(kiss_fft_cpx));
            produced = n;
            st->n_ov = 0;
            st->n_buf = 0;
        } else if (st->n_ov > 0) {
            /* signal length was an exact multiple of B: only the tail remains */
            if (st->n_ov > cap_out) {
                kfc_last_error = KFC_BLOCKCONV_INVALID;
                return KFC_BLOCKCONV_INVALID;
            }
            memcpy(out, st->ov, (size_t)st->n_ov * sizeof(kiss_fft_cpx));
            produced = st->n_ov;
            st->n_ov = 0;
        }
    } else { /* OVERLAP_SAVE
             * At this point ov[0..hist_len) holds the L-1 input samples that
             * precede the queue and Xbuf[0..n_buf) holds the queued fresh data
             * (n_buf may be 0 when the signal ended on a full-frame boundary).
             * The full convolution has n_seen + (L-1) samples and `remaining`
             * are still owed. Two steps, matching the textbook construction:
             *  1. if samples are queued, emit their n_buf real outputs from a
             *     padded frame [ ov | queue | zeros ];
             *  2. emit a ring-out frame whose history is the last L-1 real
             *     inputs and whose valid region is all zero; its first L-1
             *     valid outputs are the filter tail. When fewer than L-1 real
             *     inputs exist (signal shorter than the filter), the missing
             *     leading history slots stay zero. */
        int total_out = st->n_seen + st->hist_len;
        int remaining = total_out - st->n_emitted;
        if (remaining > 0) {
            int hlen = st->hist_len;
            int queued = st->n_buf;          /* remember before step 1 clears it */
            if (remaining > cap_out) {
                kfc_last_error = KFC_BLOCKCONV_INVALID;
                return KFC_BLOCKCONV_INVALID;
            }

            /* (1) partial frame: the queued samples' real outputs */
            if (queued > 0) {
                memset(st->fbuf, 0,
                       (size_t)st->nfft * sizeof(kiss_fft_cpx));
                if (hlen > 0)
                    memcpy(st->fbuf, st->ov,
                           (size_t)hlen * sizeof(kiss_fft_cpx));
                memcpy(st->fbuf + hlen, st->Xbuf,
                       (size_t)queued * sizeof(kiss_fft_cpx));
                kfc_emit_frame(st);
                memcpy(out, st->obuf + hlen,
                       (size_t)queued * sizeof(kiss_fft_cpx));
                produced = queued;
                st->n_buf = 0;
            }

            /* (2) ring-out history. tail[] is maintained incrementally in
             * process() as the sliding window of the last hlen real inputs.
             * Right-justify its n_tail valid entries so that any missing
             * leading inputs (when n_seen < hlen) become explicit zeros. */
            if (hlen > 0) {
                int tail_n = remaining - produced;     /* samples still owed */
                int pad = hlen - st->n_tail;
                int z;
                if (pad > 0) {
                    for (z = hlen - 1; z >= pad; --z)
                        st->tail[z] = st->tail[z - pad];
                    for (z = 0; z < pad; ++z) {
                        st->tail[z].r = 0;
                        st->tail[z].i = 0;
                    }
                }

                /* ring-out frame [ tail history | zero valid region ] */
                memset(st->fbuf, 0,
                       (size_t)st->nfft * sizeof(kiss_fft_cpx));
                memcpy(st->fbuf, st->tail,
                       (size_t)hlen * sizeof(kiss_fft_cpx));
                kfc_emit_frame(st);
                if (tail_n > 0) {
                    if (produced + tail_n > cap_out) {
                        kfc_last_error = KFC_BLOCKCONV_INVALID;
                        return KFC_BLOCKCONV_INVALID;
                    }
                    memcpy(out + produced, st->obuf + hlen,
                           (size_t)tail_n * sizeof(kiss_fft_cpx));
                    produced += tail_n;
                }
            }
        }
    }

    kfc_blockconv_reset(st);
    return produced;
}

int kfc_blockconv_convolve(
        kfc_blockconv_cfg st,
        const kiss_fft_cpx *sig, int n_sig,
        kiss_fft_cpx *out, int cap_out)
{
    int total, got, tail;

    if (!st || !sig || n_sig < 0 || !out) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return KFC_BLOCKCONV_INVALID;
    }
    kfc_blockconv_reset(st);
    /* an empty signal convolves to an empty result, regardless of filter length */
    if (n_sig == 0)
        return 0;
    total = n_sig + st->n_filt - 1;
    if (cap_out < total) {
        kfc_last_error = KFC_BLOCKCONV_INVALID;
        return KFC_BLOCKCONV_INVALID;
    }
    got = kfc_blockconv_process(st, sig, n_sig, out, cap_out);
    if (got < 0)
        return got;
    tail = kfc_blockconv_flush(st, out + got, cap_out - got);
    if (tail < 0)
        return tail;
    got += tail;
    return got;
}

#ifdef KFC_TEST
static void check(int nc)
{
    if (ncached != nc) {
        fprintf(stderr,"ncached should be %d,but it is %d\n",nc,ncached);
        exit(1);
    }
}

int main(void)
{
    kiss_fft_cpx buf1[1024],buf2[1024];
    memset(buf1,0,sizeof(buf1));
    check(0);
    kfc_fft(512,buf1,buf2);
    check(1);
    kfc_fft(512,buf1,buf2);
    check(1);
    kfc_ifft(512,buf1,buf2);
    check(2);
    kfc_cleanup();
    check(0);
    return 0;
}
#endif
