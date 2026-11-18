/*
 *  Copyright (c) 2024, KISS FFT contributors. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * kfc_blockconv.h -- memory-bounded, block-wise fast convolution.
 *
 * Long signals are convolved with a fixed FIR filter without ever holding the
 * whole signal (or one giant FFT) in memory. Two equivalent, switchable
 * implementations are provided on top of the *unchanged* kiss_fft API:
 *
 *   KFC_BLOCKCONV_OVERLAP_ADD  (overlap-add,  OLA):
 *       The signal is cut into blocks of `block_len` samples, each block is
 *       zero-padded to an FFT of length N, multiplied by the spectrum of the
 *       zero-padded filter, inverse transformed and the length-(L-1) tails
 *       are added to the following block.
 *
 *   KFC_BLOCKCONV_OVERLAP_SAVE (overlap-save, OLS):
 *       Each FFT input frame is N samples long and overlaps the previous
 *       frame by (L-1) samples. Only the last (N-L+1) samples of every
 *       inverse transform are valid convolution output; the first (L-1)
 *       "wrap-around" samples are discarded.
 *
 * Both algorithms are linear (not circular) convolution and produce bit-for-bit
 * identical-length output of length (n_sig + n_filt - 1).
 *
 * The original kiss_fft / kfc functions are untouched.
 */
#ifndef KFC_BLOCKCONV_H
#define KFC_BLOCKCONV_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kfc_blockconv_state *kfc_blockconv_cfg;

typedef enum {
    KFC_BLOCKCONV_OVERLAP_ADD = 0,
    KFC_BLOCKCONV_OVERLAP_SAVE = 1
} kfc_blockconv_method;

/* Returned by kfc_blockconv_alloc when the request cannot be honoured. */
#define KFC_BLOCKCONV_INVALID   (-1) /* bad argument, e.g. block_len < filter length */
#define KFC_BLOCKCONV_NOMEM     (-2) /* an underlying allocation failed          */
#define KFC_BLOCKCONV_UNSUPPORTED (-3) /* build cannot satisfy the request (fixed point) */

/*
 * kfc_blockconv_alloc
 *
 * Build a streaming block-convolution plan for a filter `filt` of length
 * `n_filt` (the filter is copied internally and may be freed afterwards).
 *
 *   method     : KFC_BLOCKCONV_OVERLAP_ADD or KFC_BLOCKCONV_OVERLAP_SAVE
 *   block_len  : number of fresh signal samples consumed per FFT frame.
 *                Must satisfy block_len >= n_filt for OVERLAP_SAVE and
 *                block_len >= 1 for OVERLAP_ADD (a block shorter than the
 *                filter is legal for overlap-add: the FFT simply grows).
 *                If block_len <= 0 a sensible default of n_filt is used.
 *   zero_pad   : extra zero-padding factor. The FFT length is the smallest
 *                kiss_fft "fast" size >= zero_pad * (block_len + n_filt - 1);
 *                zero_pad <= 0 is treated as 1.
 *
 * The function never performs the transform, it only allocates and prepares
 * constant-size working storage. Returns a non-NULL opaque config, or NULL
 * (see KFC_BLOCKCONV_* for the reason, retrievable via kfc_blockconv_last_error).
 */
kfc_blockconv_cfg KISS_FFT_API kfc_blockconv_alloc(
        const kiss_fft_cpx *filt, int n_filt,
        kfc_blockconv_method method,
        int block_len, int zero_pad);

/*
 * kfc_blockconv_process
 *
 * Feed `n_in` new signal samples and receive up to `cap_out` convolution
 * samples in `out`. Returns the number of samples written (>= 0), or a
 * KFC_BLOCKCONV_* negative code. The function buffers internally, so `n_in`
 * may have any size, including 0. The caller never needs to resize buffers:
 * at most (n_in + n_filt - 1) samples can be produced by one call.
 */
int KISS_FFT_API kfc_blockconv_process(
        kfc_blockconv_cfg cfg,
        const kiss_fft_cpx *in, int n_in,
        kiss_fft_cpx *out, int cap_out);

/*
 * kfc_blockconv_flush
 *
 * Drives the buffered tail through a final, zero-padded FFT frame and writes
 * the remaining convolution samples. After flush the config is reset and can
 * be reused on a new signal. Returns samples written or a negative error code.
 */
int KISS_FFT_API kfc_blockconv_flush(
        kfc_blockconv_cfg cfg,
        kiss_fft_cpx *out, int cap_out);

/* Reset all streaming state but keep the prepared filter and FFT configs. */
void KISS_FFT_API kfc_blockconv_reset(kfc_blockconv_cfg cfg);

/* Free every resource owned by the config (NULL is tolerated). */
void KISS_FFT_API kfc_blockconv_free(kfc_blockconv_cfg cfg);

/* Accessors describing the geometry that was actually chosen. */
int KISS_FFT_API kfc_blockconv_nfft(const kfc_blockconv_cfg cfg);
int KISS_FFT_API kfc_blockconv_block_len(const kfc_blockconv_cfg cfg);
int KISS_FFT_API kfc_blockconv_filter_len(const kfc_blockconv_cfg cfg);
int KISS_FFT_API kfc_blockconv_last_error(void);

/*
 * One-shot convenience wrapper: convolves `sig` (n_sig) with the filter given
 * at alloc time and writes exactly (n_sig + n_filt - 1) samples to `out`.
 * Returns the output length, or a negative KFC_BLOCKCONV_* code.
 * `cap_out` must be at least (n_sig + n_filt - 1).
 */
int KISS_FFT_API kfc_blockconv_convolve(
        kfc_blockconv_cfg cfg,
        const kiss_fft_cpx *sig, int n_sig,
        kiss_fft_cpx *out, int cap_out);

/*
 * --------------------------------------------------------------------------
 * Memory instrumentation.
 *
 * The block convolver routes every internal allocation (including the two
 * kiss_fft configs) through these overridable hooks. Installing a tracking
 * pair lets a caller measure the *real* peak working-set size without
 * valgrind or any external tool. Default hooks are plain malloc/free.
 * Hook installation is process-global and not thread-safe by design.
 * --------------------------------------------------------------------------
 */
typedef struct {
    unsigned long long n_alloc;     /* number of allocation calls            */
    unsigned long long n_free;      /* number of free calls                  */
    unsigned long long current;     /* bytes currently outstanding           */
    unsigned long long peak;        /* high-water mark of `current`          */
} kfc_mem_stats;

void KISS_FFT_API kfc_set_alloc_hooks(
        void *(*malloc_fn)(size_t),
        void (*free_fn)(void *));
void KISS_FFT_API kfc_get_mem_stats(kfc_mem_stats *out);
void KISS_FFT_API kfc_reset_mem_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* KFC_BLOCKCONV_H */
