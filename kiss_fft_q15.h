/*
 *  16-bit fixed-point (Q15) radix-2 FFT for small fixed-point DSPs.
 *
 *  Input and output samples are int16_t Q1.15 values. All internal
 *  arithmetic is Q15: twiddle factors are int16_t Q15, products are
 *  accumulated in int32_t and every butterfly add/sub is saturated,
 *  counting the number of saturations ("overflows").
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_FFT_Q15_H
#define KISS_FFT_Q15_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int16_t r;
    int16_t i;
} kiss_fft_q15_cpx;

typedef struct kiss_fft_q15_state *kiss_fft_q15_cfg;

/* Scaling mode, see kiss_fft_q15_alloc(). */
#define KISS_FFT_Q15_SCALED   0 /* block-float: halve a stage when it would
                                   overflow; best accuracy, no saturation */
#define KISS_FFT_Q15_UNSCALED 1 /* no stage scaling; every butterfly that
                                   exceeds int16 saturates and is counted */

/*
 * Allocate a power-of-two Q15 FFT of length nfft.
 *
 * The Q15 twiddle table (nfft/2 complex entries) is generated here and
 * can be regenerated for any supported length, so no compile-time size
 * is baked in. As with kiss_fft_alloc, mem/lenmem allow the state to be
 * placed in a caller supplied buffer; otherwise the state is malloc'ed
 * and must be released with kiss_fft_q15_free().
 *
 * mode selects KISS_FFT_Q15_SCALED (recommended) or KISS_FFT_Q15_UNSCALED.
 *
 * Returns NULL (and stores the required size in *lenmem) on error.
 */
kiss_fft_q15_cfg kiss_fft_q15_alloc(int nfft, int inverse_fft, int mode,
                                    void *mem, size_t *lenmem);

void kiss_fft_q15_free(kiss_fft_q15_cfg cfg);

/*
 * Complex forward/inverse FFT, Q15 in and Q15 out.
 *
 * The transform is unnormalized, matching the floating-point kissfft
 * convention (forward then inverse gives nfft times the input). Each
 * butterfly saturates to the int16 range; saturations are accumulated
 * in the state and readable via kiss_fft_q15_overflows().
 * In KISS_FFT_Q15_SCALED mode the output may instead have been divided by
 * 2^s; kiss_fft_q15_scale_shift() reports s so callers can rescale.
 */
void kiss_fft_q15(kiss_fft_q15_cfg cfg,
                  const kiss_fft_q15_cpx *fin, kiss_fft_q15_cpx *fout);

/* Number of saturated butterfly adds/subs seen since reset. */
unsigned long kiss_fft_q15_overflows(const kiss_fft_q15_cfg cfg);
/* Number of stage halving (block-float scaling) shifts applied. */
int kiss_fft_q15_scale_shift(const kiss_fft_q15_cfg cfg);
void kiss_fft_q15_reset_overflows(kiss_fft_q15_cfg cfg);

/*
 * Error reporting against a floating-point reference spectrum.
 *
 * ref is the float FFT of the same input, expressed on the int16
 * scale (i.e. computed on the same sample values, no /32768 scaling).
 * SNR is returned in dB; the maximum absolute complex error
 * sqrt(dr^2+di^2) is stored in *max_err when non-NULL.
 */
double kiss_fft_q15_snr(const kiss_fft_q15_cpx *q15,
                        const double *ref_re, const double *ref_im,
                        int nfft, double *max_err);

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFT_Q15_H */
