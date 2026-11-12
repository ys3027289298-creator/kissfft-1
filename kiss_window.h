/*
 *  Window functions for spectrum analysis with KISS FFT.
 *
 *  Provides Hann, Hamming, Blackman and rectangular windows of any
 *  length, plus apply / unapply helpers for overlap-add processing.
 *  Window tables are cached: repeated requests for the same type and
 *  length return the same table without recomputing it.
 */

#ifndef KISS_WINDOW_H
#define KISS_WINDOW_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KISS_WIN_RECTANGULAR = 0,
    KISS_WIN_HANN,
    KISS_WIN_HAMMING,
    KISS_WIN_BLACKMAN
} kiss_window_type;

/*
 *  kiss_window
 *
 *  Return a read-only table of n window coefficients for the given type.
 *  The table is kept in an internal cache keyed by (type, n), so repeated
 *  calls with the same arguments do not recompute anything. The caller
 *  must not free the returned pointer; use kiss_window_cleanup() instead.
 *  Returns NULL when n <= 0, the type is unknown, or allocation fails.
 */
const kiss_fft_scalar *KISS_FFT_API
kiss_window(kiss_window_type type, int n);

/*
 *  kiss_window_apply
 *
 *  Multiply src[0..n-1] by the cached window table and write the result
 *  into dst. dst may equal src for in-place use.
 */
void KISS_FFT_API
kiss_window_apply(kiss_window_type type, int n,
                  const kiss_fft_scalar *src, kiss_fft_scalar *dst);

/*
 *  kiss_window_unapply
 *
 *  Divide src[0..n-1] by the window table, i.e. undo kiss_window_apply.
 *  Used to normalize overlapping frames before overlap-add. Coefficients
 *  smaller than 1e-6 are treated as zero and produce a zero output, which
 *  avoids dividing by the null endpoints of Hann / Blackman windows.
 *  dst may equal src for in-place use.
 */
void KISS_FFT_API
kiss_window_unapply(kiss_window_type type, int n,
                    const kiss_fft_scalar *src, kiss_fft_scalar *dst);

/* Free every cached window table. */
void KISS_FFT_API kiss_window_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
