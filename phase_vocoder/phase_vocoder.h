/*
 * Phase vocoder for time-scale modification (time stretching without
 * pitch shifting), built on kissfft.
 *
 * Algorithm: Laroche & Dolson (1999) phase vocoder.
 *   - short-time Fourier analysis with a Hann window
 *   - phase unwrapping and instantaneous-frequency reassignment per bin
 *   - phase-locked synthesis driven by the time-stretch factor alpha
 *   - weighted overlap-add (75% synthesis overlap) with COLA normalization
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef PHASE_VOCODER_H
#define PHASE_VOCODER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pv_state pv_state;

/*
 * Create a phase vocoder state.
 *
 *   nfft        FFT size (even). 2048 is a good default for 44.1/48 kHz.
 *   hop_ana     analysis hop in samples. Use nfft/8 so that the synthesis
 *               hop alpha*hop_ana stays at or below nfft/4 (>= 75% overlap)
 *               for every alpha in [0.5, 2]; must be <= nfft/2.
 *   alpha       time-stretch factor: output_duration = alpha * input_duration.
 *               alpha > 1 slows down, alpha < 1 speeds up; alpha == 1 gives
 *               near-perfect reconstruction. The synthesis hop is derived as
 *               hop_syn = round(alpha * hop_ana) (Laroche-Dolson convention).
 *
 * Returns NULL on invalid parameters or allocation failure.
 * Free with pv_free().
 */
pv_state *pv_alloc(int nfft, int hop_ana, double alpha);

void pv_free(pv_state *pv);

/*
 * Process input samples and write stretched samples to out.
 *
 *   in          input samples (interleaved frames; mono only)
 *   nin         number of input samples available
 *   out         output buffer; must hold at most
 *               (size_t)(nin / alpha) + 2 * nfft samples
 *
 * Returns the number of output samples written, or -1 on error.
 * Call repeatedly with successive input chunks, then pv_flush().
 */
long pv_process(pv_state *pv, const float *in, size_t nin, float *out);

/*
 * Drain the internal overlap-add buffer. Writes at most hop_syn samples
 * (one final frame tail) and returns the number written.
 */
long pv_flush(pv_state *pv, float *out);

/* Exact output length for a given input length, including the flush tail. */
long pv_expected_output_len(const pv_state *pv, long input_len);

#ifdef __cplusplus
}
#endif

#endif /* PHASE_VOCODER_H */
