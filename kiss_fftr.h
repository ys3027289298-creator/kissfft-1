/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_FTR_H
#define KISS_FTR_H

#include "kiss_fft.h"
#ifdef __cplusplus
extern "C" {
#endif

    
/* 
 
 Real optimized version can save about 45% cpu time vs. complex fft of a real seq.

 
 
 */

typedef struct kiss_fftr_state *kiss_fftr_cfg;


kiss_fftr_cfg KISS_FFT_API kiss_fftr_alloc(int nfft,int inverse_fft,void * mem, size_t * lenmem);
/*
 nfft may be even or odd. An odd nfft is handled internally by appending
 one zero sample and running the even-length transform of nfft + 1.
 Odd lengths are supported for floating-point builds only; fixed-point
 and USE_SIMD builds reject them.

 If you don't care to allocate space, use mem = lenmem = NULL 
*/


void KISS_FFT_API kiss_fftr(kiss_fftr_cfg cfg,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata);
/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points (floor(nfft/2)+1 when nfft is
 odd; the internally padded Nyquist bin is discarded)
*/

void KISS_FFT_API kiss_fftri(kiss_fftr_cfg cfg,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata);
/*
 input freqdata has  nfft/2+1 complex points
 output timedata has nfft scalar points
 For floating-point builds, a kiss_fftr / kiss_fftri round trip returns
 nfft times the input (the true, unpadded length for odd nfft); divide
 timedata by nfft to recover the input.
 Fixed-point builds scale internally and use a different convention.
*/

#define kiss_fftr_free KISS_FFT_FREE

#ifdef __cplusplus
}
#endif
#endif
