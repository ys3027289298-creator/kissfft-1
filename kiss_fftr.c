/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"

struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
    kiss_fft_cpx * work;
    kiss_fft_cpx * outbuf;
    int nfft;
    int odd;
#ifdef USE_SIMD
    void * pad;
#endif
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem)
{
	KISS_FFT_ALIGN_CHECK(mem)

    int i;
    kiss_fftr_cfg st = NULL;
    size_t subsize = 0, memneeded;
    int odd, ncfft;

    if (nfft <= 0) {
        KISS_FFT_ERROR("Real FFT size must be positive.");
        return NULL;
    }

    /* Odd lengths are supported by padding one zero internally so that the
     * underlying transform runs on the even length nfft + 1. */
    odd = nfft & 1;
    ncfft = (nfft + odd) >> 1;

    kiss_fft_alloc (ncfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state) + subsize
        + sizeof(kiss_fft_cpx) * (ncfft + ncfft / 2);
    if (odd) {
        /* work: padded 2*ncfft real samples (forward) or ncfft+1 bins
         * (inverse); outbuf: even-length real FFT result (ncfft+1 bins). */
        memneeded += sizeof(kiss_fft_cpx) * (2 * ncfft + 2);
        memneeded += sizeof(kiss_fft_cpx) * (2 * ncfft);
    }

#ifdef USE_SIMD
    if (odd) {
        KISS_FFT_ERROR("Odd length Real FFT is not supported for USE_SIMD.");
        return NULL;
    }
#endif
#ifdef FIXED_POINT
    if (odd) {
        KISS_FFT_ERROR("Odd length Real FFT requires a floating-point build.");
        return NULL;
    }
#endif

    if (lenmem == NULL) {
        st = (kiss_fftr_cfg) KISS_FFT_MALLOC (memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_fftr_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->substate = (kiss_fft_cfg) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (kiss_fft_cpx *) (((char *) st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + ncfft;
    st->work = odd ? (st->super_twiddles + (ncfft / 2)) : NULL;
    st->outbuf = odd ? (st->work + (2 * ncfft + 2)) : NULL;
    st->nfft = nfft;
    st->odd = odd;
    kiss_fft_alloc(ncfft, inverse_fft, st->substate, &subsize);

    for (i = 0; i < ncfft/2; ++i) {
        double phase =
            -3.14159265358979323846264338327 * ((double) (i+1) / ncfft + .5);
        if (inverse_fft)
            phase *= -1;
        kf_cexp (st->super_twiddles+i,phase);
    }
    return st;
}

static void kiss_fftr_even(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k,ncfft;
    kiss_fft_cpx fpnk,fpk,f1k,f2k,tw,tdc;

    if ( st->substate->inverse) {
        KISS_FFT_ERROR("kiss fft usage error: improper alloc");
        return;/* The caller did not call the correct function */
    }

    ncfft = st->substate->nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;
    C_FIXDIV(tdc,2);
    CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
    CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD
#ifdef HAVE_LASX
    freqdata[0].i = (__m256)(__lasx_xvreplgr2vr_w(0));
    freqdata[ncfft].i = freqdata[0].i;
#elif defined(HAVE_LSX)
    freqdata[0].i = (__m128)(__lsx_vreplgr2vr_w(0));
    freqdata[ncfft].i = freqdata[0].i;
#else
    freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#endif
#else
    freqdata[ncfft].i = freqdata[0].i = 0;
#endif

    for ( k=1;k <= ncfft/2 ; ++k ) {
        fpk    = st->tmpbuf[k];
        fpnk.r =   st->tmpbuf[ncfft-k].r;
        fpnk.i = - st->tmpbuf[ncfft-k].i;
        C_FIXDIV(fpk,2);
        C_FIXDIV(fpnk,2);

        C_ADD( f1k, fpk , fpnk );
        C_SUB( f2k, fpk , fpnk );
        C_MUL( tw , f2k , st->super_twiddles[k-1]);

        freqdata[k].r = HALF_OF(f1k.r + tw.r);
        freqdata[k].i = HALF_OF(f1k.i + tw.i);
        freqdata[ncfft-k].r = HALF_OF(f1k.r - tw.r);
        freqdata[ncfft-k].i = HALF_OF(tw.i - f1k.i);
    }
}

static void kiss_fftri_even(kiss_fftr_cfg st,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata)
{
    /* input buffer timedata is stored row-wise */
    int k, ncfft;

    if (st->substate->inverse == 0) {
        KISS_FFT_ERROR("kiss fft usage error: improper alloc");
        return;/* The caller did not call the correct function */
    }

    ncfft = st->substate->nfft;

    st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
    C_FIXDIV(st->tmpbuf[0],2);

    for (k = 1; k <= ncfft / 2; ++k) {
        kiss_fft_cpx fk, fnkc, fek, fok, tmp;
        fk = freqdata[k];
        fnkc.r = freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;
        C_FIXDIV( fk , 2 );
        C_FIXDIV( fnkc , 2 );

        C_ADD (fek, fk, fnkc);
        C_SUB (tmp, fk, fnkc);
        C_MUL (fok, tmp, st->super_twiddles[k-1]);
        C_ADD (st->tmpbuf[k],     fek, fok);
        C_SUB (st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD
#ifdef HAVE_LASX
        __m256 neg_one = (__m256)__lasx_xvreplgr2vr_w(0xBF800000); // -1.0f
        st->tmpbuf[ncfft - k].i = __lasx_xvfmul_s(st->tmpbuf[ncfft - k].i, neg_one);
#elif defined(HAVE_LSX)
        __m128 neg_one = (__m128)__lsx_vreplgr2vr_w(0xBF800000); // -1.0f
        st->tmpbuf[ncfft - k].i = __lsx_vfmul_s(st->tmpbuf[ncfft - k].i, neg_one);
#else
        st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#endif
#else
        st->tmpbuf[ncfft - k].i *= -1;
#endif
    }
    kiss_fft (st->substate, st->tmpbuf, (kiss_fft_cpx *) timedata);
}

void kiss_fftr(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    int k;

    if (st->substate->inverse) {
        KISS_FFT_ERROR("kiss fft usage error: improper alloc");
        return;/* The caller did not call the correct function */
    }

    if (!st->odd) {
        kiss_fftr_even(st, timedata, freqdata);
        return;
    }

    /* Copy the nfft real samples (contiguous scalars, as the even real FFT
     * reinterprets them as an interleaved complex array) and append one zero.
     * The padded Nyquist bin is discarded, leaving (nfft + 1)/2 output bins. */
    {
        kiss_fft_scalar *padded = (kiss_fft_scalar *) st->work;
        for (k = 0; k < st->nfft; ++k)
            padded[k] = timedata[k];
        padded[st->nfft] = 0;
    }

    /* even transform writes into the dedicated outbuf so its tmpbuf scratch
     * can never touch the padded input held in work */
    kiss_fftr_even(st, (const kiss_fft_scalar *) st->work, st->outbuf);
    for (k = 0; k <= st->substate->nfft - 1; ++k)
        freqdata[k] = st->outbuf[k];
}

void kiss_fftri(kiss_fftr_cfg st,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata)
{
    int k, ncfft;
    const kiss_fft_scalar *evenout;

    if (st->substate->inverse == 0) {
        KISS_FFT_ERROR("kiss fft usage error: improper alloc");
        return;/* The caller did not call the correct function */
    }

    if (!st->odd) {
        kiss_fftri_even(st, freqdata, timedata);
        return;
    }

    ncfft = st->substate->nfft; /* (nfft + 1) / 2 */

    for (k = 0; k < ncfft; ++k)
        st->work[k] = freqdata[k];

    /* Reconstruct the padded Nyquist bin. The padded length is 2*ncfft and
     * its last sample is zero, so the IDFT there vanishes:
     *   0 = sum_k X[k] exp(-i pi k/ncfft), giving
     *   X[ncfft] = sum_{k != ncfft} X[k] exp(-i pi k/ncfft).
     * Pairing k with 2*ncfft-k via real-signal conjugate symmetry yields the
     * purely real form:
     *   X[ncfft] = Re X[0] + 2*sum_{j=1..ncfft-1} Re(X[j] exp(-i pi j/ncfft)). */
    st->work[ncfft].r = st->work[0].r;
    st->work[ncfft].i = 0;
    for (k = 0; k < ncfft; ++k) {
        kiss_fft_cpx phase, rotated;
        double angle = -3.14159265358979323846264338327 * (double) k / ncfft;
        kf_cexp(&phase, angle);
        C_MUL(rotated, st->work[k], phase);
        if (k != 0)
            st->work[ncfft].r += (kiss_fft_scalar) (2.0 * rotated.r);
    }

    kiss_fftri_even(st, st->work, (kiss_fft_scalar *) st->outbuf);
    evenout = (const kiss_fft_scalar *) st->outbuf;

    /* Even inverse scales by nfft + 1; renormalize by the true (odd) length so
     * that a forward/inverse round trip returns nfft times the input. */
    for (k = 0; k < st->nfft; ++k)
        timedata[k] = evenout[k] *
            ((kiss_fft_scalar) st->nfft / (kiss_fft_scalar) (st->nfft + 1));
}
