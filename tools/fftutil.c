/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <float.h>

#include "kiss_fft.h"
#include "kiss_fftndr.h"

#define STFT_WIN_RECT 0
#define STFT_WIN_HANN 1

static
void spectrogram_file(FILE * fin,FILE * fout,int win,int hop,int wintype)
{
    kiss_fftr_cfg st;
    kiss_fftr_cfg ist;
    kiss_fft_scalar *sig;
    kiss_fft_scalar *w;
    kiss_fft_scalar *frame;
    kiss_fft_scalar *iframe;
    kiss_fft_scalar *recon;
    kiss_fft_scalar *norm;
    kiss_fft_cpx *spec;
    kiss_fft_scalar *mag;
    long nsig=0, cap=0;
    long nframes, f, i;
    int n, nfreq = win/2+1;
    double maxnorm=0, maxabs=0, maxerr=0, sumsq=0, rms;
    long ncovered=0;
    double eps = (sizeof(kiss_fft_scalar)==sizeof(float)) ? FLT_EPSILON : DBL_EPSILON;
    double bound;
    size_t nr;

    /* read the whole real-valued input (needed again for the error check) */
    cap = 1<<16;
    sig = (kiss_fft_scalar*)malloc(sizeof(kiss_fft_scalar)*cap);
    while (1) {
        if (nsig==(long)cap) {
            cap *= 2;
            sig = (kiss_fft_scalar*)realloc(sig,sizeof(kiss_fft_scalar)*cap);
        }
        nr = fread(sig+nsig, sizeof(kiss_fft_scalar), cap-nsig, fin);
        nsig += (long)nr;
        if (nr==0)
            break;
    }
    if (nsig==0) {
        fprintf(stderr,"spectrogram: no input samples\n");
        exit(1);
    }

    nframes = (nsig<=win) ? 1 : 1 + (nsig - win + hop - 1)/hop;

    w      = (kiss_fft_scalar*)malloc(sizeof(kiss_fft_scalar)*win);
    frame  = (kiss_fft_scalar*)malloc(sizeof(kiss_fft_scalar)*win);
    iframe = (kiss_fft_scalar*)malloc(sizeof(kiss_fft_scalar)*win);
    spec   = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx)*nfreq);
    mag    = (kiss_fft_scalar*)malloc(sizeof(kiss_fft_scalar)*nfreq);
    recon  = (kiss_fft_scalar*)calloc(nsig,sizeof(kiss_fft_scalar));
    norm   = (kiss_fft_scalar*)calloc(nsig,sizeof(kiss_fft_scalar));

    for (n=0;n<win;++n) {
        if (wintype==STFT_WIN_HANN)
            w[n] = (kiss_fft_scalar)( 0.5*(1.0-cos(2.0*3.14159265358979323846*n/win)) );
        else
            w[n] = 1;
    }

    st  = kiss_fftr_alloc(win,0,0,0);
    ist = kiss_fftr_alloc(win,1,0,0);

    for (f=0;f<nframes;++f) {
        long start = f*hop;
        int k;
        for (n=0;n<win;++n) {
            long idx = start+n;
            frame[n] = (idx<nsig) ? sig[idx]*w[n] : 0;
        }
        kiss_fftr(st,frame,spec);
        for (k=0;k<nfreq;++k)
            mag[k] = (kiss_fft_scalar)sqrt( (double)spec[k].r*spec[k].r + (double)spec[k].i*spec[k].i );
        fwrite(mag,sizeof(kiss_fft_scalar),nfreq,fout);

        /* overlap-add reconstruction (kiss_fftri is unscaled, so divide by win) */
        kiss_fftri(ist,spec,iframe);
        for (n=0;n<win;++n) {
            long idx = start+n;
            if (idx<nsig) {
                recon[idx] += w[n]*iframe[n]/win;
                norm[idx]  += w[n]*w[n];
            }
        }
    }

    for (i=0;i<nsig;++i) {
        double a = fabs((double)sig[i]);
        if (a>maxabs) maxabs=a;
        if ((double)norm[i]>maxnorm) maxnorm=norm[i];
    }
    for (i=0;i<nsig;++i) {
        if ((double)norm[i] > 1e-6*maxnorm) {
            double e = fabs((double)recon[i]/norm[i] - (double)sig[i]);
            if (e>maxerr) maxerr=e;
            sumsq += e*e;
            ++ncovered;
        }
    }
    rms = ncovered ? sqrt(sumsq/ncovered) : 0;
    /* conservative round-trip bound: per-frame FFT/IFFT error grows at most
       linearly with win*eps, overlap-add normalization only shrinks it */
    bound = 16.0 * win * eps * (maxabs>0?maxabs:1);

    fprintf(stderr,"spectrogram: nsamples=%ld win=%d hop=%d window=%s nframes=%ld nfreq=%d\n",
            nsig,win,hop,(wintype==STFT_WIN_HANN?"hann":"rect"),nframes,nfreq);
    fprintf(stderr,"reconstruction: covered=%ld/%ld maxerr=%.6g rms=%.6g bound=%.6g (%s)\n",
            ncovered,nsig,maxerr,rms,bound,(maxerr<=bound?"OK":"FAIL"));

    free(sig);free(w);free(frame);free(iframe);free(spec);free(mag);free(recon);free(norm);
    free(st);
    free(ist);
}

static
void fft_file(FILE * fin,FILE * fout,int nfft,int isinverse)
{
    kiss_fft_cfg st;
    kiss_fft_cpx * buf;
    kiss_fft_cpx * bufout;

    buf = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft );
    bufout = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft );
    st = kiss_fft_alloc( nfft ,isinverse ,0,0);

    while (1) {
        size_t nread = fread(buf, sizeof(kiss_fft_cpx), nfft, fin);
        if (nread == (size_t)nfft) {
            kiss_fft( st , buf ,bufout);
            fwrite( bufout , sizeof(kiss_fft_cpx) , nfft , fout );
            continue;
        }
        if (nread != 0)
            fprintf(stderr,"short read on complex input\n");
        break;
    }
    free(st);
    free(buf);
    free(bufout);
}

static
void fft_filend(FILE * fin,FILE * fout,int *dims,int ndims,int isinverse)
{
    kiss_fftnd_cfg st;
    kiss_fft_cpx *buf;
    int dimprod=1,i;
    for (i=0;i<ndims;++i) 
        dimprod *= dims[i];

    buf = (kiss_fft_cpx *) malloc (sizeof (kiss_fft_cpx) * dimprod);
    st = kiss_fftnd_alloc (dims, ndims, isinverse, 0, 0);

    while (1) {
        size_t nread = fread(buf, sizeof(kiss_fft_cpx), dimprod, fin);
        if (nread == (size_t)dimprod) {
            kiss_fftnd(st, buf, buf);
            fwrite(buf, sizeof(kiss_fft_cpx), dimprod, fout);
            continue;
        }
        if (nread != 0)
            fprintf(stderr,"short read on nd complex input\n");
        break;
    }
    free (st);
    free (buf);
}



static
void fft_filend_real(FILE * fin,FILE * fout,int *dims,int ndims,int isinverse)
{
    int dimprod=1,i;
    kiss_fftndr_cfg st;
    void *ibuf;
    void *obuf;
    int insize,outsize; // size in bytes

    for (i=0;i<ndims;++i) 
        dimprod *= dims[i];
    insize = outsize = dimprod;
    int rdim = dims[ndims-1];

    if (isinverse)
        insize = insize*2*(rdim/2+1)/rdim;
    else
        outsize = outsize*2*(rdim/2+1)/rdim;

    ibuf = malloc(insize*sizeof(kiss_fft_scalar));
    obuf = malloc(outsize*sizeof(kiss_fft_scalar));

    st = kiss_fftndr_alloc(dims, ndims, isinverse, 0, 0);

    while (1) {
        size_t nread = fread(ibuf, sizeof(kiss_fft_scalar), insize, fin);
        if (nread == (size_t)insize) {
            if (isinverse) {
                kiss_fftndri(st,
                        (kiss_fft_cpx*)ibuf,
                        (kiss_fft_scalar*)obuf);
            } else {
                kiss_fftndr(st,
                        (kiss_fft_scalar*)ibuf,
                        (kiss_fft_cpx*)obuf);
            }
            fwrite(obuf, sizeof(kiss_fft_scalar), outsize, fout);
            continue;
        }
        if (nread != 0)
            fprintf(stderr,"short read on nd real input\n");
        break;
    }
    free(st);
    free(ibuf);
    free(obuf);
}

static
void fft_file_real(FILE * fin,FILE * fout,int nfft,int isinverse)
{
    kiss_fftr_cfg st;
    kiss_fft_scalar * rbuf;
    kiss_fft_cpx * cbuf;

    rbuf = (kiss_fft_scalar*)malloc(sizeof(kiss_fft_scalar) * nfft );
    cbuf = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * (nfft/2+1) );
    st = kiss_fftr_alloc( nfft ,isinverse ,0,0);

    if (isinverse==0) {
        while (1) {
            size_t nread = fread(rbuf, sizeof(kiss_fft_scalar), nfft, fin);
            if (nread == nfft) {
                kiss_fftr( st , rbuf ,cbuf);
                fwrite( cbuf , sizeof(kiss_fft_cpx) , (nfft/2 + 1) , fout );
                continue;
            }
            if (nread != 0)
                fprintf(stderr,"short read on real input\n");
            break;
        }
    }else{
        while (1) {
            size_t nread = fread(cbuf, sizeof(kiss_fft_cpx), (nfft/2+1), fin);
            if (nread == (size_t)(nfft/2+1)) {
                kiss_fftri( st , cbuf ,rbuf);
                fwrite( rbuf , sizeof(kiss_fft_scalar) , nfft , fout );
                continue;
            }
            if (nread != 0)
                fprintf(stderr,"short read on complex input\n");
            break;
        }
    }
    free(st);
    free(rbuf);
    free(cbuf);
}

static
int get_dims(char * arg,int * dims)
{
    char *p0;
    int ndims=0;

    do{
        p0 = strchr(arg,',');
        if (p0)
            *p0++ = '\0';
        dims[ndims++] = atoi(arg);
//         fprintf(stderr,"dims[%d] = %d\n",ndims-1,dims[ndims-1]); 
        arg = p0;
    }while (p0);
    return ndims;
}

int main(int argc,char ** argv)
{
    int isinverse=0;
    int isreal=0;
    int stft_win=0;
    int stft_hop=0;
    int stft_wintype=STFT_WIN_HANN;
    const char * stft_winname="hann";
    FILE *fin=stdin;
    FILE *fout=stdout;
    int ndims=1;
    int dims[32];
    dims[0] = 1024; /*default fft size*/

    while (1) {
        int c=getopt(argc,argv,"n:iRS:W:");
        if (c==-1) break;
        switch (c) {
            case 'n':
                ndims = get_dims(optarg,dims);
                break;
            case 'i':isinverse=1;break;
            case 'R':isreal=1;break;
            case 'S':
                     if (sscanf(optarg,"%d,%d",&stft_win,&stft_hop)!=2) {
                         fprintf(stderr,"error: -S expects win,hop (e.g. -S 1024,256)\n");
                         exit(1);
                     }
                     break;
            case 'W':stft_winname=optarg;break;
            case '?':
                     fprintf(stderr,"usage options:\n"
                            "\t-n d1[,d2,d3...]: fft dimension(s)\n"
                            "\t-i : inverse\n"
                            "\t-R : real input samples, not complex\n"
                            "\t-S win,hop : spectrogram mode, real input, per-frame magnitude output\n"
                            "\t-W rect|hann : window function for -S (default hann)\n");
                     exit (1);
            default:fprintf(stderr,"bad %c\n",c);break;
        }
    }

    if (stft_win) {
        if (strcmp(stft_winname,"hann")==0)
            stft_wintype = STFT_WIN_HANN;
        else if (strcmp(stft_winname,"rect")==0)
            stft_wintype = STFT_WIN_RECT;
        else {
            fprintf(stderr,"error: unknown window '%s' (expected rect or hann)\n",stft_winname);
            exit(1);
        }
        if (isinverse) {
            fprintf(stderr,"error: -S (spectrogram) cannot be combined with -i\n");
            exit(1);
        }
        if (ndims!=1) {
            fprintf(stderr,"error: -S (spectrogram) is 1-D only, cannot be combined with -n d1,d2,...\n");
            exit(1);
        }
        if (stft_win<2 || (stft_win&1)) {
            fprintf(stderr,"error: window length must be even and >= 2 (got %d)\n",stft_win);
            exit(1);
        }
        if (stft_hop<1) {
            fprintf(stderr,"error: hop must be >= 1 (got %d)\n",stft_hop);
            exit(1);
        }
        if (stft_hop>stft_win) {
            fprintf(stderr,"error: hop (%d) must not exceed window length (%d)\n",stft_hop,stft_win);
            exit(1);
        }
    }

    if ( optind < argc ) {
        if (strcmp("-",argv[optind]) !=0)
            fin = fopen(argv[optind],"rb");
        ++optind;
    }

    if ( optind < argc ) {
        if ( strcmp("-",argv[optind]) !=0 ) 
            fout = fopen(argv[optind],"wb");
        ++optind;
    }

    if (stft_win) {
        spectrogram_file(fin,fout,stft_win,stft_hop,stft_wintype);
    }else if (ndims==1) {
        if (isreal)
            fft_file_real(fin,fout,dims[0],isinverse);
        else
            fft_file(fin,fout,dims[0],isinverse);
    }else{
        if (isreal)
            fft_filend_real(fin,fout,dims,ndims,isinverse);
        else
            fft_filend(fin,fout,dims,ndims,isinverse);
    }

    if (fout!=stdout) fclose(fout);
    if (fin!=stdin) fclose(fin);

    return 0;
}
