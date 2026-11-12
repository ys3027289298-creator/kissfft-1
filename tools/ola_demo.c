/*
 * Streaming overlap-add demo for kiss_ola.h (real-time style, chunked I/O).
 * Usage: ola_demo <1|2|3|4>   (runs one test case per invocation)
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "kiss_ola.h"

#define MAXCHUNK 512

static float sig(long i) /* smooth test signal: no true discontinuities */
{
    return sinf(2.f * (float)M_PI * i / 50.f)
         + 0.5f * sinf(2.f * (float)M_PI * i / 13.f);
}

static void nap(void) { usleep(250 * 1000); } /* mimic real-time pacing */

/* stream total samples of sig() through the OLA engine in push/pop chunks */
static void run_stream(const char *name, int nfft, float overlap,
                       int total, int push_chunk, int pop_max)
{
    kiss_ola st = kiss_ola_alloc(nfft, overlap);
    float in[MAXCHUNK], out[MAXCHUNK];
    long off = 0, pos = 0; /* off: samples pushed, pos: samples compared */
    int seg = 0, n, i;
    printf("== %s (nfft=%d overlap=%.0f%% hop=%d) ==\n",
           name, nfft, overlap * 100.f, st->hop);
    if (!st) { printf("alloc failed\n"); return; }
    while (pos < total) {
        if (off < total) { /* push one chunk of signal, then flush zeros */
            n = push_chunk < total - off ? push_chunk : (int)(total - off);
            for (i = 0; i < n; i++) in[i] = sig(off + i);
            off += kiss_ola_push(st, in, n);
        } else if (off < total + nfft) {
            for (i = 0; i < push_chunk; i++) in[i] = 0.f;
            off += kiss_ola_push(st, in, push_chunk);
        }
        n = kiss_ola_ready(st) > 0 ? kiss_ola_pop(st, out, pop_max) : 0;
        if (n > 0) {
            float err = 0.f;
            for (i = 0; i < n && pos + i < total; i++) {
                float d = fabsf(out[i] - sig(pos + i));
                if (d > err) err = d;
            }
            printf("seg %2d | out_len=%4d | max_err=%.3e\n", seg++, n, err);
            nap();
        }
        pos += n;
    }
    kiss_ola_free(st);
}

static void run_passthrough(const char *path) /* case 4: single block */
{
    int nfft = 128, i, n;
    kiss_ola st = kiss_ola_alloc(nfft, 0.5f);
    float ref[MAXCHUNK] = {0}, zeros[MAXCHUNK] = {0}, out[MAXCHUNK], err = 0.f;
    FILE *f = fopen(path, "r");
    if (!st || !f) { printf("alloc/open failed: %s\n", path); return; }
    for (i = 0; i < nfft && fscanf(f, "%f", &ref[i]) == 1; i++) {}
    fclose(f);
    printf("== case4 single-block passthrough (nfft=%d, file=%s) ==\n", nfft, path);
    kiss_ola_push(st, ref, nfft);   /* one whole block in a single push */
    kiss_ola_push(st, zeros, nfft); /* zero tail to finalize the block  */
    n = kiss_ola_pop(st, out, nfft);
    for (i = 0; i < n; i++) {
        float d = fabsf(out[i] - ref[i]);
        if (d > err) err = d;
    }
    printf("seg  0 | out_len=%4d | max_err=%.3e\n", n, err);
    kiss_ola_free(st);
}

int main(int argc, char **argv)
{
    int which = argc > 1 ? atoi(argv[1]) : 0;
    switch (which) {
    case 1: run_stream("case1 50% overlap", 256, 0.50f, 2048, 256, 256); break;
    case 2: run_stream("case2 75% overlap", 256, 0.75f, 2048, 192, 200); break;
    case 3: run_stream("case3 continuous push", 128, 0.50f, 1500, 37, 100); break;
    case 4: run_passthrough("tools/ola_signal.txt"); break;
    default: fprintf(stderr, "usage: %s <1|2|3|4>\n", argv[0]); return 1;
    }
    return 0;
}
