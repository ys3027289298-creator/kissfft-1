/*
 * Command line interface for the kissfft phase vocoder.
 *
 * Usage:
 *   pv_stretch -a <factor> [-n <fft_size>] [-ha <analysis_hop>] in.wav out.wav
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "phase_vocoder.h"
#include "wav_io.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s -a <stretch_factor> [-n <fft_size>] "
            "[-ha <analysis_hop>] in.wav out.wav\n", prog);
}

int main(int argc, char **argv)
{
    double alpha = -1.0;
    int nfft = 2048;
    int hop_ana = -1;
    const char *in_path = NULL;
    const char *out_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            nfft = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ha") == 0 && i + 1 < argc) {
            hop_ana = atoi(argv[++i]);
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else if (in_path == NULL) {
            in_path = argv[i];
        } else if (out_path == NULL) {
            out_path = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (alpha <= 0.0 || in_path == NULL || out_path == NULL) {
        usage(argv[0]);
        return 2;
    }
    if (hop_ana <= 0)
        hop_ana = nfft / 8;

    wav_file w;
    if (wav_read(in_path, &w) != 0) {
        fprintf(stderr, "error: cannot read WAV '%s'\n", in_path);
        return 1;
    }

    pv_state *pv = pv_alloc(nfft, hop_ana, alpha);
    if (pv == NULL) {
        fprintf(stderr, "error: pv_alloc failed (nfft=%d hop_ana=%d alpha=%g)\n",
                nfft, hop_ana, alpha);
        wav_free(&w);
        return 1;
    }

    long cap = pv_expected_output_len(pv, w.num_frames) + (long)nfft;
    float *out = (float *)malloc((size_t)cap * sizeof(float));
    if (out == NULL) {
        fprintf(stderr, "error: out of memory\n");
        pv_free(pv);
        wav_free(&w);
        return 1;
    }

    /* Feed the stream in several chunks to exercise the streaming path. */
    long produced = 0;
    enum { CHUNK = 4096 };
    for (long off = 0; off < w.num_frames; off += CHUNK) {
        size_t n = (size_t)(w.num_frames - off);
        if (n > CHUNK)
            n = CHUNK;
        long got = pv_process(pv, w.data + off, n, out + produced);
        if (got < 0) {
            fprintf(stderr, "error: pv_process failed\n");
            free(out);
            pv_free(pv);
            wav_free(&w);
            return 1;
        }
        produced += got;
    }
    produced += pv_flush(pv, out + produced);

    long expect = pv_expected_output_len(pv, w.num_frames);
    if (produced != expect)
        fprintf(stderr,
                "warning: produced %ld samples, expected %ld\n",
                produced, expect);

    int rc = wav_write_mono_pcm16(out_path, out, produced, w.sample_rate);
    if (rc != 0)
        fprintf(stderr, "error: cannot write WAV '%s'\n", out_path);
    else
        printf("%s: %ld -> %ld samples (alpha=%.4f, ratio=%.4f)\n",
               out_path, w.num_frames, produced, alpha,
               w.num_frames ? (double)produced / w.num_frames : 0.0);

    free(out);
    pv_free(pv);
    wav_free(&w);
    return rc == 0 ? 0 : 1;
}
