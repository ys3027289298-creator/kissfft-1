/*
 * Minimal mono WAV reader/writer.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "wav_io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define WAVE_FORMAT_PCM        0x0001
#define WAVE_FORMAT_IEEE_FLOAT 0x0003
#define WAVE_FORMAT_EXTENSIBLE 0xFFFE

static uint32_t rd_u32(FILE *f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4)
        return 0;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint16_t rd_u16(FILE *f)
{
    uint8_t b[2];
    if (fread(b, 1, 2, f) != 2)
        return 0;
    return (uint16_t)(b[0] | (b[1] << 8));
}

static void wr_u32(FILE *f, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8),
                     (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    fwrite(b, 1, 4, f);
}

static void wr_u16(FILE *f, uint16_t v)
{
    uint8_t b[2] = { (uint8_t)v, (uint8_t)(v >> 8) };
    fwrite(b, 1, 2, f);
}

static int find_chunk(FILE *f, const char *want, uint32_t *size_out)
{
    char id[4];
    while (fread(id, 1, 4, f) == 4) {
        uint32_t size = rd_u32(f);
        if (memcmp(id, want, 4) == 0) {
            *size_out = size;
            return 0;
        }
        /* chunks are word-aligned (padded to an even byte count) */
        long off = (long)size + (long)(size & 1u);
        if (fseek(f, off, SEEK_CUR) != 0)
            return -1;
    }
    return -1;
}

int wav_read(const char *path, wav_file *w)
{
    memset(w, 0, sizeof(*w));
    FILE *f = fopen(path, "rb");
    if (f == NULL)
        return -1;

    char riff[4], wave[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0)
        goto fail;
    (void)rd_u32(f);
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0)
        goto fail;

    uint32_t fmt_size = 0;
    if (find_chunk(f, "fmt ", &fmt_size) != 0)
        goto fail;
    long fmt_end = ftell(f) + (long)fmt_size;

    uint16_t format = rd_u16(f);
    uint16_t channels = rd_u16(f);
    uint32_t sample_rate = rd_u32(f);
    (void)rd_u32(f); /* byte rate */
    (void)rd_u16(f); /* block align */
    uint16_t bits = rd_u16(f);

    if (format == WAVE_FORMAT_EXTENSIBLE && fmt_size >= 40) {
        uint16_t ext_size = rd_u16(f);
        (void)rd_u16(f); /* valid bits per sample */
        (void)rd_u32(f); /* channel mask */
        uint8_t guid[16];
        if (fread(guid, 1, 16, f) != 16)
            goto fail;
        (void)ext_size;
        /* KSDATAFORMAT_SUBTYPE_PCM starts 01 00 00 00 ... ;
           KSDATAFORMAT_SUBTYPE_IEEE_FLOAT starts 03 00 00 00 ... */
        format = (uint16_t)guid[0];
    }

    if (fseek(f, fmt_end + (fmt_end & 1l), SEEK_SET) != 0)
        goto fail;

    uint32_t data_size = 0;
    if (find_chunk(f, "data", &data_size) != 0)
        goto fail;

    int bytes_per_sample = bits / 8;
    if (channels == 0 || bytes_per_sample <= 0)
        goto fail;
    if (format != WAVE_FORMAT_PCM && format != WAVE_FORMAT_IEEE_FLOAT)
        goto fail;
    if (format == WAVE_FORMAT_PCM &&
        !(bits == 16 || bits == 24 || bits == 32))
        goto fail;
    if (format == WAVE_FORMAT_IEEE_FLOAT && bits != 32)
        goto fail;

    long total_samples = (long)(data_size / (uint32_t)bytes_per_sample);
    long frames = total_samples / channels;
    float *data = (float *)malloc((size_t)frames * sizeof(float));
    if (data == NULL)
        goto fail;

    uint8_t *raw = (uint8_t *)malloc(data_size ? data_size : 1);
    if (raw == NULL) {
        free(data);
        goto fail;
    }
    size_t got = fread(raw, 1, data_size, f);
    if (got != data_size) {
        /* tolerate truncated tails by zero-filling the missing bytes */
        memset(raw + got, 0, data_size - got);
    }
    fclose(f);
    f = NULL;

    for (long i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (int c = 0; c < channels; ++c) {
            uint8_t *p = raw + ((size_t)i * channels + c) * bytes_per_sample;
            double v;
            if (format == WAVE_FORMAT_IEEE_FLOAT) {
                float tmp;
                memcpy(&tmp, p, sizeof(float));
                v = tmp;
            } else if (bits == 16) {
                int16_t s = (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
                v = s / 32768.0;
            } else if (bits == 24) {
                int32_t s = (int32_t)p[0] | ((int32_t)p[1] << 8) |
                            ((int32_t)p[2] << 16);
                if (s & 0x800000)
                    s |= ~0xFFFFFF;
                v = s / 8388608.0;
            } else { /* 32-bit PCM */
                int32_t s = (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                      ((uint32_t)p[2] << 16) |
                                      ((uint32_t)p[3] << 24));
                v = s / 2147483648.0;
            }
            acc += v;
        }
        data[i] = (float)(acc / channels);
    }
    free(raw);

    w->sample_rate = (int)sample_rate;
    w->channels = 1;
    w->num_frames = frames;
    w->data = data;
    return 0;

fail:
    if (f != NULL)
        fclose(f);
    return -1;
}

int wav_write_mono_pcm16(const char *path, const float *data,
                         long num_samples, int sample_rate)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL)
        return -1;

    uint32_t data_bytes = (uint32_t)(num_samples * 2);
    fwrite("RIFF", 1, 4, f);
    wr_u32(f, 36u + data_bytes);
    fwrite("WAVE", 1, 4, f);

    fwrite("fmt ", 1, 4, f);
    wr_u32(f, 16);
    wr_u16(f, WAVE_FORMAT_PCM);
    wr_u16(f, 1);
    wr_u32(f, (uint32_t)sample_rate);
    wr_u32(f, (uint32_t)sample_rate * 2);
    wr_u16(f, 2);
    wr_u16(f, 16);

    fwrite("data", 1, 4, f);
    wr_u32(f, data_bytes);
    for (long i = 0; i < num_samples; ++i) {
        double v = data[i];
        if (v > 1.0) v = 1.0;
        if (v < -1.0) v = -1.0;
        int16_t s = (int16_t)lrint(v * 32767.0);
        wr_u16(f, (uint16_t)s);
    }

    int ok = (ferror(f) == 0);
    fclose(f);
    return ok ? 0 : -1;
}

void wav_free(wav_file *w)
{
    if (w == NULL)
        return;
    free(w->data);
    w->data = NULL;
    w->num_frames = 0;
}
