"""Minimal mono 16-bit PCM WAV reader/writer using numpy only.

Kept dependency-free so the evaluation runs on a bare numpy install.
"""

import numpy as np

_WAVE_FORMAT_PCM = 0x0001


def read_wav(path):
    """Read a mono WAV (PCM16/24/32 or float32); downmixes multi-channel.

    Returns (samples_float64, sample_rate).
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file: %s" % path)

    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = pos + 8
        if chunk_id == b"fmt ":
            fmt = raw[body:body + size]
        elif chunk_id == b"data":
            data = raw[body:body + size]
        pos = body + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("missing fmt/data chunk: %s" % path)

    format_tag = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    sample_rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")

    if format_tag == 0xFFFE and len(fmt) >= 40:  # WAVE_FORMAT_EXTENSIBLE
        format_tag = fmt[24]  # first byte of the subtype GUID

    if format_tag == _WAVE_FORMAT_PCM:
        if bits == 16:
            x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
        elif bits == 24:
            b = np.frombuffer(data, dtype=np.uint8)
            n = len(b) // 3
            b = b[:n * 3].reshape(n, 3)
            i32 = (b[:, 0].astype(np.int32)
                   | (b[:, 1].astype(np.int32) << 8)
                   | (b[:, 2].astype(np.int32) << 16))
            i32 = np.where(i32 & 0x800000, i32 - 0x1000000, i32)
            x = i32 / 8388608.0
        elif bits == 32:
            x = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError("unsupported PCM bit depth: %d" % bits)
    elif format_tag == 0x0003 and bits == 32:  # IEEE float
        x = np.frombuffer(data, dtype="<f4").astype(np.float64)
    else:
        raise ValueError("unsupported format: tag=%d bits=%d" % (format_tag, bits))

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, sample_rate


def write_wav(path, x, sample_rate):
    """Write mono 16-bit PCM WAV from a float array."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    pcm = np.rint(x * 32767.0).astype("<i2")
    data_bytes = pcm.tobytes()
    n = len(pcm)

    hdr = b"RIFF"
    hdr += (36 + len(data_bytes)).to_bytes(4, "little")
    hdr += b"WAVE"
    hdr += b"fmt "
    hdr += (16).to_bytes(4, "little")
    hdr += _WAVE_FORMAT_PCM.to_bytes(2, "little")
    hdr += (1).to_bytes(2, "little")                    # mono
    hdr += int(sample_rate).to_bytes(4, "little")
    hdr += (sample_rate * 2).to_bytes(4, "little")      # byte rate
    hdr += (2).to_bytes(2, "little")                    # block align
    hdr += (16).to_bytes(2, "little")                   # bits
    hdr += b"data"
    hdr += len(data_bytes).to_bytes(4, "little")

    with open(path, "wb") as f:
        f.write(hdr)
        f.write(data_bytes)
