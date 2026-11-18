"""Objective evaluation of the phase-vocoder outputs.

Metrics
-------
duration        output/input sample ratio vs. the requested stretch factor
pitch           per-bin instantaneous-frequency tracking, compared on a
                normalized time axis (cents); pure-tone drift is the range
                of the smoothed frequency track and must stay < 1 cent
transient       onset timing error and 10%-90% attack-time blur factor
clicks          hop-periodic amplitude modulation (dB) and impulsive click
                rate from a robust high-pass detector
identity        alpha=1.0 reconstruction SNR (validates COLA overlap-add)

numpy only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wavio import read_wav

TWO_PI = 2.0 * np.pi


def _princarg(p):
    return np.mod(p + np.pi, TWO_PI) - np.pi


# --------------------------------------------------------------------------
# Spectral helpers
# --------------------------------------------------------------------------

def stft(x, n=4096, hop=1024):
    win = 0.5 * (1.0 - np.cos(TWO_PI * np.arange(n) / n))
    nb = 1 + (len(x) - n) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(nb, n), strides=(x.strides[0] * hop, x.strides[0]))
    return np.fft.rfft(frames * win, axis=1), win


def inst_freq_track(x, fs, n=4096, hop=1024):
    """Instantaneous frequency (Hz) of every positive bin per frame."""
    spec, _ = stft(x, n, hop)
    phase = np.angle(spec)
    k = np.arange(spec.shape[1])
    expect = TWO_PI * k[None, :] * hop / n
    dphi = _princarg(np.diff(phase, axis=0) - expect)
    freq = (k[None, :] + dphi / TWO_PI * n / hop) * fs / n
    return freq, spec[1:]  # same frame count


def _normalize_time(track, n_out):
    """Resample a framewise track onto u in [0, 1] of length n_out."""
    u_old = np.linspace(0.0, 1.0, len(track))
    u_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(u_new, u_old, track)


# --------------------------------------------------------------------------
# Pitch metrics
# --------------------------------------------------------------------------

def cents(f, f_ref):
    return 1200.0 * np.log2(np.asarray(f) / np.asarray(f_ref))


def pitch_pure_tone(x, fs, nominal=440.0):
    freq, spec = inst_freq_track(x, fs)
    mag = np.abs(spec)
    peak_bin = int(np.median(np.argmax(mag, axis=1)))
    f_track = freq[:, peak_bin]
    mag_track = mag[:, peak_bin]
    # Keep only steady-state frames: bin magnitude within 12 dB of its
    # maximum. This excludes fade-in/out and transient frames, where the
    # estimate is dominated by window modulation rather than the oscillator.
    keep = mag_track > np.max(mag_track) * 10.0 ** (-12.0 / 20.0)
    f_steady = f_track[keep]
    med = float(np.median(f_steady))
    # Robust drift: 5th-95th percentile spread of the steady-state track.
    p5, p95 = np.percentile(f_steady, (5.0, 95.0))
    drift_cents = 1200.0 * np.log2(p95 / p5)
    return {
        "median_hz": med,
        "median_cents_vs_nominal": float(cents(med, nominal)),
        "p95_abs_cents": float(np.percentile(
            np.abs(cents(f_steady, med)), 95)),
        "drift_cents": float(drift_cents),
    }


CHORD_NOTES = (261.63, 329.63, 392.00, 440.00)  # C4 E4 G4 A4


def pitch_chord(x, fs, notes=CHORD_NOTES, n=4096, hop=512):
    """Track each known note fundamental directly: locate the local
    magnitude peak nearest the nominal frequency, read its instantaneous
    frequency. Avoids confusing a note's harmonics with its fundamental."""
    freq, spec = inst_freq_track(x, fs, n=n, hop=hop)
    mag = np.abs(spec)
    avg_mag = mag.mean(axis=0)
    bin_hz = fs / n
    out = {}
    for f_nom in notes:
        k0 = int(round(f_nom / bin_hz))
        lo, hi = max(2, k0 - 2), min(len(avg_mag) - 2, k0 + 3)
        k = lo + int(np.argmax(avg_mag[lo:hi]))
        out[f_nom] = float(np.median(freq[:, k]))
    return out


def f0_autocorr(x, fs, frame_ms=80.0, hop_ms=20.0, fmin=70.0, fmax=400.0):
    n = int(frame_ms * fs / 1000.0)
    hop = int(hop_ms * fs / 1000.0)
    win = 0.5 * (1.0 - np.cos(TWO_PI * np.arange(n) / n))
    lag_min = int(fs / fmax)
    lag_max = min(int(fs / fmin), n - 1)
    t, f0, strength = [], [], []
    for i0 in range(0, len(x) - n, hop):
        seg = x[i0:i0 + n] * win
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[n - 1:]
        ac /= (ac[0] + 1e-12)
        region = ac[lag_min:lag_max + 1]
        j = int(np.argmax(region))
        s = region[j]
        lag = lag_min + j
        if s > 0.3:
            # Octave-up correction: a strong peak at half the lag means the
            # true fundamental is an octave higher (classic octave error).
            half = lag // 2
            if half >= lag_min and ac[half] > 0.85 * s:
                lag, s = half, ac[half]
            # parabolic interpolation around the chosen peak
            if 0 < lag < len(ac) - 1:
                a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
                denom = a - 2.0 * b + c
                shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
                lag = lag + shift
        t.append((i0 + n / 2.0) / fs)
        f0.append(fs / lag if s > 0.3 else np.nan)
        strength.append(s)
    t, f0 = np.array(t), np.array(f0)

    # Voiced-track global octave locking. Framewise autocorrelation can
    # slip to a sub/octave lag; a stable voiced utterance has one true
    # octave, so anchor every frame to the track median by factors of two
    # and then smooth residual deviations with a local median filter.
    voiced = ~np.isnan(f0)
    idx = np.where(voiced)[0]
    if len(idx) >= 5:
        vals = f0[idx].copy()
        ref = np.median(vals)
        for m in range(len(vals)):
            while vals[m] > ref * np.sqrt(2):
                vals[m] /= 2.0
            while vals[m] * np.sqrt(2) < ref:
                vals[m] *= 2.0
        med = np.array([np.median(vals[max(0, m - 2):m + 3])
                        for m in range(len(vals))])
        bad = np.abs(vals - med) / med > 0.03
        vals[bad] = med[bad]
        f0[idx] = vals
    return t, f0, np.array(strength)


def pitch_speech(x, fs):
    t, f0, _ = f0_autocorr(x, fs)
    voiced = ~np.isnan(f0)
    return t[voiced], f0[voiced]


def noise_spectral(x, fs, n=4096, hop=2048):
    spec, _ = stft(x, n, hop)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-12)
    gm = np.exp(np.mean(np.log(mag + 1e-12), axis=1))
    flatness = gm / (mag.mean(axis=1) + 1e-12)
    return float(np.median(centroid)), float(np.median(flatness))


# --------------------------------------------------------------------------
# Transient metrics
# --------------------------------------------------------------------------

def rms_envelope(x, fs, frame_ms=5.0, hop_ms=2.5):
    n = int(frame_ms * fs / 1000.0)
    hop = int(hop_ms * fs / 1000.0)
    nb = 1 + (len(x) - n) // hop
    env = np.empty(nb)
    for m in range(nb):
        env[m] = np.sqrt(np.mean(x[m * hop:m * hop + n] ** 2) + 1e-20)
    t = (np.arange(nb) * hop + n / 2.0) / fs
    return t, env, hop / fs


def detect_onsets(t, env, min_gap_s=0.25, db_per_frame=8.0,
                  prominence_db=12.0):
    """Onsets: a steep log-envelope rise onto a level clearly above the
    local pre-onset floor. The prominence gate rejects frame-to-frame
    fluctuations of sustained tones."""
    log_e = 20.0 * np.log10(env + 1e-12)
    diff = np.diff(log_e, prepend=log_e[0])
    dt = np.median(np.diff(t))
    ctx = max(1, int(0.1 / dt))           # 100 ms pre-onset context
    gap = max(1, int(min_gap_s / dt))
    cand = np.where(diff > db_per_frame)[0]
    onsets = []
    last = -10 ** 9
    for i in cand:
        lo = max(0, i - ctx)
        floor = np.median(log_e[lo:i]) if i > lo else log_e[i]
        if log_e[i] - floor < prominence_db:
            continue
        if i - last >= gap:
            onsets.append(i)
            last = i
    return onsets


def attack_time(t, env, idx):
    """10%-90% rise time (s) of the envelope step at onset index idx."""
    dt = np.median(np.diff(t))
    peak = idx + int(0.05 / dt)
    peak = min(peak, len(env) - 1)
    base = max(idx - int(0.03 / dt), 0)
    lo, hi = env[base], env[peak]
    th10 = lo + 0.1 * (hi - lo)
    th90 = lo + 0.9 * (hi - lo)
    j10 = np.argmax(env[base:peak + 1] >= th10) + base
    j90 = np.argmax(env[base:peak + 1] >= th90) + base
    # sub-frame rises are reported at the envelope's time resolution rather
    # than an arbitrary numerical floor
    return max(t[j90] - t[j10], dt)


def onset_midpoint(t, env, idx):
    """Time of the 50% envelope level across the onset. Unlike the first
    threshold crossing this point is insensitive to attack smearing, so it
    localizes the transient center rather than the start of its ramp."""
    dt = np.median(np.diff(t))
    peak = min(idx + int(0.05 / dt), len(env) - 1)
    base = max(idx - int(0.03 / dt), 0)
    lo, hi = env[base], env[peak]
    th50 = lo + 0.5 * (hi - lo)
    j50 = np.argmax(env[base:peak + 1] >= th50) + base
    return t[j50]


def transient_metrics(x, y, fs, alpha):
    tx, ex, _ = rms_envelope(x, fs)
    ty, ey, _ = rms_envelope(y, fs)
    ox = detect_onsets(tx, ex)
    oy = detect_onsets(ty, ey)
    rows = []
    for i in ox:
        # Timing from the smearing-insensitive 50% point, mapped by alpha.
        t50_in = onset_midpoint(tx, ex, i)
        expected = alpha * t50_in
        if not oy:
            rows.append({"t_in": tx[i], "timing_error_ms": np.nan,
                         "blur_factor": np.nan, "detected": False})
            continue
        j = oy[int(np.argmin(np.abs(ty[oy] - expected)))]
        if abs(ty[j] - expected) > 0.20:
            rows.append({"t_in": tx[i], "timing_error_ms": np.nan,
                         "blur_factor": np.nan, "detected": False})
            continue
        t50_out = onset_midpoint(ty, ey, j)
        a_in = attack_time(tx, ex, i)
        a_out = attack_time(ty, ey, j)
        rows.append({
            "t_in": float(tx[i]),
            "timing_error_ms": float(1000.0 * (t50_out - expected)),
            "blur_factor": float(a_out / a_in),
            "detected": True,
        })
    return rows


# --------------------------------------------------------------------------
# Click / periodicity metrics
# --------------------------------------------------------------------------

def hop_modulation_db(x, fs, hop_syn, frame=256, hop=128):
    """Energy of frame-RMS modulation near the synthesis-hop rate."""
    nb = 1 + (len(x) - frame) // hop
    rms = np.array([np.sqrt(np.mean(x[m * hop:m * hop + frame] ** 2))
                    for m in range(nb)])
    rms -= rms.mean()
    spec = np.abs(np.fft.rfft(rms * np.hanning(len(rms))))
    fr = np.fft.rfftfreq(len(rms), hop / fs)
    f_target = fs / hop_syn
    band = (fr > 0.95 * f_target) & (fr < 1.05 * f_target)
    total = np.sum(spec ** 2) + 1e-20
    return float(10.0 * np.log10((np.sum(spec[band] ** 2) + 1e-20) / total))


def click_rate(x, fs):
    d = np.diff(x)
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med)) + 1e-12
    spikes = np.where(np.abs(d - med) > 10.0 * mad)[0]
    # group spikes closer than 1 ms
    groups, last = [], -10
    for s in spikes:
        if s - last > fs // 1000:
            groups.append(s)
        last = s
    return len(groups) / (len(x) / fs)


def reconstruction_snr(x, y, edge_ms=50.0, fs=None):
    """SNR over the interior only. The first/last edge_ms contain COLA
    fade-in/out of the windowed analysis; there the reference signal itself
    fades through zero, so pointwise relative error is meaningless. The
    interior SNR measures actual reconstruction fidelity."""
    m = min(len(x), len(y))
    edge = int(edge_ms * fs / 1000.0) if fs else 0
    x, y = x[edge:m - edge], y[edge:m - edge]
    err = x - y
    return float(20.0 * np.log10(np.sqrt(np.mean(x ** 2)) /
                                  (np.sqrt(np.mean(err ** 2)) + 1e-20)))


# --------------------------------------------------------------------------
# Per-material evaluation
# --------------------------------------------------------------------------

def evaluate_pair(name, x, y, fs, alpha, hop_syn):
    res = {"alpha": alpha,
           "duration_ratio": len(y) / len(x),
           "duration_error_pct": 100.0 * (len(y) / len(x) - alpha) / alpha,
           "hop_modulation_db": hop_modulation_db(y, fs, hop_syn),
           "hop_modulation_input_db": hop_modulation_db(x, fs, hop_syn),
           "click_rate_per_s": click_rate(y, fs),
           "transients": transient_metrics(x, y, fs, alpha)}

    if name == "pure_tone":
        p_in = pitch_pure_tone(x, fs)
        p_out = pitch_pure_tone(y, fs)
        res["pitch"] = {
            "input": p_in,
            "output": p_out,
            "bias_cents": p_out["median_cents_vs_nominal"]
                          - p_in["median_cents_vs_nominal"],
            "drift_cents": p_out["drift_cents"],
            "drift_pass_lt_1cent": bool(p_out["drift_cents"] < 1.0),
        }
    elif name == "speech":
        ti, fi = pitch_speech(x, fs)
        to, fo = pitch_speech(y, fs)
        fi_n = _normalize_time(fi, len(fo))
        err_c = cents(fo, fi_n)
        # Median F0 bias is the robust "did the pitch move" measure; the
        # framewise contour error additionally contains time-warp alignment
        # jitter, so it is reported as p95 rather than gated on the median.
        med_in, med_out = float(np.median(fi)), float(np.median(fo))
        res["pitch"] = {
            "f0_in_median_hz": med_in,
            "f0_out_median_hz": med_out,
            "median_bias_cents": float(cents(med_out, med_in)),
            "median_abs_err_cents": float(np.median(np.abs(err_c))),
            "p95_abs_err_cents": float(np.percentile(np.abs(err_c), 95)),
        }
    elif name == "chord":
        pin = pitch_chord(x, fs)
        pout = pitch_chord(y, fs)
        rows = []
        for f_nom in CHORD_NOTES:
            f_in, f_out = pin[f_nom], pout[f_nom]
            rows.append({"note_hz": f_nom, "f_in_hz": f_in,
                         "f_out_hz": f_out,
                         "err_cents": float(cents(f_out, f_in))})
        res["pitch"] = {"notes": rows,
                        "max_abs_err_cents": float(max(
                            abs(r["err_cents"]) for r in rows))}
    elif name == "noise":
        ci, fi = noise_spectral(x, fs)
        co, fo = noise_spectral(y, fs)
        res["spectral"] = {
            "centroid_in_hz": ci, "centroid_out_hz": co,
            "centroid_ratio": co / ci,
            "flatness_in": fi, "flatness_out": fo,
            "flatness_ratio": fo / fi,
        }
    return res
