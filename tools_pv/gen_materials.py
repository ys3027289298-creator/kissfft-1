"""Generate the four test materials (48 kHz, mono, float -> PCM16 WAV).

  pure_tone.wav : 440 Hz sine with a mid-stream amplitude step (transient)
  speech.wav    : formant-synthesized vowel sequence /a/-/i/-/u/ with a
                  time-varying F0 contour and a plosive-like onset burst
  chord.wav     : C-major-ish four-note chord, two percussive strikes
  noise.wav     : gated white Gaussian noise with two sharp onsets

Everything is deterministic (seeded) and uses numpy only.
"""

import os
import sys

import numpy as np

FS = 48000
DUR = 3.2

# Vowel formant table: (F1, F2, F3) Hz and bandwidths (B1, B2, B3).
VOWELS = {
    "a": ((730.0, 1090.0, 2440.0), (80.0, 90.0, 120.0)),
    "i": ((270.0, 2290.0, 3010.0), (60.0, 100.0, 120.0)),
    "u": ((300.0, 870.0, 2240.0), (60.0, 80.0, 110.0)),
}


def time_axis(duration=DUR):
    n = int(round(duration * FS))
    return np.arange(n) / FS, n


def edge_fade(x, ms=10.0):
    k = int(FS * ms / 1000.0)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(k) / k))
    x[:k] *= ramp
    x[-k:] *= ramp[::-1]
    return x


def gen_pure_tone():
    t, n = time_axis()
    amp = np.full(n, 0.4)
    step = int(1.5 * FS)
    ramp_k = int(0.005 * FS)
    amp[step:step + ramp_k] = np.linspace(0.4, 0.7, ramp_k)
    amp[step + ramp_k:] = 0.7
    x = amp * np.sin(2 * np.pi * 440.0 * t)
    return edge_fade(x)


def _formant_amplitudes(freqs, formants, bws):
    """Lorentzian formant envelope evaluated at the harmonic frequencies."""
    a = np.zeros_like(freqs)
    for fc, bw in zip(formants, bws):
        a += (bw / 2.0) ** 2 / ((freqs - fc) ** 2 + (bw / 2.0) ** 2)
    return a


def gen_speech():
    t, n = time_axis()
    # F0 contour: three vowel segments with smooth piecewise-linear F0.
    seg_bounds = [0.0, 1.0, 2.1, DUR]
    f0_levels = [110.0, 135.0, 95.0]
    f0 = np.zeros(n)
    for s in range(3):
        i0 = int(seg_bounds[s] * FS)
        i1 = int(seg_bounds[s + 1] * FS)
        f0[i0:i1] = np.linspace(f0_levels[s], f0_levels[(s + 1) % 3],
                                i1 - i0)
    # instantaneous phase of the fundamental, then harmonic h is h*phase
    phase0 = 2 * np.pi * np.cumsum(f0) / FS

    max_h = int(6000.0 / f0_levels[0])  # harmonics up to ~6 kHz
    hs = np.arange(1, max_h + 1)
    x = np.zeros(n)
    cross = int(0.03 * FS)  # 30 ms vowel crossfade
    for s, vowel in enumerate(("a", "i", "u")):
        formants, bws = VOWELS[vowel]
        i0 = int(seg_bounds[s] * FS)
        i1 = int(seg_bounds[s + 1] * FS)
        seg = np.zeros(i1 - i0)
        for h in hs:
            f_h = h * f0[i0:i1]
            env = _formant_amplitudes(f_h, formants, bws)
            seg += env * np.sin(h * phase0[i0:i1])
        win = np.ones(i1 - i0)
        if s > 0:
            win[:cross] = np.linspace(0.0, 1.0, cross)
        if s < 2:
            win[-cross:] = np.linspace(1.0, 0.0, cross)
        x[i0:i1] += seg * win

    x /= np.max(np.abs(x)) + 1e-12
    x *= 0.7

    # Plosive-like onset: short broadband burst at t = 0.08 s.
    rng = np.random.default_rng(7)
    b0 = int(0.08 * FS)
    bk = int(0.02 * FS)
    burst = rng.standard_normal(bk) * np.exp(-np.arange(bk) / (bk / 4.0))
    x[b0:b0 + bk] += 0.5 * burst
    x *= 0.95 / (np.max(np.abs(x)) + 1e-12)
    return edge_fade(x)


def gen_chord():
    t, n = time_axis()
    notes = (261.63, 329.63, 392.00, 440.00)  # C4 E4 G4 A4
    strikes = (0.2, 1.7)
    tau = 0.9
    x = np.zeros(n)
    for t0 in strikes:
        i0 = int(t0 * FS)
        tt = t[i0:] - t0
        env = np.exp(-tt / tau)
        attack = int(0.003 * FS)
        env[:attack] *= np.linspace(0.0, 1.0, attack)
        for f in notes:
            x[i0:] += env * np.sin(2 * np.pi * f * tt)
    x *= 0.7 / (np.max(np.abs(x)) + 1e-12)
    return edge_fade(x)


def gen_noise():
    t, n = time_axis()
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n)
    env = np.zeros(n)
    for onset, offset in ((0.3, 1.2), (1.8, 2.8)):
        i0, i1 = int(onset * FS), int(offset * FS)
        env[i0:i1] = 1.0
        ka = int(0.005 * FS)   # 5 ms attack: sharp onset
        kr = int(0.02 * FS)    # 20 ms release
        env[i0:i0 + ka] = np.linspace(0.0, 1.0, ka)
        env[i1 - kr:i1] = np.linspace(1.0, 0.0, kr)
    x *= env
    x *= 0.9 / (np.max(np.abs(x)) + 1e-12)
    return edge_fade(x)


MATERIALS = {
    "pure_tone": gen_pure_tone,
    "speech": gen_speech,
    "chord": gen_chord,
    "noise": gen_noise,
}


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wavio import write_wav

    for name, fn in MATERIALS.items():
        x = fn()
        path = os.path.join(out_dir, name + ".wav")
        write_wav(path, x, FS)
        rms = np.sqrt(np.mean(x ** 2))
        print("%-10s %6.2f s  peak=%.3f rms=%.4f -> %s"
              % (name, len(x) / FS, np.max(np.abs(x)), rms, path))


if __name__ == "__main__":
    main()
