# Phase vocoder time-stretching for kissfft

Time-scale modification that changes duration without changing pitch.
Built directly on the repository's `kiss_fft` / `kiss_fftr` real FFT.

## Algorithm

Laroche & Dolson (1999) phase vocoder, in three stages:

1. **Short-time Fourier analysis** — Hann-windowed frames of `nfft`
   samples taken every `hop_ana` samples, transformed with `kiss_fftr`.
2. **Phase unwrapping and instantaneous-frequency reassignment** — for each
   spectral peak the phase increment over the analysis hop is wrapped to
   `(-pi, pi]`, converted to an instantaneous frequency, and propagated by
   one *synthesis* hop (`hop_syn = round(alpha * hop_ana)`). Non-peak bins
   are **phase locked** to their nearest magnitude peak, rotating rigidly so
   the spectral shape stays coherent. Peak detection runs on an
   EMA-smoothed magnitude so a tone landing between two bins cannot flip
   the peak identity between frames.
3. **Weighted overlap-add** — inverse FFT, re-windowed, summed on a
   `hop_syn` lattice with a COLA normalization that gives exactly unit
   steady-state gain. The OLA tail is drained one synthesis hop at a time.

`alpha` is the stretch factor: output duration = `alpha` × input duration.
`alpha > 1` slows down, `alpha < 1` speeds up, `alpha == 1` reconstructs the
input (interior SNR 88–113 dB at float precision).

### Why the default hops are `nfft/8`

The synthesis hop must stay at or below `nfft/4` (at least 75% overlap) for
the Hann-windowed overlap-add to average out per-frame phase error; at 50%
overlap a strong hop-rate amplitude modulation appears. With
`hop_ana = nfft/8` and `hop_syn = alpha * hop_ana`, every `alpha` in
[0.5, 2] keeps `hop_syn <= nfft/4`.

## Files

| file | purpose |
|---|---|
| `phase_vocoder.h/.c` | streaming phase-vocoder library (`pv_alloc/process/flush/free`) |
| `wav_io.h/.c` | mono WAV reader (PCM16/24/32, float32) / PCM16 writer |
| `pv_stretch.c` | command-line interface |
| `test_phase_vocoder.c` | C unit tests (validation, duration, identity, streaming) |
| `Makefile` | builds `pv_stretch` and `make test` against the repo's kissfft |

The evaluation harness lives in `../tools_pv/` (numpy only):

| script | purpose |
|---|---|
| `gen_materials.py` | four 48 kHz test signals (pure tone, formant speech, chord, gated noise) |
| `evaluate.py` | pitch / transient / periodicity / identity metrics |
| `run_all.py` | end-to-end driver, writes `results.json` and `report.txt` |

## Build and run

```sh
make            # builds pv_stretch
make test       # C unit tests
./pv_stretch -a 2.0 input.wav output_slow.wav
./pv_stretch -a 0.5 input.wav output_fast.wav
```

Options: `-n <fft size>` (default 2048), `-ha <analysis hop>` (default
`nfft/8`). The library is also usable directly for streaming: call
`pv_process()` on successive chunks and finish with `pv_flush()`.

## Evaluation

```sh
python3 -P ../tools_pv/run_all.py . ../artifacts/pv_work
```

(`-P` keeps the untrusted data directory off `sys.path` while retaining the
user-site numpy install; `-I` would also work where numpy is system-wide.)

Headline results, N=2048, 48 kHz, all 22 acceptance checks passing:

- **Duration**: exact to within 0.5% at alpha 0.5 and 2.
- **Pure-tone pitch drift**: 0.00–0.02 cents (requirement: < 1 cent);
  median pitch bias < 0.01 cent.
- **Chord**: each note within 0.1 cent; speech median F0 bias within 3.5
  cents; noise spectral centroid / flatness preserved within 1%.
- **No periodic clicks**: hop-rate modulation at or below the input's own
  level (best case -75 dB, i.e. numerical floor); click detector reports
  zero impulsive clicks on all tonal material.
- **Identity (alpha=1)**: interior reconstruction SNR 88–113 dB.
- **Transients**: timing preserved to within ~30 ms; attack-time blur
  factor 1.3–8× is reported per material (the expected phase-vocoder
  trade-off with a 43 ms analysis window).
