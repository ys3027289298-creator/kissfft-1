# Memory-bounded block fast convolution

This note describes the block-wise fast convolution added to `kfc.c`
(public API in `kfc_blockconv.h`, exercised by `test/test_blockconv.c`).
It targets long-signal convolution on memory-constrained devices: the working
set depends only on the chosen block/FFT length, **never on the total signal
length**. Nothing in the original kiss_fft/kfc interface changes.

## 1. The two algorithms

Let the filter have length `L`, the configured block length be `B`, and the
FFT length be

```
N = next_fast_size( zero_pad * (B + L - 1) )
```

where `next_fast_size` is kiss_fft's smallest 2,3,5-smooth size at least as
large as the argument, and `zero_pad >= 1` is configurable.

### Overlap-Add (OLA), `KFC_BLOCKCONV_OVERLAP_ADD`

1. Cut the signal into blocks of `B` samples.
2. Zero-pad each block to `N`, forward FFT, multiply by the pre-transformed,
   pre-scaled (`1/N`) filter spectrum, inverse FFT. Each frame yields a
   length-`(B+L-1)` linear convolution of one block with the filter.
3. Add successive frames with an overlap of `L-1`: the tail of frame *k* is
   summed into the head of frame *k+1*. A length-`(L-1)` overlap buffer holds
   the pending tail.
4. A final partial (zero-padded) block flushes the tail.

A block shorter than the filter is **legal** for OLA: `N` simply grows to
`B+L-1`, so the circular product never aliases.

### Overlap-Save (OLS), `KFC_BLOCKCONV_OVERLAP_SAVE`

Each frame is an `N`-point window `[ L-1 history | fresh samples ]`. The number
of fresh (valid) samples per frame is

```
V = N - (L-1) = N - L + 1   (>= B)
```

1. The first frame is primed with `L-1` leading zeros.
2. Forward FFT the window, multiply by the filter spectrum, inverse FFT.
3. The first `L-1` inverse outputs are circular wrap-around and are discarded;
   the last `V` outputs are valid linear-convolution samples.
4. The final `L-1` input samples of the window become the next frame's
   history.
5. At end-of-signal, any queued samples produce their valid outputs, followed
   by a dedicated *ring-out* frame `[ last L-1 real inputs | zeros ]` whose
   first `L-1` valid outputs are the filter's final tail. When fewer than
   `L-1` real inputs exist (signal shorter than the filter), the missing
   leading history slots are zero.

OLS **requires** `B >= L`: with a block shorter than the filter there is no
positive number of valid samples per frame. The allocator rejects this with
`KFC_BLOCKCONV_INVALID` rather than producing a wrong result.

Both methods emit exactly `n_sig + L - 1` samples and are linear convolution.

## 2. Handling "block length < filter length"

| Method | `B < L` behaviour |
|--------|-------------------|
| Overlap-Add | Supported. `N = next_fast_size(B+L-1)` grows so each block's linear convolution fits; correctness test "filter longer than block" covers it. |
| Overlap-Save | Rejected at allocation with `KFC_BLOCKCONV_INVALID` (`-1`), because `V = N-L+1` must stay positive. |

## 3. Memory upper bound

Let `S = sizeof(kiss_fft_cpx)` and `T = sizeof(struct kiss_fft_state)` (the FFT
state is a header plus `N` twiddle complex values, i.e. `T + (N-1)*S` bytes).
One block-convolution config owns, regardless of signal length:

| Item | Size (bytes) |
|------|--------------|
| block-conv state struct | `sizeof(state)` |
| forward FFT config | `T + (N-1)*S` |
| inverse FFT config | `T + (N-1)*S` |
| four `N`-length complex buffers (`fbuf, Xbuf, Hbuf, obuf`) | `4*N*S` |
| overlap/history buffer (`L-1` complex), when `L > 1` | `(L-1)*S` |
| OLS sliding-tail buffer (`L-1` complex), when `L > 1` | `(L-1)*S` |

Because `N` is 2,3,5-smooth, kiss_fft never takes its generic-prime branch and
therefore performs **no additional scratch allocation** during a transform.

So the payload upper bound is

```
M_payload(L, N) = sizeof(state)
                + 2 * (T + (N-1)*S)
                + 4 * N * S
                + 2 * (L-1) * S          (only the (L-1)*S terms when L > 1)
```

The implementation routes every allocation through instrumented hooks that add
an 8-byte header (`kfc_hdr`, one `size_t`) per allocation. With `A` tracked
allocations (`A = 7` for `L = 1`, `A = 9` for `L > 1`), the measured peak is

```
M_peak = M_payload + 8 * A
```

The test computes this formula independently and asserts it equals the
instrumented high-water mark exactly (see section C of the test output). The
same peak is observed after feeding an 8000-sample signal as after a short
one, demonstrating the bound is independent of signal length.

This is a *working* bound for the streaming convolver. A caller's own
input/output buffers are separate and sized by the chunk they choose; because
the API streams, those too can be kept to a fixed, signal-length-independent
size.

## 4. What the test prints (`test/test_blockconv.c`)

- **A. Correctness** — four input classes (short signal, long signal,
  equal-length, filter longer than the block) × both methods × block lengths
  {8,16,32,64} × zero-pad {1,2}, compared against a direct time-domain
  convolution with relative RMS and relative max error.
- **B. Timing** — direct vs OLA vs OLS on a small case and on a long-filter
  case (512 taps, 65536 samples) where FFT convolution wins clearly.
- **C. Peak memory** — analytic formula vs measured high-water mark, plus the
  signal-length-independence check.
- **D. Block-length sweep** — frame count, per-frame valid count, OLA/OLS time
  and peak memory across block lengths 32..512, showing the
  memory/throughput trade-off curve.
- **E. Validation/edges** — `B < L` rejection (OLS) vs acceptance (OLA),
  NULL/bad-argument checks, `L = 1`, and config reuse.

## 5. Scope and constraints honoured

- Only `kfc.c` is modified (purely additive; no original line removed) and new
  files are added: `kfc_blockconv.h`, `test/test_blockconv.c`,
  `test/run_blockconv_test.sh`.
- No build script and no existing public signature is changed. The original
  `kiss_fft*` and `kfc_*` functions are untouched.
- No third-party or network dependency is introduced; the test uses only
  kissfft sources and the C standard library.
- Fixed-point builds are rejected explicitly
  (`KFC_BLOCKCONV_UNSUPPORTED`) because the `1/N` spectrum pre-scaling assumes
  the floating-point FFT scaling convention; float and double are supported.
