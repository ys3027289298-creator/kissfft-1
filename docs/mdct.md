# MDCT / IMDCT for kissfft

## What was added

- `kiss_mdct.h` / `kiss_mdct.c`: floating-point MDCT and IMDCT built only on
  the existing, unmodified `kiss_fft` complex transform. No new dependency and
  no change to any existing public function signature.
- `kiss_mdct_sine_window()`: Princen-Bradley sine window
  `w[i] = sin(pi (i + 1/2) / N)`, satisfying `w[i]^2 + w[i+N/2]^2 = 1`.
- `test/test_mdct.c`: small self-contained test (naive O(N^2) reference,
  overlap-add PR, invalid-length rejection, memory size, timing).
- Build wiring only: `CMakeLists.txt`, `Makefile`, `test/CMakeLists.txt`,
  `test/Makefile`. No transform code outside the new module was changed.

## Transform definition

Block length N (even, >= 2), M = N/2 coefficients:

```
X[k] = sum_n x[n] cos( (2 pi / N) (n + N/4 + 1/2) (k + 1/2) )
y[n] = (2/M) sum_k X[k] cos( (2 pi / N) (n + N/4 + 1/2) (k + 1/2) )
```

Each direction uses one length-2N complex FFT with a pre/post twiddle, so any
even N works, including arbitrary mixed-radix sizes such as 1000. The same
configuration serves both directions.

Perfect reconstruction uses 50% overlap-add with the sine window applied on
both analysis and synthesis. A rectangular window does not satisfy the
Princen-Bradley condition and therefore does not reconstruct, by design.

## Error bounds (measured)

Reference = independent O(N^2) double-precision evaluation of the formulas
above. `maxerr` is the maximum absolute sample/coefficient difference.

Float build (`kiss_fft_scalar = float`):

| N | MDCT vs ref maxerr | IMDCT vs ref maxerr | PR maxerr |
|---|--------------------|---------------------|-----------|
| 1000 | 6.39e-06 | 7.59e-07 | 4.37e-07 |
| 1024 | 6.12e-06 | 4.95e-07 | 4.02e-07 |

Double build (`-Dkiss_fft_scalar=double`):

| N | MDCT vs ref maxerr | IMDCT vs ref maxerr | PR maxerr |
|---|--------------------|---------------------|-----------|
| 1000 | 6.76e-12 | 5.74e-13 | 6.33e-14 |
| 1024 | 4.77e-12 | 5.35e-13 | 6.83e-14 |

PR threshold in the test is `1e-4`; every measured float PR error is below
`5e-7`, i.e. > 200x inside the threshold.

## Existing FFT output: unchanged

`artifacts/fft_dump.c` writes complex FFT, inverse FFT and real-FFT outputs
for 17 sizes (powers of two and not) to a binary file. The byte-for-byte
hashes before and after the change are identical:

```
before d50b15d4bc584107020db0f60476dfed096299519eb2c979a928a6422fbdab08
after  d50b15d4bc584107020db0f60476dfed096299519eb2c979a928a6422fbdab08
```

The existing `twotonetest` and `test_real` selftests still pass.

## Timing (best of 7 x 10000 pairs, float, one core)

| N | MDCT+IMDCT per pair |
|---|---------------------|
| 1024 | ~23.0 us |
| 1000 | ~21.9 us |

## Memory upper bound

`kiss_mdct_alloc` returns the exact required size through `lenmem`. With
`S = sizeof(kiss_fft_cpx)` and `C(N)` the bytes of one length-2N kiss_fft
configuration (reported by `kiss_fft_alloc`):

```
bytes <= sizeof(struct kiss_mdct_state) + 2*C(N) + S*(N + N/2 + 4N)
```

Measured (float, S = 8):

| N | reported bytes |
|---|----------------|
| 1000 | 76592 |
| 1024 | 78416 |
| 4096 | 311888 |

Caller buffers follow the existing kiss_fft `mem/lenmem` convention; an
undersized buffer makes `kiss_mdct_alloc` return NULL and write the required
size to `*lenmem`.

## Invalid block lengths

`kiss_mdct_alloc` rejects anything that is not an even integer >= 2
(`0`, negative, odd). It logs an error (when `NDEBUG` is not defined) and
returns NULL. The test exercises N = 0, -8, 1, 3, 7.

## How to reproduce

```sh
# library (CMake, no FFTW needed for the library itself)
cmake -S . -B build -DKISSFFT_TOOLS=OFF -DKISSFFT_TEST=OFF
cmake --build build

# MDCT test
gcc -O2 -I. test/test_mdct.c -Lbuild -lkissfft-float \
    -Wl,-rpath,$PWD/build -lm -o mdct_test
./mdct_test

# double precision
gcc -O2 -I. -Dkiss_fft_scalar=double test/test_mdct.c \
    -Lbuild-double -lkissfft-double -lm -o mdct_test_double

# Makefile path
make
make -C test mdct-float KISSFFTLIB_SHORTNAME=kissfft-float KISSFFT_DATATYPE=float
LD_LIBRARY_PATH=. test/mdct-float

# bit-exact FFT regression
gcc -O2 -I. artifacts/fft_dump.c -Lbuild -lkissfft-float -lm -o fft_dump
./fft_dump fft_out.bin && sha256sum fft_out.bin
```

## Scope note

The module is documented and tested for the floating-point `float` and
`double` configurations. Fixed-point (`int16_t`/`int32_t`) and SIMD vector
builds use a different internal scaling model and are not claimed as
supported; the source still compiles in those builds but `test_mdct.c`
explicitly guards itself to floating point.
