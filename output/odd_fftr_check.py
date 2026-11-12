#!/usr/bin/env python3
# Small self-contained check: odd-length real FFT via the freshly built lib.
import ctypes, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib = ctypes.CDLL(os.path.join(ROOT, "libkissfft-float.so"))

class Cpx(ctypes.Structure):
    _fields_ = [("r", ctypes.c_float), ("i", ctypes.c_float)]

lib.kiss_fftr_alloc.restype = ctypes.c_void_p
lib.kiss_fftr_alloc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_size_t)]
lib.kiss_fftr.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                          ctypes.POINTER(Cpx)]
lib.kiss_fftri.argtypes = [ctypes.c_void_p, ctypes.POINTER(Cpx),
                           ctypes.POINTER(ctypes.c_float)]

def signal(n):  # tiny hand-made material: offset + two sinusoids
    return [1.5 + .8 * math.sin(.07 * k) + .4 * math.cos(.013 * k + 1)
            for k in range(n)]

def forward(n):
    xs, nbin = signal(n), n // 2 + 1
    f = (Cpx * nbin)()
    lib.kiss_fftr(lib.kiss_fftr_alloc(n, 0, None, None),
                  (ctypes.c_float * n)(*xs), f)
    # reference: even FFT of the same data plus one trailing zero
    even = (ctypes.c_float * (n + 1))(*xs, 0.0)
    ref = (Cpx * (nbin + 1))()
    lib.kiss_fftr(lib.kiss_fftr_alloc(n + 1, 0, None, None), even, ref)
    peak = max(math.hypot(ref[b].r, ref[b].i) for b in range(nbin))
    err = max(math.hypot(f[b].r - ref[b].r, f[b].i - ref[b].i)
              for b in range(nbin)) / peak
    print(f"  n={n:5d} bins={nbin:4d} forward_rel_err_vs_padded_even={err:.3e}")
    assert err < 1e-5

def roundtrip(n):
    xs, nbin = signal(n), n // 2 + 1
    f, back = (Cpx * nbin)(), (ctypes.c_float * n)()
    cfg = lib.kiss_fftr_alloc(n, 0, None, None)
    lib.kiss_fftr(cfg, (ctypes.c_float * n)(*xs), f)
    lib.kiss_fftri(lib.kiss_fftr_alloc(n, 1, None, None), f, back)
    coef = 1.0 / n  # true length, never the n+1 padded length
    err = max(abs(back[k] * coef - xs[k]) for k in range(n))
    peak = max(map(abs, xs))
    print(f"  n={n:5d} bins={nbin:4d} norm_coef=1/{n}={coef:.3e} "
          f"max_abs_err={err:.3e} rel={err / peak:.3e}")
    assert err / peak < 1e-5

forward(1001)
forward(999)
roundtrip(37)  # dedicated inverse round-trip on a small odd length
print("  all cases within tolerance")
