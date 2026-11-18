#!/bin/sh
#
# Build and run the memory-bounded block fast-convolution test.
#
# This script deliberately does NOT touch the repository's existing Makefiles
# or CMake files: it rebuilds the normal kissfft shared library through the
# root Makefile (the new header kfc_blockconv.h lives at the repo root, beside
# kfc.c, so no extra include path is needed), then compiles the standalone
# test against that library. All dependencies come from the repo.
#
# Usage:  sh test/run_blockconv_test.sh
#
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

DATATYPE=${KISSFFT_DATATYPE:-float}

echo "==> Building libkissfft-${DATATYPE} (root Makefile, unchanged)"
make KISSFFT_DATATYPE="$DATATYPE" all

echo "==> Compiling test/test_blockconv"
CC=${CC:-cc}
$CC -O3 -ffast-math -fomit-frame-pointer -I. -Itest \
    -Dkiss_fft_scalar="$DATATYPE" \
    -o test/test_blockconv test/test_blockconv.c \
    -L. -lkissfft-"$DATATYPE" -lm -Wl,-rpath,"$ROOT"

echo "==> Running test/test_blockconv"
./test/test_blockconv
