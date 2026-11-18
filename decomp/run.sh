#!/bin/sh
# Build and run the kiss_decomp acceptance test without touching the repo
# build system. Usage: decomp/run.sh
set -e
cd "$(dirname "$0")/.."
cc -O2 -Wall -Wextra -I. kiss_fft.c decomp/test_decomp.c -lm -o decomp/test_decomp
./decomp/test_decomp
