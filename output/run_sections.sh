#!/usr/bin/env bash
# Sectioned verification for odd-length real FFT support.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/output"
LOG="$OUT/run_sections.log"
: > "$LOG"

section() {
    echo
    echo "==================== $1 ===================="
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG"
    sleep 8
}

run() {
    echo "\$ $*"
    "$@" 2>&1 | tee -a "$LOG"
}

cd "$ROOT"

section "1/4 build shared library (make all, float)"
make clean >/dev/null
run make all KISSFFT_TOOLS=0

section "2/4 build bundled example test/tr-float (even control)"
gcc -O2 -Itest -I. test/test_real.c -L. -lkissfft-float -lm -o "$OUT/tr-float"

section "3/4 four cases: 1001, 999, even control, inverse round-trip"
LD_LIBRARY_PATH="$ROOT" run "$OUT/tr-float" 1000
LD_LIBRARY_PATH="$ROOT" run python3 "$OUT/odd_fftr_check.py"

section "4/4 summary"
echo "artifacts: $LOG, $OUT/tr-float, $OUT/odd_fftr_check.py"
echo "DONE"
