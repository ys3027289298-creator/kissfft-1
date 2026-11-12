#!/bin/sh
# End-to-end verification for the kissfft window-function module.
# Four titled sections with pauses; all artifacts stay in output/.
set -e
cd "$(dirname "$0")/.."
OUT=output
LOG=$OUT/run.log
PAUSE=${PAUSE:-12}
CFLAGS="-std=c99 -O2 -W -Wall -D_GNU_SOURCE -I."
: > "$LOG"

title() {
    printf '\n========== %s ==========\n' "$1"
    sleep "$PAUSE"
}

title "1/4 baseline: build kissfft library (repo Makefile)"
make clean >>"$LOG" 2>&1
make KISSFFT_TOOLS=0 all >>"$LOG" 2>&1 && echo "library built: $(ls libkissfft-float.so* | head -1)"

title "2/4 repo examples: twotonetest, test_real, kfc cache self-test"
cc $CFLAGS test/twotonetest.c kiss_fft.c kiss_fftr.c -lm -o "$OUT/st-float"
cc $CFLAGS test/test_real.c   kiss_fft.c kiss_fftr.c -lm -o "$OUT/tr-float"
cc $CFLAGS -DKFC_TEST kfc.c   kiss_fft.c -lm -o "$OUT/tkfc-float"
"$OUT/st-float" 240 | tee -a "$LOG"
"$OUT/tr-float" 120 | grep nfft | tee -a "$LOG"
"$OUT/tkfc-float" && echo "kfc cache self-test: PASS" | tee -a "$LOG"

title "3/4 window module: 4 cases (Hann, Hamming, Blackman, rectangular)"
cc $CFLAGS -DKISS_WINDOW_TEST kiss_window.c kiss_fft.c kiss_fftr.c -lm \
    -o "$OUT/kiss_window_test"
"$OUT/kiss_window_test" 64 | tee "$OUT/window_report.txt" | tee -a "$LOG"

title "4/4 summary"
SYM=$(grep -c "PASS (symmetric)" "$OUT/window_report.txt")
CACHE=$(grep -o "PASS" "$OUT/window_report.txt" | wc -l)
CACHE=$((CACHE - SYM))
echo "symmetry: $SYM/4 windows PASS; cache: $CACHE/2 checks PASS" | tee -a "$LOG"
echo "report: $OUT/window_report.txt, full log: $LOG"
