#!/bin/bash
# Overlap-add streaming demo: staged run, logs land in output/.
set -e
cd "$(dirname "$0")"
OUT=output
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
stage() { printf '\n===== [stage %s] %s =====\n' "$1" "$2"; }
nap() { sleep 2.5; }

stage 1/10 "build kissfft library"
make -s KISSFFT_TOOLS=0
nap

stage 2/10 "build stock example tools/fft-float"
make -s -C tools fft-float
nap

stage 3/10 "verify stock example: FFT roundtrip"
python3 -c "import array,random;random.seed(7);array.array('f',[random.uniform(-1,1) for _ in range(512)]).tofile(open('$OUT/fft_in.bin','wb'))"
tools/fft-float -n 256 < "$OUT/fft_in.bin" | tools/fft-float -n 256 -i > "$OUT/fft_rt.bin"
python3 -c "
import array
a=array.array('f',open('$OUT/fft_in.bin','rb').read())
b=array.array('f',open('$OUT/fft_rt.bin','rb').read())
e=max(abs(x-y/256) for x,y in zip(a,b))
print('fft-float roundtrip max_err=%.3e'%e)
assert e<1e-3, 'stock example check failed'
" | tee "$OUT/stock_example.log"
nap

stage 4/10 "build overlap-add demo"
cc -O2 -Wall -I. -Itools tools/ola_demo.c kiss_fftr.c kiss_fft.c -lm -o "$OUT/ola_demo"
nap

stage 5/10 "case 1: 50% overlap"
"$OUT/ola_demo" 1 | tee "$OUT/case1_overlap50.log"
nap

stage 6/10 "case 2: 75% overlap"
"$OUT/ola_demo" 2 | tee "$OUT/case2_overlap75.log"
nap

stage 7/10 "case 3: continuous push (decoupled chunk sizes)"
"$OUT/ola_demo" 3 | tee "$OUT/case3_continuous.log"
nap

stage 8/10 "case 4: single block passthrough"
"$OUT/ola_demo" 4 | tee "$OUT/case4_passthrough.log"
nap

stage 9/10 "verify results"
grep -h "max_err" "$OUT"/case*.log | awk -F'max_err=' '{e=$2+0;if(e>m)m=e}
END{printf "worst max_err over all segments: %.3e -> %s\n",m,(m<1e-5?"PASS":"FAIL");exit(m<1e-5?0:1)}' | tee "$OUT/verdict.txt"
nap

stage 10/10 "summary"
ls -1 "$OUT" && echo "all logs in $OUT/"
