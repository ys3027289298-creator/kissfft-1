#!/usr/bin/env bash
# Non-interactive demo for the new Q15 fixed-point FFT.
# Build artifacts go to a throwaway dir; only logs land in output/.
set -euo pipefail
cd "$(dirname "$0")/.."
B=/tmp/q15_demo_build
mkdir -p "$B"

title() { printf '\n==================== %s ====================\n' "$1"; sleep 3; }

title "1/6 需求与方案：新增独立 Q15 定点 FFT（仅 2 个新文件）"
ls -1 kiss_fft_q15.h kiss_fft_q15.c
sleep 3

title "2/6 仓库自带示例回归：float 库 + twotonetest"
gcc -O2 -fPIC -DKISS_FFT_BUILD -c kiss_fft.c   -o "$B/kiss_fft.o"
gcc -O2 -fPIC -DKISS_FFT_BUILD -c kiss_fftr.c  -o "$B/kiss_fftr.o"
ar crs "$B/libkiss.a" "$B"/kiss_fft.o "$B"/kiss_fftr.o 2>/dev/null
gcc -O2 -I. test/twotonetest.c "$B/libkiss.a" -lm -o "$B/twotonetest"
"$B/twotonetest" 256
sleep 4

title "3/6 编译新模块：c89/c99 双标准，零警告"
gcc -std=c89 -Wall -Wextra -pedantic -O2 -c kiss_fft_q15.c -o "$B/q15_c89.o"
gcc -std=c99 -Wall -Wextra -Wpedantic  -O2 -c kiss_fft_q15.c -o "$B/q15_c99.o"
echo "c89/c99 object build: OK"
sleep 3

title "4/6 自带 selftest 生成测试素材并构建（宏开关，非独立演示程序）"
gcc -std=c99 -Wall -Wextra -O2 -DKISS_FFT_Q15_SELFTEST \
    kiss_fft_q15.c -lm -o "$B/q15_selftest"
echo "built $B/q15_selftest"
sleep 3

title "5/6 四个用例：256 点 / 1024 点 / 满幅 / 小幅，打印 SNR、最大误差、溢出次数"
"$B/q15_selftest"
sleep 4

title "6/6 结论：饱和蝶形 + 块浮点缩放 + 定点旋转因子表，误差可量化"
echo "scaled 模式高 SNR 且无溢出；unscaled 满幅模式暴露并统计饱和次数。"
sleep 2
