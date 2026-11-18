# Standalone Makefile for the kissfft-based phase vocoder.
# Builds the library sources from the repository root together with the
# phase_vocoder code, so the upstream build system stays untouched.

ROOT ?= ..

CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS += -I$(ROOT) -I.
LDLIBS += -lm

KISS_SRC = $(ROOT)/kiss_fft.c $(ROOT)/kiss_fftr.c
PV_SRC = phase_vocoder.c wav_io.c pv_stretch.c
LIB_SRC = phase_vocoder.c wav_io.c

ALL_SRC = $(PV_SRC) $(KISS_SRC)
ALL_OBJ = $(ALL_SRC:.c=.o)

pv_stretch: $(ALL_OBJ)
	$(CC) $(CFLAGS) -o $@ $(ALL_OBJ) $(LDLIBS)

TEST_OBJ = test_phase_vocoder.o phase_vocoder.o $(ROOT)/kiss_fft.o $(ROOT)/kiss_fftr.o

test: $(TEST_OBJ)
	$(CC) $(CFLAGS) -o test_phase_vocoder $(TEST_OBJ) $(LDLIBS)
	./test_phase_vocoder

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

# kissfft objects land next to the upstream sources.
$(ROOT)/kiss_fft.o: $(ROOT)/kiss_fft.c $(ROOT)/kiss_fft.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<
$(ROOT)/kiss_fftr.o: $(ROOT)/kiss_fftr.c $(ROOT)/kiss_fftr.h $(ROOT)/kiss_fft.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJ) test_phase_vocoder.o pv_stretch test_phase_vocoder

.PHONY: clean test
