// perf study 51 -- does Zen 4 honour VMOVNTDQA / PREFETCHNTA on ordinary WB memory?
//
// ONE variable.  Stream N chunks of a large cold array with one of three
// loaders and count how many lines ENTER L1 (`all_l1_data_cache_fills`).
//
//   mode 0  normal 512-bit loads          -> expect ~1 fill per line
//   mode 1  VMOVNTDQA (streaming load)    -> 0 fills iff the hint is honoured
//   mode 2  PREFETCHNTA + normal load     -> fills either way; the question is
//                                            whether the line is marked for
//                                            early eviction (test B answers it)
//   mode 3  PREFETCHT0 + normal load      -> positive control for "a prefetch
//                                            hint reaches the L1 at all"
//
// Test B (mode >= 10): residency.  Interleave a HOT array with a COLD chunk of
// equal size so the pair overflows the 32 KB L1.  If the cold loader really
// bypasses/de-prioritises L1, the HOT array survives and total fills stay near
// the cold-only floor; if it allocates normally, the pair thrashes and fills
// roughly double.
//
//   mode 10 hot + normal cold
//   mode 11 hot + NT cold
//   mode 12 hot + prefetchnta cold
//
// Build: gcc -O2 -march=native -o ntprobe ntprobe.c

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 64
#define COLD_BYTES (64u * 1024u * 1024u)
#define HOT_BYTES (24u * 1024u)
#define CHUNK_BYTES (24u * 1024u)

static __m512i sink;

__attribute__((noinline)) static void pass_normal(const char *p, size_t bytes) {
    __m512i a = _mm512_setzero_si512();
    for (size_t o = 0; o < bytes; o += 64)
        a = _mm512_add_epi32(a, _mm512_load_si512((const void *)(p + o)));
    sink = _mm512_add_epi32(sink, a);
}

__attribute__((noinline)) static void pass_nt(const char *p, size_t bytes) {
    __m512i a = _mm512_setzero_si512();
    for (size_t o = 0; o < bytes; o += 64)
        a = _mm512_add_epi32(a, _mm512_stream_load_si512((void *)(p + o)));
    sink = _mm512_add_epi32(sink, a);
}

__attribute__((noinline)) static void pass_nta(const char *p, size_t bytes) {
    __m512i a = _mm512_setzero_si512();
    for (size_t o = 0; o < bytes; o += 64) {
        _mm_prefetch(p + o + 512, _MM_HINT_NTA);
        a = _mm512_add_epi32(a, _mm512_load_si512((const void *)(p + o)));
    }
    sink = _mm512_add_epi32(sink, a);
}

__attribute__((noinline)) static void pass_t0(const char *p, size_t bytes) {
    __m512i a = _mm512_setzero_si512();
    for (size_t o = 0; o < bytes; o += 64) {
        _mm_prefetch(p + o + 512, _MM_HINT_T0);
        a = _mm512_add_epi32(a, _mm512_load_si512((const void *)(p + o)));
    }
    sink = _mm512_add_epi32(sink, a);
}

int main(int argc, char **argv) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    long iters = argc > 2 ? atol(argv[2]) : 20000;

    char *cold = aligned_alloc(4096, COLD_BYTES);
    char *hot = aligned_alloc(4096, HOT_BYTES);
    if (!cold || !hot) return 2;
    memset(cold, 1, COLD_BYTES);
    memset(hot, 2, HOT_BYTES);

    size_t span = COLD_BYTES - CHUNK_BYTES;
    // Stride by a prime number of lines so consecutive iterations never reuse a
    // line and the chunk start is not 4 KB-congruent every time.
    size_t step = 24u * 1024u + 64u * 7u;

    void (*cold_pass)(const char *, size_t) = pass_normal;
    int with_hot = mode >= 10;
    switch (mode % 10) {
        case 0: cold_pass = pass_normal; break;
        case 1: cold_pass = pass_nt; break;
        case 2: cold_pass = pass_nta; break;
        case 3: cold_pass = pass_t0; break;
        case 9: cold_pass = pass_normal; break;   /* mode 99: hot only */
        default: return 2;
    }

    size_t off = 0;
    for (long i = 0; i < iters; i++) {
        if (with_hot) pass_normal(hot, HOT_BYTES);
        if (mode != 99) cold_pass(cold + off, CHUNK_BYTES);
        off += step;
        if (off > span) off -= span;
    }

    // Keep the accumulator alive.
    int32_t out[16];
    _mm512_storeu_si512((void *)out, sink);
    fprintf(stderr, "mode=%d iters=%ld hot=%d sink=%d\n", mode, iters, with_hot, out[0]);
    // lines touched per iteration, for the expected-fill arithmetic
    fprintf(stderr, "cold_lines/iter=%zu hot_lines/iter=%zu\n", (size_t)(CHUNK_BYTES / LINE),
            with_hot ? (size_t)(HOT_BYTES / LINE) : (size_t)0);
    return 0;
}
