/* perf study 40 — what does ONE 2560-byte libc memcpy actually RETIRE?
 *
 *   gcc -O2 -o bin34/tmp40/memcost40 tools/perf40/memcost40.c
 *   perf stat -e instructions:u,branches:u bin34/tmp40/memcost40 <n> [bytes]
 *
 * The whole of perf study 38's price for the threat channel's pure-copy fold —
 * 187.9 instructions per call — rests on a doubled-copy probe. Study 40's arm
 * ladder says the same copy is worth ~112 instructions when it is elided, and
 * that pinning it to a library CALL *lowers* the engine's retired instruction
 * count by ~41 per copy against the 40 inline zmm moves LLVM otherwise emits.
 * Both of those imply a call far cheaper than 187.9, so the absolute number
 * gets measured here instead of inferred.
 *
 * The size is opaque to the compiler (it comes from argv) so this is the same
 * indirect-call path the engine takes, not an inlined expansion. The empty
 * baseline pass is subtracted by running with n and 2n.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    size_t sz = argc > 2 ? (size_t)atol(argv[2]) : 2560;
    /* Two 64-byte-aligned buffers, non-overlapping, both L1-resident — the
     * same geometry as one `ThreatEntry::acc[p]` pair. */
    char *src = aligned_alloc(64, sz);
    char *dst = aligned_alloc(64, sz);
    memset(src, 1, sz);
    memset(dst, 0, sz);
    for (long i = 0; i < n; i++) {
        memcpy(dst, src, sz);
        /* Defeat dead-store elimination without adding a memory op per byte. */
        __asm__ volatile("" : : "r"(dst) : "memory");
    }
    printf("%ld copies of %zu bytes, checksum %d\n", n, sz, dst[sz - 1]);
    return 0;
}
