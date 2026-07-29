// LD_PRELOAD interposer: counts memcpy/memmove/memset calls made by the ENGINE
// binary, bucketed by size and by caller return address (relative to the
// engine's load base, so it can be looked up in the disassembly).
// Deterministic: call counts and sizes are workload facts.
//
// study 16: `memset` added (study 15 §8 item 1). Study 15's biggest arm was a
// 1280-byte libc memset per evaluation that this interposer, hooking only
// memcpy/memmove, could not see; it took reading disassembly to find. The
// three hooks share one table, so a caller's row is its TOTAL libc byte
// traffic; the `kind` column says which.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <link.h>

static uintptr_t base = 0;

#define NSLOT 8192
static struct {
  uintptr_t key; // (kind << 56) | offset-from-load-base; 0 == empty
  unsigned long long n;
  unsigned long long bytes;
} slots[NSLOT];

// kind tags, stored in the key's top byte so one table serves all three hooks
#define K_CPY 1
#define K_MOV 2
#define K_SET 3

static int phdr_cb(struct dl_phdr_info *info, size_t sz, void *data) {
  (void)sz;
  (void)data;
  if (info->dlpi_name == NULL || info->dlpi_name[0] == '\0') {
    base = (uintptr_t)info->dlpi_addr;
    return 1;
  }
  return 0;
}

__attribute__((constructor)) static void init(void) { dl_iterate_phdr(phdr_cb, NULL); }

static void record(uintptr_t ra, size_t n, unsigned kind) {
  uintptr_t off = ra - base;
  if (off > 0x2000000) return; // not our binary
  uintptr_t key = off | ((uintptr_t)kind << 56);
  size_t h = (key * 2654435761u) & (NSLOT - 1);
  for (size_t i = 0; i < NSLOT; i++) {
    size_t j = (h + i) & (NSLOT - 1);
    if (slots[j].key == key) { slots[j].n++; slots[j].bytes += n; return; }
    if (slots[j].key == 0) { slots[j].key = key; slots[j].n = 1; slots[j].bytes = n; return; }
  }
}

// The replacement bodies must not call libc's mem* (infinite recursion), but a
// byte loop is too slow to be usable: study 16's first attempt ran a 3000-node
// bench for >10 minutes without finishing, because start-up zeroes ~20 MiB of
// history tables and a 16 MiB TT. Word-at-a-time keeps it to ~1/8 the trips.
// `-fno-builtin` is not needed: these are already the definitions.
static inline void copy_fwd(unsigned char *d, const unsigned char *s, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) { unsigned long long w; __builtin_memcpy(&w, s + i, 8); __builtin_memcpy(d + i, &w, 8); }
  for (; i < n; i++) d[i] = s[i];
}

void *memcpy(void *d, const void *s, size_t n) {
  record((uintptr_t)__builtin_return_address(0), n, K_CPY);
  copy_fwd(d, s, n);
  return d;
}

void *memmove(void *d, const void *s, size_t n) {
  record((uintptr_t)__builtin_return_address(0), n, K_MOV);
  unsigned char *dd = d; const unsigned char *ss = s;
  if (dd < ss) copy_fwd(dd, ss, n);
  else { for (size_t i = n; i > 0; i--) dd[i - 1] = ss[i - 1]; }
  return d;
}

// memset is the study-15 blind spot. Interposing it is riskier than memcpy:
// the dynamic loader and libc's own startup call memset before our constructor
// has run, so `base` may still be 0 -- record() throws those away via the
// 0x2000000 offset filter, and the byte loop below is always correct.
void *memset(void *d, int c, size_t n) {
  record((uintptr_t)__builtin_return_address(0), n, K_SET);
  unsigned char *dd = d;
  unsigned long long w = 0x0101010101010101ULL * (unsigned char)c;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) __builtin_memcpy(dd + i, &w, 8);
  for (; i < n; i++) dd[i] = (unsigned char)c;
  return d;
}

__attribute__((destructor)) static void dump(void) {
  FILE *f = fopen(getenv("MEMSPY_OUT") ? getenv("MEMSPY_OUT") : "/dev/stderr", "w");
  if (!f) return;
  static const char *kn[] = {"?", "memcpy", "memmove", "memset"};
  for (size_t i = 0; i < NSLOT; i++)
    if (slots[i].n) {
      unsigned kind = (unsigned)(slots[i].key >> 56);
      fprintf(f, "%llu %llu 0x%lx %s\n", slots[i].n, slots[i].bytes,
              (unsigned long)(slots[i].key & 0xffffffffffffffUL),
              kn[kind < 4 ? kind : 0]);
    }
  fclose(f);
}
