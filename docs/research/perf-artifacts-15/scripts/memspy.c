// LD_PRELOAD interposer: counts memcpy/memmove calls made by the ENGINE
// binary, bucketed by size and by caller return address (relative to the
// engine's load base, so it can be looked up in the disassembly).
// Deterministic: call counts and sizes are workload facts.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <link.h>

static uintptr_t base = 0;

#define NSLOT 4096
static struct {
  uintptr_t ra;
  unsigned long long n;
  unsigned long long bytes;
} slots[NSLOT];

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

static void record(uintptr_t ra, size_t n) {
  uintptr_t off = ra - base;
  if (off > 0x2000000) return; // not our binary
  size_t h = (off * 2654435761u) & (NSLOT - 1);
  for (size_t i = 0; i < NSLOT; i++) {
    size_t j = (h + i) & (NSLOT - 1);
    if (slots[j].ra == off) { slots[j].n++; slots[j].bytes += n; return; }
    if (slots[j].ra == 0) { slots[j].ra = off; slots[j].n = 1; slots[j].bytes = n; return; }
  }
}

void *memcpy(void *d, const void *s, size_t n) {
  record((uintptr_t)__builtin_return_address(0), n);
  char *dd = d; const char *ss = s;
  for (size_t i = 0; i < n; i++) dd[i] = ss[i];
  return d;
}

void *memmove(void *d, const void *s, size_t n) {
  record((uintptr_t)__builtin_return_address(0), n);
  char *dd = d; const char *ss = s;
  if (dd < ss) { for (size_t i = 0; i < n; i++) dd[i] = ss[i]; }
  else { for (size_t i = n; i > 0; i--) dd[i - 1] = ss[i - 1]; }
  return d;
}

__attribute__((destructor)) static void dump(void) {
  FILE *f = fopen(getenv("MEMSPY_OUT") ? getenv("MEMSPY_OUT") : "/dev/stderr", "w");
  if (!f) return;
  for (size_t i = 0; i < NSLOT; i++)
    if (slots[i].n)
      fprintf(f, "%llu %llu 0x%lx\n", slots[i].n, slots[i].bytes, (unsigned long)slots[i].ra);
  fclose(f);
}
