//! Perf study 49 — a stall whose TRUE owner is known by construction.
//!
//! The question this exists to answer: `ex_no_retire.load_not_complete` is not
//! a precise event, so the rip in a sample is not guaranteed to be the
//! instruction whose load is blocking retirement. Study 48 measured
//! `HalfkaContext::current_full` at **3.04 stall cycles per load dispatch**
//! against an engine average of 0.906 and could not explain it; one candidate
//! it named and did not test is sample skid from its stall-heavy callees.
//!
//! Observational profiles cannot settle that, because "this symbol really does
//! stall" and "this symbol is absorbing its callee's stall" predict the same
//! profile. So this module injects a stall we already know the answer for: a
//! strictly serial dependent pointer chase through a 32 MB buffer, in its own
//! `#[inline(never)]` leaf function that does nothing else. Every stall cycle
//! it produces belongs to `skidprobe::stall`, with no ambiguity at all. If the
//! profiler then hands a large share of those cycles to the CALLER, attribution
//! is what is broken -- and that is measured, not argued.
//!
//! DIAGNOSTIC ONLY. Behind `--features skidprobe`; never shipped. The result is
//! `black_box`ed and discarded, so the search tree -- and therefore the node
//! count -- is unchanged, which is the gate that says the probe is inert to
//! everything except time.

use std::hint::black_box;
use std::sync::OnceLock;

/// 4 Mi × `u64` = **32 MB**: larger than the 32 MB L3 once the engine's own
/// working set is in it, so a random step is an L3 miss or a DRAM fill.
const LEN: usize = 4 << 20;

static BUF: OnceLock<Box<[u64]>> = OnceLock::new();

fn buf() -> &'static [u64] {
    BUF.get_or_init(|| {
        // A single Hamiltonian cycle over the whole buffer, so `i = b[i]`
        // never falls into a short orbit that would fit in cache. (My first
        // attempt used a multiplicative hash as `next`, which has short rhos
        // and sat entirely in L1 -- it measured "L1 hit" for every step and
        // would have made the probe inert.)
        let mut perm: Vec<u64> = (0..LEN as u64).collect();
        // xorshift64*, so the shuffle needs no dependency
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for i in (1..LEN).rev() {
            let j = (rnd() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        let mut next = vec![0u64; LEN].into_boxed_slice();
        for i in 0..LEN {
            next[perm[i] as usize] = perm[(i + 1) % LEN];
        }
        next
    })
}

/// `steps` strictly serial cache-missing loads. Not inlined, so the stall has
/// its own symbol and cannot be confused with the caller's.
#[inline(never)]
pub(crate) fn stall(seed: usize, steps: usize) -> u64 {
    let b = buf();
    let mut i = seed & (LEN - 1);
    let mut acc = 0u64;
    for _ in 0..steps {
        i = b[i] as usize;
        acc = acc.wrapping_add(i as u64);
    }
    acc
}

/// Call site helper: the value is discarded through `black_box` so LLVM cannot
/// delete the chase, and nothing downstream can observe it.
#[inline(always)]
pub(crate) fn poke(seed: usize) {
    let steps: usize = option_env!("ARCHON_SKID_STEPS")
        .and_then(|s| usize::from_str_radix(s, 10).ok())
        .unwrap_or(4);
    black_box(stall(seed, steps));
}
