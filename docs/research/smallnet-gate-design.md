# Smallnet gate design (perf study #3 #8) — the biggest remaining one-box lever

Date: 2026-07-19. Written while the GPU is busy (g2 retrain) + CPU busy (relabel);
design only, ready to build/train when a resource frees.

## Why this, now

The one-box EVAL-quality rungs are tapped out (stage-1/stage-2/width all measured
shut; data 0-for-15 self-sourced). But **speed converts to equal-time Elo on the
D40 arch — proven** (opt#1 +13-15% NPS = **+28.8±10.3 Elo**, exp-0101; ~2 Elo/1%).
The smallnet gate is the largest remaining *bit-identical-eval* speed lever:
**+8-15% NPS → ≈ +16-30 Elo**, and it's SF-proven Elo-positive at LTC. Bigger
than EXP-UNC-1 (+3-15 internal). It's a training + routing-infra project, not a
micro-opt.

## The idea (SF-proven, reimplemented idiomatically — never copy)

SF ships a big net + a cheap "small" net and routes lopsided/decided positions
(large |eval|) to the small net. A large fraction of real-tree nodes are decided
→ take the cheap path → higher NPS → more nodes at equal time → Elo.

ARCHON's cheap path is natural: **a HalfKA-only net skips the entire FullThreats
channel** — the 46.6 MB threat table (the #1 D1/LL miss source), the whole lazy
observer, and the i32 activate/threat-fold blocks. The loader ALREADY supports
HalfKA-only CENN nets end-to-end (`threat_weights.is_empty()` short-circuits;
zero threat channel in the head — sota_threats.rs / sota_head.rs). So the small
net is ~⅓ the eval cost with the infra already half-built.

## Components

1. **Train the smallnet** (GPU, ~2h — the D40 pipeline already has it): a
   HalfKA-only distilled net, `D40_HALFKA_ONLY=1` in d40_features.rs. Distill
   from the big net (d40-uho-720sb) so the two agree on non-decided positions
   (SF distills the smallnet from the big net). A smaller L1 (e.g. 512) makes it
   cheaper still; the loader takes any L1.
2. **Second accumulator context** (Rust): SotaContext today owns one HalfKA +
   one FullThreats stack. Add an optional second HalfkaContext for the smallnet
   (its own king-bucketed accumulator). The threat stack is unused on the cheap
   path. Per-thread, ~small.
3. **The gate** (Rust, in `SotaContext::evaluate` or `eval/mod.rs`): route to the
   smallnet when the position is DECIDED. SF keys on |eval| / material. First cut:
   after computing the CHEAP (smallnet) eval, if |smallnet_eval| ≥ THRESH (large
   margin → decided) use it; else fall through to the big net. (SF actually runs
   the smallnet first and only escalates to the big net when the smallnet's
   output is near zero / uncertain — "psqt-ish" gating.) THRESH is a tuned
   constant (env-gated for the SPSA/gate sweep, default = big-net-only so it's
   bit-identical off).
4. **Loader/format**: a second `EvalFile` (env `ARCHON_SMALLNET`) → a second
   leaked SotaNetwork; `set_eval_file` installs both. CENN format unchanged (the
   smallnet is just a HalfKA-only CENN).

## Gate criterion — the design decision

- **SF's actual gate** (reimplement the idea): run the SMALL net first; escalate
  to the BIG net only when |small eval| is small (position unclear, where the big
  net's extra capacity matters). So the big net runs on the ~decided-unclear
  minority; the cheap net covers the decided majority. Ceiling: 30-50% of evals
  at ~⅓ cost → +8-15% NPS.
- Elo risk lives entirely in the gate: if it escalates too rarely, strength drops
  where the big net mattered; too often, no speed win. It is SPSA-testable cheaply
  (one threshold). Env-gate THRESH so default = big-only (bit-identical 147691).

## Sequencing (when a resource frees)

1. ~~Build the second-context + gate infra~~ **DONE 2026-07-19, commit 62102ca**
   (Rust, env-gated default-off → bit-identical, sota 469205 == baseline / default
   depth-6 701605). `ARCHON_SMALLNET` + `ARCHON_SMALLNET_THRESH`; second
   `HalfkaContext` in lockstep; `evaluate()` routes on `|cheap eval| ≥ thresh`.
2. **Train the HalfKA-only cheap net (GPU — after g2/EXP-UNC-1 frees it).**
3. Gate the package: sweep `ARCHON_SMALLNET_THRESH`; equal-time vs big-net-only
   (the ship metric). Expected +8-15% NPS → +16-30 Elo if the gate replicates SF's.

### Cheap-net launch recipe (one step, when the GPU frees)

The trainer's L1 is baked at **768** (`ACC`; the `pairwise_mul` on 768-wide
accumulators — there is NO width knob today, so a narrower cheap net is a
deferred trainer arch change). So the cheapest net WITHOUT a trainer change is
**HalfKA-only at L1=768**: `D40_HALFKA_ONLY=1` skips the whole FullThreats channel
(the 46.6 MB table + observer + 768-lane fold, ~half the D40 eval cost). That IS
the speed win the gate exploits — the runtime already skips threats on the cheap
path (`ZERO_THREAT`, no fold), and a HalfKA-only CENN's `threat_weights` are
empty, so the two agree by construction.

Adapt the g2 config (`data/nnue/experiments/d40-uho-unc-g2/config.json`) — env
deltas only:
- `D40_HALFKA_ONLY`: `"0"` → `"1"`   (drop the threat channel)
- `D40_HETERO_NLL`:  `"1"` → `"0"`   (no variance head on the cheap net)
- `D40_ID` / `D40_OUT` / `checkpoint_dir` / `net_id`: a fresh id, e.g.
  `d40-uho-halfka-cheap-240sb` (append-only ledger rejects a reused id)
- `parity.raw`: repoint to the new ckpt dir; keep parity enabled
- Corpus/SB: UHO stage-1 full, 240 SB (match the big net so they agree on decided
  positions — SF distills the smallnet from the big net; same-corpus SF labels is
  the affordable first cut). Reconsider stage-2 warm-retrain vs the then-current
  incumbent at launch.

Then: pack → `ARCHON_SMALLNET=<cheap.cenn> ARCHON_SMALLNET_THRESH=<t>` and sweep
`t` (start ~300-600cp: only clearly-decided nodes take the cheap path) at
fixed-node (NPS + node-count) AND equal-time (the ship metric) vs `t=0`.

## Relation to EXP-UNC-1

Orthogonal + composable: EXP-UNC-1 improves the eval's *consumers*; the smallnet
gate improves *speed*. Both compound. If EXP-UNC-1 lands a variance head, the
smallnet path simply reports σ̂=None (no variance head) → consumers no-op there,
which is correct (decided positions need no uncertainty modulation).
