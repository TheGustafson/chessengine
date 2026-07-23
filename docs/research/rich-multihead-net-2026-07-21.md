# Rich multi-head net — heads for everything in the canonical data (2026-07-21)

**Goal (user):** one shared D40 trunk feeding an output head for *every* signal SF18
gives us, not just the cp. Two payoffs: (1) **aux heads that improve the eval itself**
and are *dropped at inference* (T=1, zero NPS cost — the H-AUX bet); (2) **output
heads search/TM consume** (WDL→TM, decisiveness→margins). Built on the existing
"extra-head + sidecar + env-gated consumer" machinery (wdl-head-design Part A) — mostly
plumbing-reuse.

## The canonical signals (design/16 `.ceobs`) and the head for each

| signal | where it lives | head | shape | loss | deploy | status |
|---|---|---|---|---|---|---|
| `final_score` (cp) | sfbinpack `.score` | **value** | L3→BUCKETS | MSE (WDL-blended) | output (shipped) | ✅ live |
| game result (WDL) | sfbinpack `.result` | **WDL** | L3→3 | softmax-CE | output → TM/optimism | ✅ **trainer built** (this session) |
| `best_move` | sfbinpack `entry.mv` → packed into `ChessBoard.extra` | **policy / best-move** (the H-AUX star) | 2× from/to 64-CE off `hl1` | softmax-CE | **aux, dropped at inference** | ✅ **trainer built** (this session) |
| `id_snapshots` (eval-vs-depth) | ceobs only | **decisiveness / uncertainty** | L3→1 (`l3_var` exists) | NLL or residual-MSE | output → margins/TM | ⏳ needs trajectory export + target |
| `final − shallow` (search-aug) | ceobs id_snapshots | **search-value delta** | L3→1 | MSE | aux, dropped | ⏳ needs ceobs export |
| `pv` / per-move value (L4) | ceobs L4 (35× relabel) | deeper policy | — | — | aux | ⏳ expensive relabel; last |

## The governing principle (from wdl-head-design Part D — do NOT re-derive)
- **Additive loss only:** `L = value_MSE + Σ λ_h · loss_h`. Gradients ADD; an aux head
  never reweights the eval gradient (that multiplicative NLL is what sank EXP-UNC-1:
  −179). At small λ it's a gentle regularizer; λ→0 recovers the control exactly.
- **Test co-train AND frozen for every head.** Co-train may improve the eval
  (AlphaZero/KataGo; our policy pilot λ=0.01 deployed eval-only = **+63**). Frozen-trunk
  (`train_frozen` + `D40_INIT_RAW`) is the zero-eval-risk fallback / output-only path.
- **Parity discipline:** each head is env-gated; unset → not output-reachable, not in the
  loss → dropped from the raw ckpt → value CENN byte-identical (D24). λ→0 is the control.

## WDL head — implemented this session (trainer side)
`d40_features.rs`: added `wdl_target` (3-way one-hot from `pos.result`, stm-relative),
a `l3_wdl` head (`L3→3` off `hl3`, so bucket-specific), and `D40_WDL_LAMBDA` folding
`softmax_crossentropy_loss` additively into the loss. λ=0 default → head dropped, CENN
parity. Compiles; smoke + parity gate pending the GPU (from-scratch-1024 owns it).
**Remaining for WDL end-to-end** (wdl-head-design Part C): (a) netpack/sidecar the 3-wide
head (extend `extract_unc_head.py` → wdl); (b) engine loads it (widen the optional head
1→3); (c) map to a consumer scalar (decisiveness `|W−L|` → existing `unc_margin_scale`/
`unc_tm_factor`; optional new optimism consumer from W-prob). Then SPSA c1/c2 + gate.

## Policy head — implemented this session (the move-plumbing SOLVED)
`ChessBoard` has `pub extra: [u8; 3]` unused bytes (struct is exactly 32 B). So the
vendored `bullet_lib` `convert_to_bulletformat` now packs the teacher best move
(stm-relative from/to squares, vertical-flip for black) into `extra[0..2]` — layout
unchanged, `.bin` compat intact, value training byte-identical (features come from
`pcs`/`occ`). The trainer reads `extra` → two 64-way from/to one-hot targets → two
policy heads (`HL→64` off the shared pairwise trunk `hl1`, so the accumulator+trunk
must encode move preference) → additive softmax-CE, `D40_POL_LAMBDA`-gated. Deploy
**eval-only** (drop the heads → T=1, zero NPS). λ=0 → dropped → CENN parity. Compiles.
TODO refinement: the target uses only the vertical flip, not the HalfKA `_hm`
horizontal king-file mirror — a first-cut inconsistency the trunk can absorb; refine
if the signal is weak.

## THE experiment this unlocks (H-AUX)
`A = value-only` (control, D40_POL_LAMBDA=0) vs `B = value + λ·policy` (sweep λ ∈
{0.1, 0.3, 1.0}), identical arch/data/schedule, **both deployed eval-only** → does the
policy aux **improve the value eval at zero NPS cost**? Gate B-CENN vs A-CENN. This is
the whole bet (AGZ/KataGo/our +63 pilot). Runs after the policy smoke confirms the
head trains + the move-packing feeds a real (decreasing-CE) target.

## Still blocked (need ceobs trajectory export)
- **Decisiveness/uncertainty (co-train BROKEN — frozen only)** and **search-aug
  (`final − shallow`)** need `id_snapshots` (eval-vs-depth) exported from ceobs to a
  trainable column. The `l3_var` head + sidecar + wired margin/TM consumers already
  exist — only the *target* is missing. Uncertainty additionally can't be co-trained
  (its target is f(eval) → needs `with_no_grad` detach = the buggy path that sank
  EXP-UNC-1); it's frozen-trunk only. Search-aug's target IS independent → co-trainable
  once exported.

## Sequencing
1. **WDL** (data ready, ~90% plumbing) → smoke+parity → engine consume → gate. First and now.
2. **Policy/best-move aux** (move-preserving loader) → the H-AUX experiment: A=value-only vs
   B=value+λ·policy, both eval-only deploy → does the aux **improve the eval** at zero NPS?
3. **Decisiveness / search-aug** (ceobs trajectory export) → TM content (ARCHON's TM has
   none today) + the search-augmentation signal (design/19 / D41).
4. **L4 per-move policy** (35× relabel) — last, only if 1–3 pay.

Gate every head vs the value-only control; internal Elo deflates 2–3× externally; track
vs SF16 not the internal ladder. This is Axis A of post-d40-research-direction.md.
