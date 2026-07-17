# Learn ARCHON: a chess engine from first principles

This is a 15-part guided tour of **ARCHON**, a UCI chess engine written in
Rust. The goal is not to skim the surface but to build genuine internal
expertise: by the end you should be able to open any file in `src/`, read the
code, and understand *why* it is shaped the way it is.

## Who this is for

You are comfortable with computer science — algorithms, big-O, a little
systems programming — but you are **not** assumed to be a chess-engine expert
or a Rust expert. Every chess term is defined the first time it appears. Every
non-obvious Rust idiom is explained in a sentence when it comes up. We build
intuition first and reach for formalism only when it earns its keep.

You do need to know how the pieces move (or be willing to keep a rules
reference open). You do *not* need to be a strong chess player. Engine
strength comes from search depth and evaluation accuracy, not from human chess
knowledge, and much of what follows will feel more like compiler engineering
and numerical optimization than like a game.

## How the series is organized

The 15 files fall into four arcs:

- **Foundations (01–04):** the problem, how the board is represented, how
  moves are generated, and how game state is tracked. This is the substrate
  everything else runs on.
- **Search (05–07):** minimax and alpha-beta, the refinements that make search
  practical (iterative deepening, quiescence, aspiration), and the
  transposition table.
- **Making search sharp (08–10):** evaluation (the classical hand-built kind),
  move ordering, and the pruning and reduction heuristics that let a modern
  engine skip almost all of the tree.
- **Learned evaluation and the whole machine (11–15):** NNUE (neural-network
  evaluation) and its training, time management and the UCI protocol, testing
  and statistics, and finally the research frontier — where ARCHON is actually
  being improved today.

Files 01–08 are written to be read in order; each builds directly on the last.
Files 09–15 lean on 01–08 but are more modular — you can read the NNUE pair
(11–12) without having internalized every pruning heuristic in 10, for
example.

## The full table of contents

| # | File | What it covers |
|---|------|----------------|
| 00 | `00-README.md` | This index. |
| 01 | `01-the-problem-of-chess.md` | Game trees, branching factor, why brute force fails, minimax, the search/evaluation split, and what "Elo" measures. |
| 02 | `02-board-representation.md` | Squares as indices 0–63, bitboards as `u64` sets, the mailbox array, and why ARCHON keeps both. |
| 03 | `03-move-generation.md` | Attack tables, sliding pieces and blockers, magic bitboards from scratch, legality and pins, staged generation, and perft as the correctness oracle. |
| 04 | `04-state-zobrist-and-repetition.md` | Make/unmake vs copy-make, the state stack, Zobrist hashing, the five-key family, and repetition and the 50-move rule. |
| 05 | `05-minimax-and-alpha-beta.md` | Minimax to negamax, alpha-beta pruning with a worked example, why move ordering halves the effective depth, the window as a contract, PVS, and mate scores. |
| 06 | `06-iterative-deepening-quiescence-aspiration.md` | Why searching depth 1,2,3,… beats jumping to depth N, the horizon effect, quiescence search, aspiration windows, and search instability. |
| 07 | `07-the-transposition-table.md` | Transpositions and the search DAG, hash tables under a fixed memory budget, replacement policies, stored bound types, and lockless probing. |
| 08 | `08-classical-evaluation.md` | Material values, piece-square tables, tapered evaluation, and the ceiling that pushed the field toward learned evaluation. |
| 09 | `09-move-ordering-and-history.md` | The MovePicker, MVV-LVA, killer/history/continuation tables, and how ordering feeds pruning. |
| 10 | `10-pruning-and-reductions.md` | Null-move pruning, late-move reductions, futility and razoring, and the pruning "waves." |
| 11 | `11-nnue-how-a-network-evaluates-fast.md` | What NNUE is, the accumulator trick, integer inference, and how ARCHON evaluates a network in the hot loop. |
| 12 | `12-nnue-training.md` | Where training data comes from, the loss function, the training pipeline, and provenance. |
| 13 | `13-time-management-and-uci.md` | The UCI protocol, the clock, and how the engine decides how long to think. |
| 14 | `14-testing-and-statistics.md` | Perft, SPRT, Elo estimation, and how you know a change actually helped. |
| 15 | `15-the-research-frontier.md` | Where ARCHON is being improved now: data scale, the research platform, and open questions. |

## Suggested reading order and prerequisites

**Read in order for the full picture.** If you want a targeted path:

- *"How does a chess engine even work?"* — Read 01, then 05 and 06. That is the
  conceptual spine: the problem, the search, and what makes search practical.
- *"How is this made fast?"* — 02, 03, and 07. Bitboards, magic move
  generation, and the transposition table are where the systems engineering
  lives.
- *"How does it decide a position is good?"* — 08 (classical), then 11 and 12
  (learned). Read 08 first even if you only care about NNUE: it frames the
  problem the network solves.
- *"How is it actually improved?"* — 09, 10, 14, and 15. Ordering and pruning
  are where most of the playing strength comes from, and 14–15 explain how any
  of it is measured and pushed forward.

Per-file prerequisites:

- **01** — none. Start here.
- **02** — 01. You need the idea that move generation is a hot inner loop.
- **03** — 02. Bitboards are the vocabulary of move generation.
- **04** — 02, 03. State tracking exists to serve make/unmake during search.
- **05** — 01. The problem statement motivates everything; 02–04 help but are
  not strictly required.
- **06** — 05. Iterative deepening and quiescence are refinements of alpha-beta.
- **07** — 05, 06. The transposition table caches results of the search 05/06
  describe.
- **08** — 01, 05. Evaluation is the leaf value that search propagates.
- **09** — 05, 06, 07. Move ordering exists to make alpha-beta and the TT pay off.
- **10** — 05, 08, 09. Pruning decisions depend on evaluation and ordering.
- **11** — 08. NNUE replaces the classical evaluation of 08.
- **12** — 11. Training produces the network 11 runs.
- **13** — 06. Time management drives iterative deepening.
- **14** — 05, 11. Testing measures the search and evaluation you built.
- **15** — most of the series; it is the "where are we now" capstone.

## A note on the code references

Throughout, we point at real files — "see `src/bitboard.rs`" — but sparingly,
and never as a substitute for the explanation. The prose stands on its own; the
references are there so you can go read the ground truth when you want it. The
design documents in `design/` are the authoritative contracts behind the code;
we cite them occasionally when a decision has a subtle rationale worth reading
in full.

One firewall worth stating up front: ARCHON studies the source of other strong
engines (Stockfish, Reckless, Integral) for *ideas*, but never copies their
code — they carry licenses incompatible with this project. Where a technique
originated elsewhere, we attribute the idea and describe ARCHON's own
implementation of it.

## What you now know

You know the shape of the series, which arc each file belongs to, and a
reading path tailored to what you want out of it. You know the audience
assumptions (CS-literate, chess-and-Rust-agnostic) and the one hard rule about
not copying other engines' code.

**Next:** `01-the-problem-of-chess.md` — why chess is hard for a computer, and
the two-part answer (search plus evaluation) that every strong engine is built
around.
