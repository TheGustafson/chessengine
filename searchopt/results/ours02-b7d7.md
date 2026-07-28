# OURS #2 — `8/1r4p1/1P2k1Pp/R1B1p3/4p1P1/p3N3/3K4/r7 b - - 7 60`

Played **b7d7** (−0.27 wp). Oracle **a3a2**. Rook ending, Black to move.

## Step 2 — where the divergence is

Free search, root, both engines (Black POV, SF printed frame):

| depth | ARCHON | SF18 |
|---|---|---|
| 10 | b7d7 | b7d7 |
| **12** | b7d7 | **a3a2** ← SF switches |
| 14–16 | b7d7 | a3a2 |
| **18** | **a3a2** ← we switch | a3a2 |

**6-ply lag.** Forced, one move at a time:

| depth | A b7d7 | A a3a2 | S b7d7 | S a3a2 |
|---|---|---|---|---|
| 14 | −41 | −98 | −18 | −33 |
| **16** | −24 | −75 | **−197** ← SF sees it | −18 |
| 18 | −34 | −42 | −221 | −7 |
| **20** | **−231** ← we see it | −25 | −227 | −8 |

Both engines value the ORACLE move `a3a2` almost identically at every depth
(−98→−21 vs −129→−6). **The entire disagreement is about the BLUNDER**: SF's score for
`b7d7` collapses at **d16**, ours at **d20**. Consistent with the corpus-wide finding
that on OURS positions we are *confidently wrong* rather than uncertain — we hold b7d7 at
about −30 for four plies after SF has already priced it at −200.

## Step 3 — the sentence

Not yet writable. The discarding is **inside the b7d7 subtree**: we do not surface its
refutation until d20. The next descent is the node after `b7d7`, asking which of Black's
replies SF resolves by d16 that we do not.

## Next
1. Make the move `b7d7`, get SF's refutation line at high nodes.
2. Walk down it one ply at a time, comparing per-depth scores, until reaching a node where
   both engines agree — the ply above is where it is discarded.
3. Only then look at `move_index` / `final_reduction` / `scout_depth` at that node.

**Do not** run any aggregate, and do not propose an arm before the Step-3 sentence exists.
