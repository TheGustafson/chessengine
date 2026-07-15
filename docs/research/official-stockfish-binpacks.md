# Official Stockfish Master Binpacks Contract

Status: remotely exposed and direct-loader smoke tested 2026-07-15.

## Identity and access

- Source: `https://huggingface.co/datasets/official-stockfish/master-binpacks/tree/main`
- Owning organization: `official-stockfish`
- Revision currently exposed: moving branch `main`
- License declared by the dataset: ODbL
- Physical release at the captured manifest: 13 `.binpack` files totaling
  287,102,338,770 bytes
- Associated trainer: `https://github.com/official-stockfish/nnue-pytorch`

The complete collection is intentionally not mirrored onto the local 251 GiB
workspace. `tools/nnue/fetch_official_stockfish_binpack.py` snapshots exact
Hugging Face paths, sizes, and LFS object hashes in
`data/corpora/official-stockfish-master-binpacks/main/remote-manifest.json`.
It can range-fetch an audit prefix or deliberately fetch one complete named
file with LFS SHA-256 verification. This makes every remote file addressable
without pretending the moving branch is an immutable release. A production
selection must record the manifest and each file's LFS hash.

ODbL attribution and share-alike obligations must be reviewed before
redistributing the original collection or a derived database. Model-artifact
licensing is a separate question and must not be inferred from this note.

## Format and first audit

The first 128 MiB of `fishpack32.binpack` was fetched by HTTP range. Because
the range ended inside a compressed chunk,
`tools/nnue/salvage_sfbinpack_prefix.py` retained the first 127 complete chunks
and discarded the 1,026,438-byte tail. The resulting 133,191,290-byte prefix
has SHA-256
`6dc3d5917b1dce843d798ccd6a9bb8d371e9c3d6d810aed34b7dad6e1fa26498`.

`tools/nnue/sfbinpack-audit/` decoded the complete prefix without error:

- 61,070,868 positions and 60,559,547 valid within-game continuation pairs;
- game result from side-to-move perspective: 12,261,179 losses, 36,801,226
  draws, and 12,008,463 wins;
- 30,342,847 White-to-move positions and 4,736,207 in-check positions;
- score, best move, ply, rule-50 counter, side to move, and game result are
  retained.

Unlike the Lichess Parquet source, this format supplies outcome/WDL and the
rule-50 clock in a compact sequential-game corpus. The exact engine, network,
search limits, and generation recipe are not sufficiently documented per
named file on the dataset page, so provenance remains
`official-stockfish/master-binpacks:fishpack32`, not `SF18 gold`.

## Direct training smoke

`tools/nnue/train_net0.rs` now accepts `NET0_FORMAT=sfbinpack` through Bullet's
streaming `SfBinpackLoader`; no multi-gigabyte expansion is necessary. The
initial compatibility filter removes early plies, in-check positions,
captures/promotions, and extreme scores. This is a compatibility smoke, not
yet a claim that Archon's filter matches the current official Stockfish
dataloader's smart and WLD sampling.

A net-0 fine-tune used a 0.3 game-outcome blend, learning rate `1e-5`, and
approximately one million accepted presentations per checkpoint. Checkpoint 4
scored 37-25-58 over 120 games against net-0 at 5,000 nodes (about -61 Elo),
dramatically less destructive than the outcome-free Lichess fine-tune (about
-407 Elo). The first checkpoint scored 29-24-27 in an 80-game screen, while
later checkpoints declined. This is promising evidence for the corpus and for
preserving WDL, but the first result has wide uncertainty and is not a
promotion. Next tests concentrate exposure and learning rate around the first
checkpoint and compare at multiple node budgets.

The calibrated follow-up reduced the learning rate to `3e-6`. At roughly four
million accepted presentations, the 0.3-outcome-blend candidate scored
152-94-154 over 400 games at 5,000 nodes (-1.7 Elo, paired 95% interval
approximately [-30, +27]) and -10.4 Elo at 20,000 nodes (interval [-37, +16]).
This establishes a non-destructive update, not an improvement. Holding the
step fixed and increasing the outcome fraction to 0.5 gave +7.8 Elo over 400
games at 5,000 nodes (interval [-21, +37]); fraction 0.7 gave +6.9 Elo and
fraction 0.1 was effectively even. The 0.5 candidate is now the provisional
recipe leader, pending its 2k/20k curve and replication. The result is exactly
why offline loss is not used for selection.

## Promotion rules

1. Keep every selected filename and LFS object hash in training provenance.
2. Audit full-file decode, score/result perspective, legal moves, chunk
   continuity, and distributions before scaling.
3. Establish sampling/filter semantics; do not assume sequential positions are
   independent or let one long-game neighborhood cross held-out partitions.
4. Retain outcome/WDL unless a controlled game test proves a replacement.
5. Use early fixed-node games to stop recipes whose offline loss improves while
   engine strength falls.
6. Treat this as high-value bootstrap data, not as the controlled Stockfish 18
   oracle required for Archon's final quality labels.
