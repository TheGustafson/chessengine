# OS Migration Prep (WSL/Windows → Ubuntu, 1 TB SSD)

**Created 2026-07-19.** Checklist for wiping this box and moving to native Ubuntu.
Goal: lose nothing irreplaceable; the new 1 TB SSD removes the disk constraint that
shaped the data strategy.

## Already safe (survives the wipe — verified 2026-07-19)
- **Repo history**: `origin/main` in GitHub (`github.com/TheGustafson/chessengine`) **+**
  a full repo tar at `gs://archon-main-bucket/backups/repo/archon-repo-20260719.tar.gz`
  (1.03 GiB — `.git` history + the **77 unpushed commits** + the **25 uncommitted
  changes** + tracked docs incl. `docs/learn`). Belt-and-suspenders: even without a
  push, the repo is recoverable from this tar.
- **Data in GCS**: 12 official binpacks (232 GiB), shipped nets incl. incumbent
  (`backups`… no — `gs://…/nets/`), lichess bulk export, c1 SF18 relabel, lc0 line.
- **PGO SF18 binaries**: `gs://archon-main-bucket/backups/binaries/{bmi2,vnni512}-pgo`
  (ephemeral scratchpad copies preserved — needed for the relabel worker image).

## Decisions before the wipe (yours)
1. **Push the code to GitHub?** Local `main` is **77 commits ahead** + 25 uncommitted.
   These exist only locally (the "never push" rule) — now tar-backed-up too. Clean fix:
   commit the 25 + `git push`. Reverses the never-push rule → **needs your OK**. (If you'd
   rather keep not-pushing, the GCS tar already covers you.)
2. **`data/obs` (2.3 GiB, local-only)** — D41 helm line, frozen. Back up to
   `gs://…/corpora/obs/` or drop?
3. **`data/nnue/experiments` (2.8 GiB, local-only)** — mostly dead-run checkpoints;
   finals are the only keepers. Drop, or back up a curated subset?
4. **Dead lc0 line (~88 GiB in GCS)** — still pending your earlier delete decision.

## Re-creatable on the new box (no backup needed)
- Repo → `git clone` (GitHub) or restore the GCS tar.
- Corpora → pull from GCS. Rust artifacts → `cargo build`. Reference clones
  (stockfish/reckless/integral) → re-clone. Engines (SF16/18, fastchess, ordo) →
  rebuild/re-download. gcloud + `docker` registry auth → reconfigure.

## The 1 TB SSD changes the data strategy (revisit `post-relabel-data-strategy.md` §3)
The storage math there assumed a 251 GB disk with ~16 GB free. With **1 TB native
Ubuntu (no WSL overhead)**:
- A **2 B-position canonical set (64 GB)** fits locally with room to spare → the budget
  ceiling can rise; the "raw binpacks must stay GCS-only" constraint relaxes.
- Holding **raw binpacks locally** (for fast FEN-manifest builds / local stage-1
  experiments) becomes feasible (~250 GB).
- Native Ubuntu should also lift NPS (no WSL2 translation) — re-baseline the bench +
  the equal-time SF16 gap on the new box (the −42.8 vs SF16 number was WSL2).

## First-boot checklist (new Ubuntu box)
1. `git clone` the repo; verify `git log` matches (or restore the tar + `git fsck`).
2. `gcloud auth login` + `gcloud auth configure-docker us-east1-docker.pkg.dev`.
3. Pull needed corpora/nets from `gs://archon-main-bucket`.
4. `RUSTFLAGS="-C target-cpu=native" cargo build --release`; run `cargo test` +
   the perft tier; re-baseline `bench`.
5. Re-clone references; rebuild/download engines into `engines/`.
6. Re-decide the relabel budget with the bigger disk in mind.
