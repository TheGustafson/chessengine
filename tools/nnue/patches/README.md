# External-engine labeling patches

These patches make the exact static-evaluator components used by Archon's
label-quality pipeline machine-readable. Apply each patch from the root of the
matching pinned external-engine checkout:

```text
git apply --check /path/to/tools/nnue/patches/stockfish18-evalraw-v2.patch
git apply /path/to/tools/nnue/patches/stockfish18-evalraw-v2.patch
```

Use `reckless-evalraw-v2.patch` the same way in the Reckless checkout. The
corpus manifest records the compiled executable hash, so rebuilding or changing
either checkout creates a different teacher identity. Version 1 emitted one
unqualified integer whose semantics differed between teachers and is not an
approved production labeling protocol.
