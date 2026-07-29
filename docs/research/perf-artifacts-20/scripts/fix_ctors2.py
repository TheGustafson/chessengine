"""perf study 20 arm 2 — second (and final) mechanical retype pass."""

EDITS = {
    "src/eval/nnue/sota_loader.rs": [
        (
            "    pub(crate) fn test_psqt_table() -> Box<[i32]> {",
            "    pub(crate) fn test_psqt_table() -> Aligned<i32> {",
        ),
    ],
    "tests/cenn_v2_smoke.rs": [
        (
            "    let halfka_weights: Box<[i16]> = (0..HALFKA_DIMS * ACC)",
            "    let halfka_weights: Aligned<i16> = (0..HALFKA_DIMS * ACC)",
        ),
        (
            "    let ft_bias: Box<[i16]> = (0..ACC).map(|i| (i % 5) as i16 - 2).collect();",
            "    let ft_bias: Aligned<i16> = (0..ACC).map(|i| (i % 5) as i16 - 2).collect();",
        ),
        (
            "    let threat_weights: Box<[i8]> = (0..THREAT_DIMS * ACC).map(|i| (i % 9) as i8 - 4).collect();",
            "    let threat_weights: Aligned<i8> =\n        (0..THREAT_DIMS * ACC).map(|i| (i % 9) as i8 - 4).collect();",
        ),
        (
            "    use chessengine::eval::nnue::sota_loader::{self, SotaNetwork};",
            "    use chessengine::eval::nnue::aligned::Aligned;\n"
            "    use chessengine::eval::nnue::sota_loader::{self, SotaNetwork};",
        ),
    ],
    "tests/eval_file_reload.rs": [],
}

for path, subs in EDITS.items():
    if not subs:
        continue
    text = open(path).read()
    for old, new in subs:
        assert text.count(old) >= 1, (path, old[:60], text.count(old))
        text = text.replace(old, new)
    open(path, "w").write(text)
print("ok")
