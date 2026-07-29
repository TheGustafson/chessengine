"""perf study 20 arm 2 — one-shot mechanical retype of the weight-table
construction sites to `Aligned<T>`. Kept as an artifact so the diff is
reproducible, not because it is a tool anyone should run twice."""

EDITS = {
    "src/eval/nnue/sota_loader.rs": [
        (
            "        let halfka_weights: Box<[i16]> = (0..HALFKA_DIMS * ACC)",
            "        let halfka_weights: Aligned<i16> = (0..HALFKA_DIMS * ACC)",
        ),
        (
            "        let ft_bias: Box<[i16]> = (0..ACC).map(|_| small_i16(&mut prng)).collect();",
            "        let ft_bias: Aligned<i16> = (0..ACC).map(|_| small_i16(&mut prng)).collect();",
        ),
        (
            "        let threat_weights: Box<[i8]> = (0..THREAT_DIMS * ACC)",
            "        let threat_weights: Aligned<i8> = (0..THREAT_DIMS * ACC)",
        ),
        (
            "            psqt_weights: Vec::new().into_boxed_slice(),",
            "            psqt_weights: Vec::new().into(),",
        ),
        (
            "        full.threat_weights = Vec::new().into_boxed_slice();",
            "        full.threat_weights = Vec::new().into();",
        ),
        (
            "        net.psqt_weights = test_psqt_table()[..HALFKA_DIMS * BUCKETS]\n"
            "            .to_vec()\n"
            "            .into_boxed_slice();",
            "        net.psqt_weights = test_psqt_table()[..HALFKA_DIMS * BUCKETS].to_vec().into();",
        ),
    ],
    "src/eval/nnue/sota_context.rs": [
        (
            "        net.threat_weights = Vec::new().into_boxed_slice();",
            "        net.threat_weights = Vec::new().into();",
        ),
    ],
    "tests/cenn_v2_smoke.rs": [
        (
            "        psqt_weights: Vec::new().into_boxed_slice(),",
            "        psqt_weights: Vec::new().into(),",
        ),
    ],
    "tests/eval_file_reload.rs": [
        (
            "        halfka_weights: vec![0i16; HALFKA_DIMS * ACC].into_boxed_slice(),",
            "        halfka_weights: vec![0i16; HALFKA_DIMS * ACC].into(),",
        ),
        (
            "        ft_bias: vec![0i16; ACC].into_boxed_slice(),",
            "        ft_bias: vec![0i16; ACC].into(),",
        ),
        (
            "        threat_weights: Vec::new().into_boxed_slice(),",
            "        threat_weights: Vec::new().into(),",
        ),
        (
            "        psqt_weights: Vec::new().into_boxed_slice(),",
            "        psqt_weights: Vec::new().into(),",
        ),
    ],
}

for path, subs in EDITS.items():
    text = open(path).read()
    for old, new in subs:
        assert text.count(old) >= 1, (path, old[:60], text.count(old))
        text = text.replace(old, new)
    open(path, "w").write(text)
print("ok")
