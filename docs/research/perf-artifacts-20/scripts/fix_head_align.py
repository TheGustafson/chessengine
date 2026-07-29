"""perf study 20 arm 5 — align the SotaHead weight tables.

`l1_dot` reads its int8 weight block as 64-byte `zmm` operands at
`group * 64`, so like every other kernel its alignment is decided entirely by
the table base — and `l1_weights` was still a `Box<[i8]>` at 16 mod 64. The
`ls_misal_loads.ma64` profile put 20.5% of all cacheline-crossing loads inside
`forward_into`, and the offset histogram put 54% of `forward_into` inside that
group loop.

This pass was deliberately NOT taken in arm 2: retyping any `SotaHead` field
breaks a struct literal in `sota_features.rs`, which perf study 19 owned while
this study was running. It is taken now that study 19 has finished, and it is
kept in its own commit so it can be dropped whole if it collides.
"""

EDITS = {
    "src/eval/nnue/sota_head.rs": [
        ("    pub l1_weights: Box<[i8]>,", "    pub l1_weights: Aligned<i8>,"),
        ("    pub l2_weights: Box<[f32]>,", "    pub l2_weights: Aligned<f32>,"),
    ],
    "src/eval/nnue/sota_loader.rs": [
        (
            "    let l1_weights = read_i8s(payload, &mut cur, BUCKETS * L2 * L1);",
            "    let l1_weights = read_i8s_aligned(payload, &mut cur, BUCKETS * L2 * L1);",
        ),
        (
            "    let l2_weights = read_f32s(payload, &mut cur, BUCKETS * L2 * L3);",
            "    let l2_weights = read_f32s_aligned(payload, &mut cur, BUCKETS * L2 * L3);",
        ),
        (
            "        let l1_weights: Box<[i8]> = (0..BUCKETS * L2 * L1)",
            "        let l1_weights: Aligned<i8> = (0..BUCKETS * L2 * L1)",
        ),
        (
            "        let l2_weights: Box<[f32]> = (0..BUCKETS * L2 * L3)",
            "        let l2_weights: Aligned<f32> = (0..BUCKETS * L2 * L3)",
        ),
    ],
    # The struct-literal test that blocked this in arm 2.
    "src/eval/nnue/sota_features.rs": [
        (
            "            l1_weights: vec![1i8; BUCKETS * L2 * L1].into_boxed_slice(),",
            "            l1_weights: vec![1i8; BUCKETS * L2 * L1].into(),",
        ),
        (
            "            l2_weights: vec![0.01f32; BUCKETS * L2 * L3].into_boxed_slice(),",
            "            l2_weights: vec![0.01f32; BUCKETS * L2 * L3].into(),",
        ),
    ],
    "tests/cenn_v2_smoke.rs": [
        (
            "    let l1_weights: Box<[i8]> = (0..BUCKETS * L2 * L1)",
            "    let l1_weights: Aligned<i8> = (0..BUCKETS * L2 * L1)",
        ),
        (
            "    let l2_weights: Box<[f32]> = (0..BUCKETS * L2 * L3)",
            "    let l2_weights: Aligned<f32> = (0..BUCKETS * L2 * L3)",
        ),
    ],
    "tests/eval_file_reload.rs": [
        (
            "            l1_weights: vec![0i8; BUCKETS * L2 * L1].into_boxed_slice(),",
            "            l1_weights: vec![0i8; BUCKETS * L2 * L1].into(),",
        ),
        (
            "            l2_weights: vec![0.0; BUCKETS * L2 * L3].into_boxed_slice(),",
            "            l2_weights: vec![0.0; BUCKETS * L2 * L3].into(),",
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
