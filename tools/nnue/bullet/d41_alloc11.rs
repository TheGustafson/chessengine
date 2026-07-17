/*
 * D41 Phase-1 allocation-head trainer — v1.1 (design/19 §3, §4.2, §4.4, §7).
 *
 * ── Why v1.1 (the 17:20 diagnosis) ─────────────────────────────────────────
 * v0/v1 proved the search-data signal is learnable (v1 big-v1-sf18: held-out
 * rank corr 0.604, top-1 0.130) but ALSO diagnosed a ~180x train/held-out
 * overfit: a 60M-param trunk trained FROM SCRATCH on ~8k positions with a
 * pure-MSE margin loss memorised. v1.1 is the anti-overfit stack:
 *
 *   1. FROZEN TRUNK (design/19 §4.2).  Load the HalfKA+threats trunk weights
 *      from the d40-uho-720sb raw checkpoint and freeze them (bullet
 *      `train_frozen`: frozen tensors stay in the forward graph, autograd never
 *      materialises their gradients, the optimiser never updates them).  Only
 *      the bilinear allocation head learns, on top of the proven, bit-identical
 *      value representation.  This removes ~63.9M of the ~64M trainable params.
 *      Env `D41_TRUNK_RAW=<raw.bin>` enables it; unset = v0 from-scratch trunk.
 *
 *   2. PAIRWISE RANKING LOSS (RankNet).  MSE on squashed margins over-weights
 *      the many clearly-bad arms a constant already fits, and under-weights the
 *      strategically important near-best regime.  In `D41_LOSS=pairwise` we
 *      instead score arm PAIRS from the same search: a pair (i,j) becomes a
 *      2-logit softmax-cross-entropy against the one-hot "better arm" label
 *      (softmax over [s_i, s_j] with target [1,0] IS the RankNet logistic loss
 *      on s_i - s_j), weighted by clamp(|margin_i-margin_j|/100, 0.1, 3).
 *      `D41_LOSS=margin` keeps v0's squashed-margin MSE (now optionally tier-
 *      weighted).  Both expose the same per-arm "score" output for eval.
 *
 *   3. DUAL-TEACHER MULTI-TIER DATA.  `D41_DATA=dir1,dir2` loads a comma list
 *      of corpora (distinct `source` per dir keeps their searches separate).
 *      With all-tiers corpora (obs-tool export-train --all-tiers) each row
 *      carries its node-budget `tier`; rows/pairs are weighted 64k=1.0,
 *      16k=0.7, 4k=0.4 (D41_TIER_W_* overridable).  Ranking groups are
 *      (position identity `key`, source, tier) — one independent root search.
 *
 *   4. EARLY STOPPING.  The held-out split is position-`key`-hashed with a seed
 *      shared across configs (D41_SPLIT_SEED).  Every superbatch we score a
 *      capped, whole-group held-out subset and track the best-by-held-out-rank-
 *      corr checkpoint separately from the final one; both are reported/saved.
 *
 *   5. MODEST HEAD WEIGHT DECAY.  AdamW `decay = D41_WD` (default 0.01).  With
 *      the trunk frozen this binds ONLY the head tensors (frozen tensors are
 *      never touched by the optimiser), i.e. it is head weight decay by
 *      construction.
 *
 * ── Architecture (unchanged proven gen-0 shape) ────────────────────────────
 * Shared D40 trunk (HalfKAv2_hm i16 tensor `l0_pstw`/`l0_pstb` + FullThreats i8
 * tensor `l0_threatw`, summed per perspective -> crelu -> pairwise_mul ->
 * concat stm/ntm = 768) exactly as d40_features.rs / d41_alloc.rs build it,
 * feeding the rank-`D41_RANK` bilinear from×to head (`s(m|x)=h(x)^T e(m)`,
 * design/19 §4.4).  The 4096 from×to logits are gathered at each observed
 * move's cell with `select`.  Everything mover-relative (trunk from the side-
 * to-move perspective; move squares vertically flipped for black), matching the
 * d40_policy / d41_alloc v0 convention.  The trunk tensor ids and the packed
 * saved-format layout are byte-for-byte v0's, so a frozen-trunk checkpoint's
 * 81,237,504-byte trunk prefix == the d40-uho-720sb trunk: the play-time
 * POLICY_TRUNK is `data/nnue/nets/d40-uho-720sb.cenn` directly (no graft).
 *
 * ── Run (registered as a bullet [[example]] like d40_features/d41_alloc) ────
 *   cd .archon-state/toolchains/bullet
 *   D41_DATA=$REPO/data/obs/train/big-v1-sf18,$REPO/data/obs/train/big-v1-reckless \
 *   D41_TRUNK_RAW=$REPO/data/nnue/experiments/d40-long-v1/ckpt/d40-uho-720sb-final-raw.bin \
 *   D41_LOSS=pairwise D41_SUPERBATCHES=12 D41_BPS=250 D41_BATCH=4096 \
 *   D41_THREADS=2 D41_OUT=$REPO/d41_v11_ckpt D41_ID=d41-v11-pairwise \
 *   cargo run --release --features cuda --example d41_alloc11
 *
 * ── Env knobs ──────────────────────────────────────────────────────────────
 *   D41_DATA            comma list of corpus dirs (default big-v1-sf18)
 *   D41_TRUNK_RAW       raw d40 ModelWeights checkpoint to freeze (unset = v0)
 *   D41_LOSS            margin | pairwise (default margin)
 *   D41_PAIRS_K         max sampled pairs per (key,source,tier) group (default 24)
 *   D41_MARGIN_SCALE    cp scale in sigmoid(margin/scale) target (default 200)
 *   D41_TIER_W_64000/16000/4000  per-tier row weights (default 1.0/0.7/0.4)
 *   D41_RANK            bilinear rank (default 8)
 *   D41_LR              initial Adam LR (default 0.001)
 *   D41_LR_FINAL_FACTOR cosine decays LR to LR*factor over the run (default 0.1)
 *   D41_WD              AdamW decay on the head (default 0.01)
 *   D41_SUPERBATCHES    superbatches (default 12)
 *   D41_BPS             batches per superbatch (default 250)
 *   D41_BATCH           batch size (default 4096)
 *   D41_THREADS         loader threads (default 2 — box rule)
 *   D41_HOLDOUT_PCT     % of POSITIONS (by key) held out (default 20; 0 = none)
 *   D41_SPLIT_SEED      position-split hash seed (default 0xA11C0DE)
 *   D41_SEED            weight-init seed (default 0x00D41A11)
 *   D41_EARLY_EVAL_MAX  cap held-out arms scored per superbatch (default 5000)
 *   D41_EVAL_MAX        cap held-out arms in the FINAL eval (0 = all)
 *   D41_LOG_RATE        batch log stride (default 50)
 *   D41_OUT             checkpoint dir (default d41_alloc11_ckpt)
 *   D41_ID              net id (default d41-alloc-v11)
 */
use bullet_compiler::tensor::TValue;
use bullet_lib::nn::InitSettings;
use bullet_trainer::{
    model::{
        ModelDefinition, ModelEvaluator, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat,
    },
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train_frozen},
};
use chessengine::eval::nnue::sota_features::{
    HALFKA_DIMS, THREAT_DIMS, for_each_halfka, for_each_threat,
};
use chessengine::{Board, Color, Piece, PieceType, Square};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const HALFKA_MAX_ACTIVE: usize = 32;
const THREAT_MAX_ACTIVE: usize = 160;
const SOURCE_FEATURES: usize = 32;
const POLICY_DIM: usize = 4096;
const HL: usize = 768;
const DEFAULT_RANK: usize = 8;
// The three frozen trunk tensors (must match d40_features.rs / the raw ckpt).
const TRUNK_TENSORS: [&str; 3] = ["l0_pstw", "l0_pstb", "l0_threatw"];

// ── One raw corpus row: a single (position, tier, Complete arm) ─────────────
#[derive(Clone, Copy)]
struct ArmRow {
    feats: [i16; SOURCE_FEATURES],
    stm: u8,
    move_idx: i16, // from*64+to, absolute board squares
    margin: i32,   // cp: value - best_arm_value at this tier, <= 0
    key: u64,      // canonical position identity (leakage-safe split + grouping)
    tier: u32,     // node budget of the arm's tier
    source: u8,    // index into the D41_DATA list (keeps teachers' searches apart)
}

/// A group of arms is one independent root search: (position key, data source,
/// node-budget tier). Ranking + pair sampling happen strictly within a group.
type GroupId = (u64, u8, u32);
fn group_of(r: &ArmRow) -> GroupId {
    (r.key, r.source, r.tier)
}

// ── One training/eval example fed to the graph mapper ───────────────────────
// margin mode: one Sample per arm (move_j == move_i, unused).
// pairwise mode: one Sample per sampled pair.
// eval (both modes): one Sample per arm, move_j == move_i, targets dummy.
#[derive(Clone, Copy)]
struct Sample {
    feats: [i16; SOURCE_FEATURES],
    stm: u8,
    move_i: i16,
    move_j: i16,
    alloc_target: f32, // margin: sigmoid(margin_i/scale)
    tgt_i: f32,        // pairwise one-hot: 1.0 if arm i is the better arm
    tgt_j: f32,        // pairwise one-hot: 1.0 if arm j is the better arm
    weight: f32,       // margin: tier weight; pairwise: |Δmargin| weight × tier
    // eval provenance (unused by the graph):
    key: u64,
    source: u8,
    tier: u32,
    margin: i32, // margin of arm i (for held-out rank corr)
}

// ── NumPy v1 .npy column reader (the exact layout export-train writes) ───────
struct NpyFile {
    file: File,
    data_offset: u64,
    row_bytes: usize,
    rows: usize,
}

impl NpyFile {
    fn open(path: impl AsRef<Path>, row_bytes: usize) -> Self {
        let mut file = File::open(path.as_ref())
            .unwrap_or_else(|e| panic!("open {}: {e}", path.as_ref().display()));
        let mut preamble = [0u8; 10];
        file.read_exact(&mut preamble).expect("read npy preamble");
        assert_eq!(&preamble[..6], b"\x93NUMPY", "bad npy magic");
        assert_eq!(preamble[6], 1, "only NumPy v1 files are supported");
        let header_len = u16::from_le_bytes([preamble[8], preamble[9]]) as u64;
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header).expect("read npy header");
        let header = String::from_utf8(header).expect("utf8 npy header");
        let shape = header
            .split("'shape':")
            .nth(1)
            .and_then(|s| s.split('(').nth(1))
            .and_then(|s| s.split(')').next())
            .expect("npy shape");
        let rows: usize = shape
            .split(',')
            .next()
            .and_then(|s| s.trim().parse().ok())
            .expect("npy row count");
        Self { file, data_offset: 10 + header_len, row_bytes, rows }
    }

    fn read_all(&mut self) -> Vec<u8> {
        self.file.seek(SeekFrom::Start(self.data_offset)).expect("seek npy data");
        let mut bytes = vec![0u8; self.rows * self.row_bytes];
        self.file.read_exact(&mut bytes).expect("read npy body");
        bytes
    }
}

fn read_i16(b: &[u8], o: usize) -> i16 {
    i16::from_le_bytes([b[o], b[o + 1]])
}
fn read_i32(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn read_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn read_u64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

/// Load one flat SoA corpus dir into `ArmRow`s, tagging every row with `source`.
fn load_corpus(dir: &Path, source: u8) -> Vec<ArmRow> {
    let mut feats = NpyFile::open(dir.join("feats.npy"), SOURCE_FEATURES * 2);
    let mut stm = NpyFile::open(dir.join("stm.npy"), 1);
    let mut move_idx = NpyFile::open(dir.join("move_idx.npy"), 2);
    let mut margin = NpyFile::open(dir.join("margin.npy"), 4);
    let mut key = NpyFile::open(dir.join("key.npy"), 8);
    let mut tier = NpyFile::open(dir.join("tier.npy"), 4);
    let n = feats.rows;
    for (name, r) in [("stm", stm.rows), ("move_idx", move_idx.rows), ("margin", margin.rows), ("key", key.rows), ("tier", tier.rows)] {
        assert_eq!(r, n, "{name} row count mismatch");
    }
    let feats_b = feats.read_all();
    let stm_b = stm.read_all();
    let move_b = move_idx.read_all();
    let margin_b = margin.read_all();
    let key_b = key.read_all();
    let tier_b = tier.read_all();

    (0..n)
        .map(|r| {
            let mut f = [-1i16; SOURCE_FEATURES];
            for (i, slot) in f.iter_mut().enumerate() {
                *slot = read_i16(&feats_b, (r * SOURCE_FEATURES + i) * 2);
            }
            ArmRow {
                feats: f,
                stm: stm_b[r],
                move_idx: read_i16(&move_b, r * 2),
                margin: read_i32(&margin_b, r * 4),
                key: read_u64(&key_b, r * 8),
                tier: read_u32(&tier_b, r * 4),
                source,
            }
        })
        .collect()
}

// ── Frozen-trunk loader: parse the raw d40 ModelWeights checkpoint ──────────
// The raw format (bullet ModelWeights::write_into) is a flat sequence of
// records: [id ascii + '\n'][usize len LE][len × f32 LE].
fn load_trunk_into(weights: &mut ModelWeights, raw_path: &Path) {
    let bytes = std::fs::read(raw_path)
        .unwrap_or_else(|e| panic!("read trunk {}: {e}", raw_path.display()));
    let mut off = 0usize;
    let mut loaded = 0usize;
    while off < bytes.len() {
        let nl = off + bytes[off..].iter().position(|&c| c == b'\n').expect("trunk record id newline");
        let id = std::str::from_utf8(&bytes[off..nl]).expect("trunk id ascii").to_string();
        off = nl + 1;
        let len = usize::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        let vals: Vec<f32> = bytes[off..off + len * 4]
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        off += len * 4;
        if TRUNK_TENSORS.contains(&id.as_str()) {
            assert!(
                weights.set(&id, TValue::F32(vals)),
                "trunk tensor {id} shape/dtype mismatch vs the graph"
            );
            loaded += 1;
        }
    }
    assert_eq!(loaded, TRUNK_TENSORS.len(), "did not find all trunk tensors in {}", raw_path.display());
    println!("frozen trunk: loaded {loaded} tensors from {}", raw_path.display());
}

// ── In-memory row-shuffling reader over the training Samples ────────────────
#[derive(Clone)]
struct SampleReader {
    data: Arc<Vec<Sample>>,
    chunk_rows: usize,
    seed: u64,
}

impl DataReader<Sample> for SampleReader {
    fn read_chunks<F: FnMut(&[Sample]) -> bool>(&self, skip_count: usize, mut f: F) {
        let n = self.data.len();
        if n == 0 {
            return;
        }
        let mut produced = 0usize;
        let mut epoch = 0u64;
        loop {
            let mut idx: Vec<u32> = (0..n as u32).collect();
            let mut s = self.seed ^ splitmix64(epoch);
            for i in (1..n).rev() {
                s = splitmix64(s);
                let j = (s % (i as u64 + 1)) as usize;
                idx.swap(i, j);
            }
            let mut pos = 0usize;
            while pos < n {
                let end = (pos + self.chunk_rows).min(n);
                let chunk: Vec<Sample> = idx[pos..end].iter().map(|&k| self.data[k as usize]).collect();
                pos = end;
                if produced + chunk.len() <= skip_count {
                    produced += chunk.len();
                    continue;
                }
                let start = skip_count.saturating_sub(produced);
                produced += chunk.len();
                if f(&chunk[start..]) {
                    return;
                }
            }
            epoch += 1;
        }
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// A position (by canonical key) is held out iff hash(key) falls in first pct%.
fn is_eval(key: u64, pct: u32, split_seed: u64) -> bool {
    pct > 0 && splitmix64(split_seed ^ key) % 100 < u64::from(pct)
}

fn board_from_features(features: &[i16; SOURCE_FEATURES]) -> Board {
    let mut board = Board::empty();
    for &feature in features {
        if feature < 0 {
            break;
        }
        let feature = feature as usize;
        assert!(feature < 768, "source feature out of range: {feature}");
        let piece_slot = feature / 64;
        let color = Color::ALL[piece_slot / 6];
        let piece_type = PieceType::ALL[piece_slot % 6];
        board.put_piece(Piece::new(color, piece_type), Square::new((feature % 64) as u8));
    }
    board
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Mover-relative bilinear cell (to*64+from), vertically flipped for black.
fn sel_index(move_idx: i16, stm: u8) -> i32 {
    let raw = move_idx as usize;
    let (from, to) = (raw / 64, raw % 64);
    let (from, to) = if stm == 0 { (from, to) } else { (from ^ 56, to ^ 56) };
    (to * 64 + from) as i32
}

/// Fill the four mover-relative sparse trunk inputs for one position.
fn fill_trunk(
    feats: &[i16; SOURCE_FEATURES],
    stm: u8,
    hk_stm: &mut [i32],
    hk_ntm: &mut [i32],
    th_stm: &mut [i32],
    th_ntm: &mut [i32],
) {
    let board = board_from_features(feats);
    let stm_color = if stm == 0 { Color::White } else { Color::Black };
    for perspective in Color::ALL {
        let is_stm = perspective == stm_color;
        let hk = if is_stm { &mut *hk_stm } else { &mut *hk_ntm };
        let mut hc = 0usize;
        for_each_halfka(&board, perspective, |idx| {
            hk[hc] = idx as i32;
            hc += 1;
        });
        hk[hc..].fill(-1);
        let th = if is_stm { &mut *th_stm } else { &mut *th_ntm };
        let mut tc = 0usize;
        for_each_threat(&board, perspective, |idx| {
            assert!(tc < THREAT_MAX_ACTIVE, "threat sparse capacity exceeded");
            th[tc] = idx as i32;
            tc += 1;
        });
        th[tc..].fill(-1);
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn tier_weight(tier: u32, w64: f32, w16: f32, w4: f32) -> f32 {
    match tier {
        64000 => w64,
        16000 => w16,
        4000 => w4,
        _ => 1.0,
    }
}

fn saved_format() -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("l0_pstw").round().quantise::<i16>(255),
        SavedFormat::id("l0_pstb").round().quantise::<i16>(255),
        SavedFormat::id("l0_threatw").round().quantise::<i8>(64),
        SavedFormat::id("pol_fromw").transpose(),
        SavedFormat::id("pol_fromb"),
        SavedFormat::id("pol_tow").transpose(),
        SavedFormat::id("pol_tob"),
    ]
}

// ── Held-out ranking metrics over per-arm (group, score, margin) triples ────
struct RankMetrics {
    groups: usize,
    top1: f32,
    rand_top1: f32,
    rank_corr: f32,
}

fn rank_metrics(mut rows: Vec<(GroupId, f32, i32)>) -> RankMetrics {
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let mut top1_hits = 0usize;
    let mut ranked = 0usize;
    let mut rand_top1_sum = 0.0f64;
    let mut pearson_sum = 0.0f64;
    let mut pearson_n = 0usize;
    let mut i = 0usize;
    while i < rows.len() {
        let g = rows[i].0;
        let mut j = i;
        while j < rows.len() && rows[j].0 == g {
            j += 1;
        }
        let group = &rows[i..j];
        i = j;
        if group.len() < 2 {
            continue;
        }
        ranked += 1;
        rand_top1_sum += 1.0 / group.len() as f64;
        let pred_best = group.iter().enumerate().max_by(|a, b| a.1.1.total_cmp(&b.1.1)).unwrap().0;
        let true_best_margin = group.iter().map(|r| r.2).max().unwrap();
        if group[pred_best].2 == true_best_margin {
            top1_hits += 1;
        }
        let n = group.len() as f64;
        let mp = group.iter().map(|r| f64::from(r.1)).sum::<f64>() / n;
        let mm = group.iter().map(|r| f64::from(r.2)).sum::<f64>() / n;
        let (mut cov, mut vp, mut vm) = (0.0, 0.0, 0.0);
        for r in group {
            let dp = f64::from(r.1) - mp;
            let dm = f64::from(r.2) - mm;
            cov += dp * dm;
            vp += dp * dp;
            vm += dm * dm;
        }
        if vp > 0.0 && vm > 0.0 {
            pearson_sum += cov / (vp.sqrt() * vm.sqrt());
            pearson_n += 1;
        }
    }
    RankMetrics {
        groups: ranked,
        top1: if ranked > 0 { top1_hits as f32 / ranked as f32 } else { 0.0 },
        rand_top1: if ranked > 0 { (rand_top1_sum / ranked as f64) as f32 } else { 0.0 },
        rank_corr: if pearson_n > 0 { (pearson_sum / pearson_n as f64) as f32 } else { 0.0 },
    }
}

fn main() {
    // ── Config ───────────────────────────────────────────────────────────
    let data_arg =
        std::env::var("D41_DATA").unwrap_or_else(|_| "data/obs/train/big-v1-sf18".to_string());
    let data_dirs: Vec<PathBuf> = data_arg.split(',').map(|s| PathBuf::from(s.trim())).collect();
    let trunk_raw = std::env::var("D41_TRUNK_RAW").ok().map(PathBuf::from);
    let loss_mode = std::env::var("D41_LOSS").unwrap_or_else(|_| "margin".to_string());
    let pairwise = match loss_mode.as_str() {
        "pairwise" => true,
        "margin" => false,
        other => panic!("D41_LOSS must be margin|pairwise, got {other}"),
    };
    let pairs_k = env_usize("D41_PAIRS_K", 24);
    let margin_scale = env_f32("D41_MARGIN_SCALE", 200.0);
    assert!(margin_scale > 0.0 && margin_scale.is_finite());
    let w64 = env_f32("D41_TIER_W_64000", 1.0);
    let w16 = env_f32("D41_TIER_W_16000", 0.7);
    let w4 = env_f32("D41_TIER_W_4000", 0.4);
    let rank = env_usize("D41_RANK", DEFAULT_RANK);
    assert!(rank > 0 && rank <= 64);
    let learning_rate = env_f32("D41_LR", 0.001);
    let final_lr_factor = env_f32("D41_LR_FINAL_FACTOR", 0.1);
    assert!(final_lr_factor > 0.0 && final_lr_factor <= 1.0);
    let weight_decay = env_f32("D41_WD", 0.01);
    let batch_size = env_usize("D41_BATCH", 4096);
    let batches_per_superbatch = env_usize("D41_BPS", 250);
    let end_superbatch = env_usize("D41_SUPERBATCHES", 12);
    let threads = env_usize("D41_THREADS", 2) as u8;
    let holdout_pct = env_usize("D41_HOLDOUT_PCT", 20) as u32;
    assert!(holdout_pct < 100);
    let split_seed = env_usize("D41_SPLIT_SEED", 0x0A11_C0DE) as u64;
    let seed = env_usize("D41_SEED", 0x00D4_1A11) as u64;
    let early_eval_max = env_usize("D41_EARLY_EVAL_MAX", 5000);
    let eval_max = env_usize("D41_EVAL_MAX", 0);
    // Eval-only tier filter: train on all tiers but score held-out ranking on
    // one tier (e.g. 64000) so an all-tiers run stays comparable to a deepest-
    // only run. 0 = eval on whatever tiers the held-out split contains.
    let eval_tier = env_usize("D41_EVAL_TIER", 0) as u32;
    let log_rate = env_usize("D41_LOG_RATE", 50).max(1);
    let out_dir = std::env::var("D41_OUT").unwrap_or_else(|_| "d41_alloc11_ckpt".to_string());
    let net_id = std::env::var("D41_ID").unwrap_or_else(|_| "d41-alloc-v11".to_string());
    std::fs::create_dir_all(&out_dir).expect("create D41_OUT");

    println!("=== D41 Phase-1 allocation-head trainer (v1.1) ===");
    println!(
        "loss={loss_mode} trunk_frozen={} data={data_arg} rank={rank} wd={weight_decay} tier_w=[{w64},{w16},{w4}]",
        trunk_raw.is_some()
    );

    // ── Load + concatenate corpora (distinct source per dir) ─────────────
    let mut all: Vec<ArmRow> = Vec::new();
    for (s, dir) in data_dirs.iter().enumerate() {
        let rows = load_corpus(dir, s as u8);
        println!("  source {s}: {} rows from {}", rows.len(), dir.display());
        all.extend(rows);
    }
    assert!(!all.is_empty(), "empty corpus");

    // ── Leakage-safe split by canonical key ──────────────────────────────
    let (train_arms, eval_arms): (Vec<ArmRow>, Vec<ArmRow>) =
        all.iter().copied().partition(|r| !is_eval(r.key, holdout_pct, split_seed));
    // Optional eval-only tier filter (training keeps all tiers).
    let eval_arms: Vec<ArmRow> = if eval_tier != 0 {
        eval_arms.into_iter().filter(|r| r.tier == eval_tier).collect()
    } else {
        eval_arms
    };
    let tkeys: BTreeSet<u64> = train_arms.iter().map(|r| r.key).collect();
    let ekeys: BTreeSet<u64> = eval_arms.iter().map(|r| r.key).collect();
    println!(
        "split: {} train arms / {} keys, {} eval arms / {} keys",
        train_arms.len(),
        tkeys.len(),
        eval_arms.len(),
        ekeys.len(),
    );
    assert!(!train_arms.is_empty(), "empty train split");
    drop(all); // free the concatenated copy — train/eval own their rows now

    // ── Build training Samples ───────────────────────────────────────────
    let target = |m: i32| sigmoid(m as f32 / margin_scale);
    let train_samples: Vec<Sample> = if pairwise {
        build_pairs(&train_arms, pairs_k, w64, w16, w4, seed)
    } else {
        train_arms
            .iter()
            .map(|r| Sample {
                feats: r.feats,
                stm: r.stm,
                move_i: r.move_idx,
                move_j: r.move_idx,
                alloc_target: target(r.margin),
                tgt_i: 1.0,
                tgt_j: 0.0,
                weight: tier_weight(r.tier, w64, w16, w4),
                key: r.key,
                source: r.source,
                tier: r.tier,
                margin: r.margin,
            })
            .collect()
    };
    println!("training samples: {} ({})", train_samples.len(), if pairwise { "pairs" } else { "arms" });

    // Margin-mode held-out MSE baselines (not applicable to pairwise).
    if !pairwise && !eval_arms.is_empty() {
        let train_mean = train_arms.iter().map(|r| target(r.margin)).sum::<f32>() / train_arms.len() as f32;
        let et: Vec<f32> = eval_arms.iter().map(|r| target(r.margin)).collect();
        let mse = |ts: &[f32], p: f32| ts.iter().map(|&t| (p - t).powi(2)).sum::<f32>() / ts.len() as f32;
        println!(
            "BASELINE(eval) zero-margin(0.5) MSE {:.6}  train-mean({train_mean:.4}) MSE {:.6}",
            mse(&et, 0.5),
            mse(&et, train_mean),
        );
    }

    // Deterministic capped held-out subset for per-superbatch early stopping
    // (whole groups, sorted — identical across configs given the same split).
    let mut eval_sorted = eval_arms.clone();
    eval_sorted.sort_by(|a, b| group_of(a).cmp(&group_of(b)));
    let early_subset: Vec<ArmRow> = if early_eval_max == 0 || eval_sorted.len() <= early_eval_max {
        eval_sorted.clone()
    } else {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < eval_sorted.len() && out.len() < early_eval_max {
            let g = group_of(&eval_sorted[i]);
            let mut j = i;
            while j < eval_sorted.len() && group_of(&eval_sorted[j]) == g {
                j += 1;
            }
            out.extend_from_slice(&eval_sorted[i..j]);
            i = j;
        }
        out
    };
    let arm_to_sample = |r: &ArmRow| Sample {
        feats: r.feats,
        stm: r.stm,
        move_i: r.move_idx,
        move_j: r.move_idx,
        alloc_target: target(r.margin),
        tgt_i: 1.0,
        tgt_j: 0.0,
        weight: 1.0,
        key: r.key,
        source: r.source,
        tier: r.tier,
        margin: r.margin,
    };
    let early_samples: Vec<Sample> = early_subset.iter().map(arm_to_sample).collect();
    let final_eval_samples: Vec<Sample> = {
        let cap = if eval_max == 0 { eval_sorted.len() } else { eval_max.min(eval_sorted.len()) };
        eval_sorted[..cap].iter().map(arm_to_sample).collect()
    };

    // ── Inputs + mapper + graph (per loss mode) ──────────────────────────
    let device = DefaultDevice::new(0).expect("device");
    let (mapper, defn) = if pairwise { build_pairwise(rank) } else { build_margin(rank) };

    let mut weights = ModelWeights::new(&defn, seed);
    let mut frozen: BTreeSet<String> = BTreeSet::new();
    if let Some(raw) = &trunk_raw {
        load_trunk_into(&mut weights, raw);
        frozen = TRUNK_TENSORS.iter().map(|s| s.to_string()).collect();
    }
    let adam = AdamWParams { decay: weight_decay, ..AdamWParams::default() };
    let mut optimiser = AdamW::new(defn, weights, device.clone(), adam).expect("optimiser");

    let reader = SampleReader {
        data: Arc::new(train_samples),
        chunk_rows: batch_size.clamp(256, 1024),
        seed,
    };
    let loader = ReadMapLoader::new(reader, mapper.clone(), threads);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size, batches_per_superbatch, start_superbatch: 1, end_superbatch },
        log_rate,
        lr_schedule: Box::new(move |step: Step| {
            let final_lr = learning_rate * final_lr_factor;
            if end_superbatch <= 1 || final_lr_factor == 1.0 {
                return learning_rate;
            }
            let progress = (step.superbatch() as f32 - 1.0) / (end_superbatch as f32 - 1.0);
            final_lr + 0.5 * (learning_rate - final_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }),
    };
    println!(
        "training: {end_superbatch} superbatch(es) x {batches_per_superbatch} x {batch_size}, lr {learning_rate}->{}, frozen={:?}",
        learning_rate * final_lr_factor,
        frozen,
    );

    // ── Early-stopping evaluator (batch-1 forward; capped subset) ─────────
    // Scoring is inlined at both call sites (the superbatch callback and the
    // final eval) rather than shared through a closure, so the concrete Gpu
    // type never has to be named.
    let mut evaluator = ModelEvaluator::new(optimiser.definition(), device.clone()).expect("evaluator");
    let eval_device = device.clone();
    let eval_mapper = mapper.clone();

    let best_quant_path = format!("{out_dir}/{net_id}-best-quant.bin");
    let mut best_corr = f32::NEG_INFINITY;
    let mut best_sb = 0usize;

    train_frozen(
        &mut optimiser,
        schedule,
        loader,
        &frozen,
        |_, step, loss| {
            if step.batch() % log_rate == 0 {
                println!("  sb {} batch {} loss {loss:.6}", step.superbatch(), step.batch());
            }
        },
        |optimiser, step| {
            let sb = step.superbatch();
            if early_samples.is_empty() {
                return;
            }
            evaluator
                .load_weights(&optimiser.cpu_weights().expect("weights for eval"))
                .expect("load eval weights");
            let mut triples: Vec<(GroupId, f32, i32)> = Vec::with_capacity(early_samples.len());
            for s in &early_samples {
                let host = eval_mapper.map(std::slice::from_ref(s), Step::default(), 1);
                let input = host.to_device(&eval_device).expect("eval inputs to device");
                let out = evaluator.evaluate(&input).expect("eval forward");
                let TValue::F32(v) = out.get("score").expect("score output").to_host().expect("score to host")
                else {
                    panic!("score output must be f32");
                };
                triples.push(((s.key, s.source, s.tier), v[0], s.margin));
            }
            let m = rank_metrics(triples);
            println!(
                "  [early sb {sb}] held-out rank_corr {:.4} top-1 {:.3} (rand {:.3}) over {} groups",
                m.rank_corr, m.top1, m.rand_top1, m.groups,
            );
            if m.rank_corr > best_corr {
                best_corr = m.rank_corr;
                best_sb = sb;
                let bytes = optimiser
                    .cpu_weights()
                    .expect("sync best weights")
                    .to_quantised_buffer(&saved_format(), true)
                    .expect("export best checkpoint");
                std::fs::write(&best_quant_path, &bytes).expect("write best checkpoint");
                println!("    new best rank_corr {:.4} -> {best_quant_path} ({} bytes)", best_corr, bytes.len());
            }
        },
    )
    .expect("alloc training");

    // ── Final checkpoints ────────────────────────────────────────────────
    let bytes = optimiser
        .cpu_weights()
        .expect("sync weights")
        .to_quantised_buffer(&saved_format(), true)
        .expect("export checkpoint");
    let quant_path = format!("{out_dir}/{net_id}-final-quant.bin");
    std::fs::write(&quant_path, &bytes).expect("write checkpoint");
    println!("final checkpoint: {quant_path} ({} bytes)", bytes.len());
    {
        let raw_path = format!("{out_dir}/{net_id}-final-raw.bin");
        let mut raw = File::create(&raw_path).expect("create raw checkpoint");
        optimiser.cpu_weights().expect("sync weights").write_into(&mut raw).expect("write raw");
        println!("final raw: {raw_path}");
    }

    // ── FINAL held-out eval (full/capped) on the FINAL weights ───────────
    if final_eval_samples.is_empty() {
        println!("no held-out split; skipping final eval");
        return;
    }
    evaluator
        .load_weights(&optimiser.cpu_weights().expect("weights for final eval"))
        .expect("load final eval weights");
    let mut triples: Vec<(GroupId, f32, i32)> = Vec::with_capacity(final_eval_samples.len());
    for s in &final_eval_samples {
        let host = eval_mapper.map(std::slice::from_ref(s), Step::default(), 1);
        let input = host.to_device(&eval_device).expect("eval inputs to device");
        let out = evaluator.evaluate(&input).expect("eval forward");
        let TValue::F32(v) = out.get("score").expect("score output").to_host().expect("score to host") else {
            panic!("score output must be f32");
        };
        triples.push(((s.key, s.source, s.tier), v[0], s.margin));
    }
    // margin-mode MSE (score is a logit; pred = sigmoid(logit)).
    if !pairwise {
        let mut sse = 0.0f64;
        for (_, logit, margin) in &triples {
            let pred = sigmoid(*logit);
            let t = target(*margin);
            sse += f64::from((pred - t).powi(2));
        }
        println!("=== final held-out MSE {:.6} over {} arms ===", (sse / triples.len() as f64) as f32, triples.len());
    }
    let m = rank_metrics(triples);
    println!("=== FINAL held-out ranking ({} groups) ===", m.groups);
    println!("  rank_corr {:.4}  top-1 {:.3} (rand {:.3})", m.rank_corr, m.top1, m.rand_top1);
    println!("=== BEST-by-held-out checkpoint: sb {best_sb}, rank_corr {:.4} -> {best_quant_path} ===", best_corr);
    println!("=== d41_alloc11 complete (loss={loss_mode}, frozen={}) ===", trunk_raw.is_some());
}

// ── Pair sampling: within each (key,source,tier) group, sample up to K pairs ─
fn build_pairs(arms: &[ArmRow], k: usize, w64: f32, w16: f32, w4: f32, seed: u64) -> Vec<Sample> {
    let mut sorted: Vec<ArmRow> = arms.to_vec();
    sorted.sort_by(|a, b| group_of(a).cmp(&group_of(b)));
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < sorted.len() {
        let g = group_of(&sorted[i]);
        let mut j = i;
        while j < sorted.len() && group_of(&sorted[j]) == g {
            j += 1;
        }
        let group = &sorted[i..j];
        i = j;
        let m = group.len();
        if m < 2 {
            continue;
        }
        let tw = tier_weight(g.2, w64, w16, w4);
        let mut rng = splitmix64(seed ^ g.0 ^ (u64::from(g.1) << 40) ^ (u64::from(g.2) << 8));
        // For small groups take all unordered pairs; else sample K random pairs.
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        if m * (m - 1) / 2 <= k {
            for a in 0..m {
                for b in (a + 1)..m {
                    pairs.push((a, b));
                }
            }
        } else {
            let mut tries = 0usize;
            while pairs.len() < k && tries < k * 8 {
                rng = splitmix64(rng);
                let a = (rng % m as u64) as usize;
                rng = splitmix64(rng);
                let b = (rng % m as u64) as usize;
                tries += 1;
                if a != b {
                    pairs.push((a.min(b), a.max(b)));
                }
            }
        }
        for (a, b) in pairs {
            let (ra, rb) = (&group[a], &group[b]);
            if ra.margin == rb.margin {
                continue; // no ordering signal
            }
            let dm = (ra.margin - rb.margin).abs() as f32;
            let weight = (dm / 100.0).clamp(0.1, 3.0) * tw;
            let i_better = ra.margin > rb.margin;
            out.push(Sample {
                feats: ra.feats,
                stm: ra.stm,
                move_i: ra.move_idx,
                move_j: rb.move_idx,
                alloc_target: 0.0,
                tgt_i: if i_better { 1.0 } else { 0.0 },
                tgt_j: if i_better { 0.0 } else { 1.0 },
                weight,
                key: ra.key,
                source: ra.source,
                tier: ra.tier,
                margin: ra.margin,
            });
        }
    }
    out
}

type BuiltGraph = (ModelInputsMapper<Sample>, ModelDefinition);

// The shared D40 trunk + rank-`rank` bilinear from×to head. `move_i`/`move_j`
// select the observed cells; the graph exposes output "score" = the arm-i logit.
// The graph is inlined per mode (v0 style) because `ModelInputs<T>`'s type
// differs by input composition and cannot be named uniformly.
fn build_margin(rank: usize) -> BuiltGraph {
    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("threat_stm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("move_i", (POLICY_DIM, 1), 1)
        .add_dense("margin_target", (1, 1))
        .add_dense("tier_weight", (1, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |d: &Sample, _step, ((((((hk_stm, hk_ntm), th_stm), th_ntm), move_i), margin_target), tier_weight)| {
            fill_trunk(&d.feats, d.stm, hk_stm, hk_ntm, th_stm, th_ntm);
            move_i[0] = sel_index(d.move_i, d.stm);
            margin_target[0] = d.alloc_target;
            tier_weight[0] = d.weight;
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        move |builder, ((((((hk_stm, hk_ntm), th_stm), th_ntm), move_i), margin_target), tier_weight)| {
            let pst = builder.new_affine("l0_pst", HALFKA_DIMS, HL);
            pst.init_with_effective_input_size(32);
            let threat = builder.new_weights("l0_threatw", (HL, THREAT_DIMS), InitSettings::Zeroed);
            let stm_acc = pst.forward(hk_stm) + threat.matmul(th_stm);
            let ntm_acc = pst.forward(hk_ntm) + threat.matmul(th_ntm);
            let trunk = stm_acc.crelu().pairwise_mul().concat(ntm_acc.crelu().pairwise_mul());
            let from = builder.new_affine("pol_from", HL, 64 * rank).forward(trunk).reshape((64, rank));
            let to = builder.new_affine("pol_to", HL, 64 * rank).forward(trunk).reshape((rank, 64));
            let logits = from.matmul(to).reshape((POLICY_DIM, 1));
            let gathered = logits.select(move_i);
            let pred = gathered.sigmoid();
            let loss = (pred.squared_error(margin_target) * tier_weight).reduce_sum_batch();
            (Some(loss), vec![("score".to_string(), gathered)])
        },
    );
    (mapper, defn)
}

fn build_pairwise(rank: usize) -> BuiltGraph {
    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("threat_stm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("move_i", (POLICY_DIM, 1), 1)
        .add_sparse("move_j", (POLICY_DIM, 1), 1)
        .add_dense("pair_target", (2, 1))
        .add_dense("pair_weight", (1, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |d: &Sample, _step, (((((((hk_stm, hk_ntm), th_stm), th_ntm), move_i), move_j), pair_target), pair_weight)| {
            fill_trunk(&d.feats, d.stm, hk_stm, hk_ntm, th_stm, th_ntm);
            move_i[0] = sel_index(d.move_i, d.stm);
            move_j[0] = sel_index(d.move_j, d.stm);
            pair_target[0] = d.tgt_i;
            pair_target[1] = d.tgt_j;
            pair_weight[0] = d.weight;
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        move |builder, (((((((hk_stm, hk_ntm), th_stm), th_ntm), move_i), move_j), pair_target), pair_weight)| {
            let pst = builder.new_affine("l0_pst", HALFKA_DIMS, HL);
            pst.init_with_effective_input_size(32);
            let threat = builder.new_weights("l0_threatw", (HL, THREAT_DIMS), InitSettings::Zeroed);
            let stm_acc = pst.forward(hk_stm) + threat.matmul(th_stm);
            let ntm_acc = pst.forward(hk_ntm) + threat.matmul(th_ntm);
            let trunk = stm_acc.crelu().pairwise_mul().concat(ntm_acc.crelu().pairwise_mul());
            let from = builder.new_affine("pol_from", HL, 64 * rank).forward(trunk).reshape((64, rank));
            let to = builder.new_affine("pol_to", HL, 64 * rank).forward(trunk).reshape((rank, 64));
            let logits = from.matmul(to).reshape((POLICY_DIM, 1));
            let li = logits.select(move_i);
            let lj = logits.select(move_j);
            let pair = li.concat(lj); // (2,1)
            let ce = pair.softmax_crossentropy_loss(pair_target).reduce_sum_rows(); // (1,1)/row
            let loss = (ce * pair_weight).reduce_sum_batch();
            (Some(loss), vec![("score".to_string(), li)])
        },
    );
    (mapper, defn)
}
