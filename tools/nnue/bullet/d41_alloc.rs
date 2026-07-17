/*
 * D41 Phase-1 allocation-head trainer (design/19 §3, §4.4, §7).
 *
 * ── What this trains ──────────────────────────────────────────────────────
 * The design/19 Phase-1 legal-move consequence head, v0: for a position `x`
 * and a legal move `m`, predict the move's VALUE-MARGIN vs the best legal move
 * — the per-arm signal the M4 audit told us to use instead of hard best-move
 * classes (median top-arm gap is only 13-16cp, so argmax labels are soft, but
 * the per-arm SCORES are reproducible). The target lives in a squashed sigmoid
 * space consistent with the doc's value framing:
 *
 *     target(m) = sigmoid(margin_cp(m) / MARGIN_SCALE)          margin <= 0
 *
 * so the best arm (margin 0) targets 0.5 and progressively worse arms decay
 * toward 0. A model that predicts "every move is the best move" (margin 0,
 * i.e. a constant 0.5) is the trivial `predict-zero-margin` baseline this smoke
 * must beat.
 *
 * ── Architecture (v0 = the PROVEN gen-0 shape, not a new invention) ────────
 * Shared D40 trunk (HalfKAv2_hm i16 tensor + FullThreats i8 tensor, summed per
 * perspective → crelu → pairwise_mul → concat stm/ntm = 768) exactly as
 * tools/nnue/bullet/d40_features.rs / d40_policy.rs build it, feeding the same
 * rank-8 bilinear from×to head as d40_policy's `bilinear` head
 * (`s(m|x) = h(x)^T e(m)`, design/19 §4.4). The 4096 from×to logits are gathered
 * at the observed move's cell with `select`, sigmoid'd, and MSE-regressed onto
 * the squashed margin. Everything is mover-relative (trunk from the side-to-
 * move perspective; move squares vertically flipped for black), matching the
 * d40_policy convention. v0 trains the trunk from scratch jointly with the head
 * — freezing a protected value trunk (design/19 §4.2) is a v1 concern; v0's job
 * is to prove the data path and that the corpus signal is learnable.
 *
 * ── Data ──────────────────────────────────────────────────────────────────
 * Streams the flat SoA corpus written by `obs-tool export-train`
 * (schema archon.obs-train.rootmove-value-v1): one row per (position, Complete
 * arm). Columns used: feats[N,32] (absolute (colour*6+pt)*64+sq, -1 padded),
 * stm, move_idx (from*64+to), margin, position_id. The corpus is small (~34k
 * rows / 1000 positions for sample-v1) so it is loaded fully into RAM and
 * row-shuffled each epoch. A held-out check splits by position_id (all arms of
 * a position stay on one side — leakage-safe) and reports eval MSE vs the same
 * baselines after training.
 *
 * ── Run ────────────────────────────────────────────────────────────────────
 * Copied into the bullet checkout like the other examples (train_run.py does
 * this; or register a [[example]] in crates/bullet_lib/Cargo.toml pointing at
 * ../../../../../tools/nnue/bullet/d41_alloc.rs, chessengine dep already carries
 * `features=["sota"]`), then:
 *
 *   cd .archon-state/toolchains/bullet
 *   D41_DATA=$REPO/data/obs/train/sample-v1-reckless \
 *   D41_SUPERBATCHES=1 D41_BPS=300 D41_BATCH=4096 \
 *   cargo run --release --features cuda --example d41_alloc
 *
 * ── Env knobs ──────────────────────────────────────────────────────────────
 *   D41_DATA            corpus dir (default data/obs/train/sample-v1-reckless)
 *   D41_MARGIN_SCALE    cp scale in sigmoid(margin/scale) target (default 200)
 *   D41_RANK            bilinear rank (default 8, the proven gen-0 rank)
 *   D41_LR              initial Adam LR (default 0.001)
 *   D41_LR_FINAL_FACTOR cosine decays LR to LR*factor over the run (default 1.0)
 *   D41_SUPERBATCHES    superbatches (default 1)
 *   D41_BPS             batches per superbatch (default 300)
 *   D41_BATCH           batch size (default 4096)
 *   D41_THREADS         loader threads (default 6)
 *   D41_HOLDOUT_PCT     % of POSITIONS held out for eval (default 20; 0 = none)
 *   D41_SPLIT_SEED      position-split hash seed (default 0xA11C_0DE)
 *   D41_SEED            weight-init seed (default 0x00D4_1A11)
 *   D41_EVAL_MAX        cap held-out rows scored at batch-1 (0 = all)
 *   D41_LOG_RATE        batch log stride (default 50)
 *   D41_SAVE_RATE       save every N superbatches (0 = final only)
 *   D41_OUT             checkpoint dir (default d41_alloc_ckpt)
 *   D41_ID              net id (default d41-alloc-v0)
 */
use bullet_compiler::tensor::TValue;
use bullet_lib::nn::InitSettings;
use bullet_trainer::{
    model::{
        ModelDefinition, ModelEvaluator, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat,
    },
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train},
};
use chessengine::eval::nnue::sota_features::{
    HALFKA_DIMS, THREAT_DIMS, for_each_halfka, for_each_threat,
};
use chessengine::{Board, Color, Piece, PieceType, Square};
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

// ── One training example: a single (position, Complete arm) row ─────────────
#[derive(Clone, Copy)]
struct AllocDatum {
    feats: [i16; SOURCE_FEATURES],
    stm: u8,
    move_idx: i16, // from*64+to, absolute board squares
    margin: i32,   // cp: value - best_arm_value, <= 0
    position_id: u32,
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
        // Parse the row count from the shape tuple in the header dict.
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
        self.file
            .seek(SeekFrom::Start(self.data_offset))
            .expect("seek npy data");
        let mut bytes = vec![0u8; self.rows * self.row_bytes];
        self.file.read_exact(&mut bytes).expect("read npy body");
        bytes
    }
}

fn read_i16(bytes: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([bytes[off], bytes[off + 1]])
}
fn read_i32(bytes: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}
fn read_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// Load the whole flat SoA corpus into a `Vec<AllocDatum>`.
fn load_corpus(dir: &Path) -> Vec<AllocDatum> {
    let mut feats = NpyFile::open(dir.join("feats.npy"), SOURCE_FEATURES * 2);
    let mut stm = NpyFile::open(dir.join("stm.npy"), 1);
    let mut move_idx = NpyFile::open(dir.join("move_idx.npy"), 2);
    let mut margin = NpyFile::open(dir.join("margin.npy"), 4);
    let mut position_id = NpyFile::open(dir.join("position_id.npy"), 4);
    let n = feats.rows;
    assert_eq!(stm.rows, n, "stm row count");
    assert_eq!(move_idx.rows, n, "move_idx row count");
    assert_eq!(margin.rows, n, "margin row count");
    assert_eq!(position_id.rows, n, "position_id row count");

    let feats_b = feats.read_all();
    let stm_b = stm.read_all();
    let move_b = move_idx.read_all();
    let margin_b = margin.read_all();
    let pid_b = position_id.read_all();

    (0..n)
        .map(|r| {
            let mut f = [-1i16; SOURCE_FEATURES];
            for (i, slot) in f.iter_mut().enumerate() {
                *slot = read_i16(&feats_b, (r * SOURCE_FEATURES + i) * 2);
            }
            AllocDatum {
                feats: f,
                stm: stm_b[r],
                move_idx: read_i16(&move_b, r * 2),
                margin: read_i32(&margin_b, r * 4),
                position_id: read_u32(&pid_b, r * 4),
            }
        })
        .collect()
}

// ── In-memory row-shuffling reader ──────────────────────────────────────────
#[derive(Clone)]
struct AllocReader {
    data: Arc<Vec<AllocDatum>>,
    chunk_rows: usize,
    seed: u64,
}

impl DataReader<AllocDatum> for AllocReader {
    fn read_chunks<F: FnMut(&[AllocDatum]) -> bool>(&self, skip_count: usize, mut f: F) {
        let n = self.data.len();
        if n == 0 {
            return;
        }
        let mut produced = 0usize;
        let mut epoch = 0u64;
        loop {
            // Fresh Fisher-Yates permutation per epoch (deterministic in seed).
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
                let chunk: Vec<AllocDatum> =
                    idx[pos..end].iter().map(|&k| self.data[k as usize]).collect();
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

/// A position is held out iff hash(position_id) falls in the first `pct%`.
fn is_eval(position_id: u32, pct: u32, split_seed: u64) -> bool {
    pct > 0 && splitmix64(split_seed ^ u64::from(position_id)) % 100 < u64::from(pct)
}

/// Reconstruct the board from the absolute source features (colour*6+pt)*64+sq.
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

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Mover-relative sparse features (STM slot first) + the from×to select index,
/// shared by the training mapper and the held-out evaluator. `move_sel` is the
/// bilinear graph's linear cell (to*64+from), vertically flipped for black to
/// stay in the mover frame; `alloc_target` is sigmoid(margin/scale).
#[allow(clippy::type_complexity)]
fn map_row(
    datum: &AllocDatum,
    margin_scale: f32,
    hk_stm: &mut [i32],
    hk_ntm: &mut [i32],
    th_stm: &mut [i32],
    th_ntm: &mut [i32],
    move_sel: &mut [i32],
    alloc_target: &mut [f32],
) {
    let board = board_from_features(&datum.feats);
    let stm_color = if datum.stm == 0 { Color::White } else { Color::Black };
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

    // Mover-relative from×to: flip both squares vertically when black moves.
    let raw = datum.move_idx as usize;
    let (from, to) = (raw / 64, raw % 64);
    let (from, to) = if datum.stm == 0 { (from, to) } else { (from ^ 56, to ^ 56) };
    // d40 bilinear layout: graph cell (from,to) lives at linear index to*64+from.
    move_sel[0] = (to * 64 + from) as i32;

    alloc_target[0] = sigmoid(datum.margin as f32 / margin_scale);
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

fn main() {
    let data = PathBuf::from(
        std::env::var("D41_DATA")
            .unwrap_or_else(|_| "data/obs/train/sample-v1-reckless".to_string()),
    );
    let margin_scale = env_f32("D41_MARGIN_SCALE", 200.0);
    assert!(margin_scale > 0.0 && margin_scale.is_finite(), "D41_MARGIN_SCALE must be positive");
    let rank = env_usize("D41_RANK", DEFAULT_RANK);
    assert!(rank > 0 && rank <= 64, "D41_RANK must be in 1..=64");
    let learning_rate = env_f32("D41_LR", 0.001);
    let final_lr_factor = env_f32("D41_LR_FINAL_FACTOR", 1.0);
    assert!(final_lr_factor > 0.0 && final_lr_factor <= 1.0);
    let batch_size = env_usize("D41_BATCH", 4096);
    let batches_per_superbatch = env_usize("D41_BPS", 300);
    let end_superbatch = env_usize("D41_SUPERBATCHES", 1);
    let threads = env_usize("D41_THREADS", 6) as u8;
    let holdout_pct = env_usize("D41_HOLDOUT_PCT", 20) as u32;
    assert!(holdout_pct < 100, "D41_HOLDOUT_PCT must be 0-99");
    let split_seed = env_usize("D41_SPLIT_SEED", 0x0A11_C0DE) as u64;
    let seed = env_usize("D41_SEED", 0x00D4_1A11) as u64;
    let eval_max = env_usize("D41_EVAL_MAX", 0);
    let log_rate = env_usize("D41_LOG_RATE", 50).max(1);
    let save_rate = env_usize("D41_SAVE_RATE", 0);
    let out_dir = std::env::var("D41_OUT").unwrap_or_else(|_| "d41_alloc_ckpt".to_string());
    let net_id = std::env::var("D41_ID").unwrap_or_else(|_| "d41-alloc-v0".to_string());
    std::fs::create_dir_all(&out_dir).expect("create D41_OUT");

    // ── Load + split by position_id (leakage-safe: all arms of a position
    //    stay on one side) ────────────────────────────────────────────────
    println!("=== D41 Phase-1 allocation-head trainer (v0) ===");
    let all = load_corpus(&data);
    let positions: std::collections::HashSet<u32> = all.iter().map(|d| d.position_id).collect();
    let (train_data, eval_data): (Vec<_>, Vec<_>) = all
        .iter()
        .copied()
        .partition(|d| !is_eval(d.position_id, holdout_pct, split_seed));
    let train_positions: std::collections::HashSet<u32> =
        train_data.iter().map(|d| d.position_id).collect();
    let eval_positions: std::collections::HashSet<u32> =
        eval_data.iter().map(|d| d.position_id).collect();
    println!(
        "corpus {}: {} rows / {} positions -> train {} rows / {} pos, eval {} rows / {} pos",
        data.display(),
        all.len(),
        positions.len(),
        train_data.len(),
        train_positions.len(),
        eval_data.len(),
        eval_positions.len(),
    );
    assert!(!train_data.is_empty(), "empty train split");

    // ── Baselines, computed explicitly from the data ─────────────────────
    // target = sigmoid(margin/scale); the trivial `predict-zero-margin`
    // model outputs 0.5 (margin 0) everywhere. The optimal-constant model
    // outputs the mean target — a model that has learned nothing
    // position-specific cannot beat it.
    let target = |d: &AllocDatum| sigmoid(d.margin as f32 / margin_scale);
    let train_targets: Vec<f32> = train_data.iter().map(target).collect();
    let train_mean = train_targets.iter().sum::<f32>() / train_targets.len() as f32;
    let mse = |ts: &[f32], pred: f32| ts.iter().map(|&t| (pred - t).powi(2)).sum::<f32>() / ts.len() as f32;
    let mse_var = |ts: &[f32]| {
        let m = ts.iter().sum::<f32>() / ts.len() as f32;
        ts.iter().map(|&t| (m - t).powi(2)).sum::<f32>() / ts.len() as f32
    };
    let train_zero_base = mse(&train_targets, 0.5);
    let train_mean_base = mse_var(&train_targets);
    println!(
        "target sigmoid(margin/{margin_scale}): train mean {train_mean:.4}",
    );
    println!(
        "BASELINE (train)  zero-margin(0.5) MSE {train_zero_base:.6}   optimal-constant MSE {train_mean_base:.6}",
    );
    let eval_targets: Vec<f32> = eval_data.iter().map(target).collect();
    if !eval_targets.is_empty() {
        let eval_zero_base = mse(&eval_targets, 0.5);
        // honest generalization baseline: the TRAIN mean applied to eval rows.
        let eval_mean_base = mse(&eval_targets, train_mean);
        println!(
            "BASELINE (eval)   zero-margin(0.5) MSE {eval_zero_base:.6}   train-mean-constant MSE {eval_mean_base:.6}",
        );
    }

    // ── Model, optimiser, loader ─────────────────────────────────────────
    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("threat_stm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("move_sel", (POLICY_DIM, 1), 1)
        .add_dense("alloc_target", (1, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |datum: &AllocDatum,
              _step,
              (((((hk_stm, hk_ntm), th_stm), th_ntm), move_sel), alloc_target)| {
            map_row(
                datum, margin_scale, hk_stm, hk_ntm, th_stm, th_ntm, move_sel, alloc_target,
            );
        },
    );

    let device = DefaultDevice::new(0).expect("device");
    // Shared D40 trunk + rank-`rank` bilinear from×to head (the proven gen-0
    // shape from d40_policy). `optimiser.definition()` later backs the held-out
    // evaluator, so the graph is built exactly once.
    let defn = ModelDefinition::build(
        &inputs,
        move |builder, (((((hk_stm, hk_ntm), th_stm), th_ntm), move_sel), alloc_target)| {
            // Two separately-quantised FT tensors (HalfKA i16 + threats i8), summed.
            let pst = builder.new_affine("l0_pst", HALFKA_DIMS, HL);
            pst.init_with_effective_input_size(32);
            let threat = builder.new_weights("l0_threatw", (HL, THREAT_DIMS), InitSettings::Zeroed);
            let stm_acc = pst.forward(hk_stm) + threat.matmul(th_stm);
            let ntm_acc = pst.forward(hk_ntm) + threat.matmul(th_ntm);
            let trunk = stm_acc
                .crelu()
                .pairwise_mul()
                .concat(ntm_acc.crelu().pairwise_mul());

            // Rank-`rank` bilinear from×to head: s(m|x) = h(x)^T e(m).
            let from = builder
                .new_affine("pol_from", HL, 64 * rank)
                .forward(trunk)
                .reshape((64, rank));
            let to = builder
                .new_affine("pol_to", HL, 64 * rank)
                .forward(trunk)
                .reshape((rank, 64));
            let logits = from.matmul(to).reshape((POLICY_DIM, 1));

            // Gather the observed move's cell, squash, MSE-regress the margin.
            let gathered = logits.select(move_sel);
            let pred = gathered.sigmoid();
            let loss = pred.squared_error(alloc_target).reduce_sum_batch();
            (Some(loss), vec![("alloc".to_string(), gathered)])
        },
    );
    let weights = ModelWeights::new(&defn, seed);
    let mut optimiser = AdamW::new(defn, weights, device.clone(), AdamWParams::default())
        .expect("optimiser");

    let reader = AllocReader {
        data: Arc::new(train_data),
        chunk_rows: batch_size.clamp(256, 1024),
        seed,
    };
    let loader = ReadMapLoader::new(reader, mapper.clone(), threads);

    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch,
        },
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
        "training: rank {rank}, {end_superbatch} superbatch(es) x {batches_per_superbatch} batches x {batch_size}, lr {learning_rate}->{}",
        learning_rate * final_lr_factor,
    );
    println!("  (compare batch loss below against train zero-margin baseline {train_zero_base:.6})");

    let save_out = out_dir.clone();
    let save_id = net_id.clone();
    train(
        &mut optimiser,
        schedule,
        loader,
        |_, step, loss| {
            if step.batch() % log_rate == 0 {
                println!("  sb {} batch {} loss {loss:.6}", step.superbatch(), step.batch());
            }
        },
        move |optimiser, step| {
            let sb = step.superbatch();
            if save_rate > 0 && sb % save_rate == 0 && sb < end_superbatch {
                let bytes = optimiser
                    .cpu_weights()
                    .expect("sync checkpoint weights")
                    .to_quantised_buffer(&saved_format(), true)
                    .expect("export intermediate checkpoint");
                let path = format!("{save_out}/{save_id}-sb{sb}-quant.bin");
                std::fs::write(&path, &bytes).expect("write intermediate checkpoint");
                println!("  saved {path} ({} bytes)", bytes.len());
            }
        },
    )
    .expect("alloc training");

    // ── Final checkpoint (bullet-style quantised + raw resume source) ────
    let bytes = optimiser
        .cpu_weights()
        .expect("sync weights")
        .to_quantised_buffer(&saved_format(), true)
        .expect("export checkpoint");
    let quant_path = format!("{out_dir}/{net_id}-quant.bin");
    std::fs::write(&quant_path, &bytes).expect("write checkpoint");
    println!("checkpoint: {quant_path} ({} bytes)", bytes.len());
    let raw_path = format!("{out_dir}/{net_id}-raw.bin");
    {
        let mut raw = File::create(&raw_path).expect("create raw checkpoint");
        optimiser
            .cpu_weights()
            .expect("sync weights")
            .write_into(&mut raw)
            .expect("write raw checkpoint");
    }
    println!("raw checkpoint: {raw_path}");

    // ── Held-out eval (batch-1 forward through a fresh evaluator) ────────
    if eval_data.is_empty() {
        println!("no held-out split (D41_HOLDOUT_PCT=0); skipping eval");
        return;
    }
    let mut evaluator =
        ModelEvaluator::new(optimiser.definition(), device.clone()).expect("evaluator");
    evaluator
        .load_weights(&optimiser.cpu_weights().expect("weights for eval"))
        .expect("load eval weights");

    let scored = if eval_max == 0 { eval_data.len() } else { eval_max.min(eval_data.len()) };
    // (position_id, predicted squashed value, margin) per scored row.
    let mut rows: Vec<(u32, f32, i32)> = Vec::with_capacity(scored);
    let mut sse = 0.0f64;
    for datum in &eval_data[..scored] {
        let host = mapper.map(std::slice::from_ref(datum), Step::default(), 1);
        let input = host.to_device(&device).expect("eval inputs to device");
        let output = evaluator.evaluate(&input).expect("eval forward");
        let TValue::F32(v) = output.get("alloc").expect("alloc output").to_host().expect("alloc to host")
        else {
            panic!("alloc output must be f32");
        };
        let pred = sigmoid(v[0]);
        let t = sigmoid(datum.margin as f32 / margin_scale);
        sse += f64::from((pred - t).powi(2));
        rows.push((datum.position_id, pred, datum.margin));
    }
    let eval_mse = (sse / scored as f64) as f32;
    let scored_targets: Vec<f32> = eval_data[..scored].iter().map(target).collect();
    let eval_zero_base = mse(&scored_targets, 0.5);
    let eval_mean_base = mse(&scored_targets, train_mean);
    println!("=== held-out eval ({scored} rows) ===");
    println!("  model MSE            {eval_mse:.6}");
    println!("  zero-margin baseline {eval_zero_base:.6}  (model {:+.1}%)", 100.0 * (eval_mse / eval_zero_base - 1.0));
    println!("  train-mean baseline  {eval_mean_base:.6}  (model {:+.1}%)", 100.0 * (eval_mse / eval_mean_base - 1.0));

    // ── Ranking metrics: the real Phase-1 objective is ordering the legal
    //    moves, and MSE is dominated by the many clearly-bad arms a constant
    //    already fits. Group held-out rows by position and measure whether the
    //    model ranks the best arm first and correlates with true margins.
    rows.sort_by_key(|r| r.0);
    let mut top1_hits = 0usize;
    let mut ranked_positions = 0usize;
    let mut rand_top1_sum = 0.0f64; // sum of 1/arms = expected random top-1
    let mut pearson_sum = 0.0f64;
    let mut pearson_n = 0usize;
    let mut i = 0usize;
    while i < rows.len() {
        let pid = rows[i].0;
        let mut j = i;
        while j < rows.len() && rows[j].0 == pid {
            j += 1;
        }
        let group = &rows[i..j];
        i = j;
        if group.len() < 2 {
            continue; // ranking is undefined for a single-arm position
        }
        ranked_positions += 1;
        rand_top1_sum += 1.0 / group.len() as f64;
        // top-1: does the highest-predicted arm carry the max margin (== best)?
        let pred_best = group.iter().enumerate().max_by(|a, b| a.1.1.total_cmp(&b.1.1)).unwrap().0;
        let true_best_margin = group.iter().map(|r| r.2).max().unwrap();
        if group[pred_best].2 == true_best_margin {
            top1_hits += 1;
        }
        // per-position Pearson(pred, margin); skip degenerate (all-equal) groups.
        let n = group.len() as f64;
        let mp = group.iter().map(|r| f64::from(r.1)).sum::<f64>() / n;
        let mm = group.iter().map(|r| f64::from(r.2)).sum::<f64>() / n;
        let mut cov = 0.0;
        let mut vp = 0.0;
        let mut vm = 0.0;
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
    if ranked_positions > 0 {
        let top1 = top1_hits as f32 / ranked_positions as f32;
        let rand_top1 = (rand_top1_sum / ranked_positions as f64) as f32;
        println!("=== held-out ranking ({ranked_positions} multi-arm positions) ===");
        println!("  top-1 best-move accuracy {top1:.3}  (random guess {rand_top1:.3})");
        if pearson_n > 0 {
            println!(
                "  mean per-position Pearson(pred, margin) {:.3}",
                pearson_sum / pearson_n as f64
            );
        }
    }
    println!("=== d41_alloc smoke complete ===");
}
