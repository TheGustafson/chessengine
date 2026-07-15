/*
 * D40 real dual-target policy trainer (design/11 §12.5/§12.7).
 *
 * Reads data/lc0_nnue/policy_eval_ds_100m directly from its NumPy SoA files:
 * 32 mover-relative piece features, STM-relative eval cp and game result, and
 * up to 40 sparse from×to probabilities. The piece features reconstruct the complete placement,
 * which is then fed through the engine's canonical HalfKAv2_hm + FullThreats
 * enumerators. No synthetic positions or policy labels remain.
 *
 * D40_POLICY_HEAD selects `eval`, `bilinear` (rank 8), or `full` (4096 logits).
 * All three use the same dual-target corpus and shared D40 trunk; `eval`
 * deliberately ignores the policy label to provide the controlled baseline.
 */
use bullet_compiler::tensor::TValue;
use bullet_lib::nn::InitSettings;
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat},
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

const HALFKA_MAX_ACTIVE: usize = 32;
const THREAT_MAX_ACTIVE: usize = 160;
const POLICY_DIM: usize = 4096;
const POLICY_MOVES: usize = 40;
const SOURCE_FEATURES: usize = 32;
const HL: usize = 768;
const L2: usize = 16;
const L3: usize = 32;
const BUCKETS: usize = 8;
const DEFAULT_POLICY_RANK: usize = 8;
const DEFAULT_ROWS_PER_SIDE: usize = 50_000_000;
// The stored label is cp=400*atanh(Q), where Lc0 Q=P(win)-P(loss).
// Its expected game score is (1+Q)/2 = sigmoid(2*atanh(Q)), making denominator
// 200 the mathematical inverse. However, a strict classic-net A/B found that
// sharper target about 80 Elo worse than the softer net-4 target. Default to
// the empirically stronger 400 and retain this knob for crossed controls.
const DEFAULT_EVAL_TARGET_SCALE: f32 = 400.0;

#[rustfmt::skip]
const OUTPUT_BUCKETS_LAYOUT: [u8; 33] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7, 7,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PolicyHead {
    Eval,
    Bilinear,
    Full,
}

impl PolicyHead {
    fn from_env() -> Self {
        match std::env::var("D40_POLICY_HEAD")
            .unwrap_or_else(|_| "bilinear".to_string())
            .as_str()
        {
            "eval" => Self::Eval,
            "bilinear" => Self::Bilinear,
            "full" => Self::Full,
            other => panic!("D40_POLICY_HEAD must be eval, bilinear, or full, got {other}"),
        }
    }
}

fn saved_format(head: PolicyHead) -> Vec<SavedFormat> {
    let mut format = vec![
        SavedFormat::id("l0_pstw").round().quantise::<i16>(255),
        SavedFormat::id("l0_pstb").round().quantise::<i16>(255),
        SavedFormat::id("l0_threatw").round().quantise::<i8>(64),
        SavedFormat::id("l1w")
            .transpose()
            .round()
            .quantise::<i8>(64),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w").transpose(),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w").transpose(),
        SavedFormat::id("l3b"),
    ];
    match head {
        PolicyHead::Eval => {}
        PolicyHead::Full => {
            format.push(SavedFormat::id("pol_fullw").transpose());
            format.push(SavedFormat::id("pol_fullb"));
        }
        PolicyHead::Bilinear => {
            format.push(SavedFormat::id("pol_fromw").transpose());
            format.push(SavedFormat::id("pol_fromb"));
            format.push(SavedFormat::id("pol_tow").transpose());
            format.push(SavedFormat::id("pol_tob"));
        }
    }
    format
}

fn transpose_saved(rows: usize, cols: usize, saved: Vec<f32>) -> Vec<f32> {
    let mut internal = vec![0.0; saved.len()];
    for row in 0..rows {
        for col in 0..cols {
            internal[rows * col + row] = saved[cols * row + col];
        }
    }
    internal
}

fn load_eval_quantised_weights(weights: &mut ModelWeights, path: &str) {
    let raw = std::fs::read(path).expect("read D40_INIT_QUANT");
    let mut offset = 0usize;
    let mut i16s = |count: usize, scale: f32| {
        let end = offset + count * 2;
        let values = raw[offset..end]
            .chunks_exact(2)
            .map(|bytes| f32::from(i16::from_le_bytes(bytes.try_into().unwrap())) / scale)
            .collect();
        offset = end;
        values
    };
    let pst = i16s(HALFKA_DIMS * HL, 255.0);
    let pst_bias = i16s(HL, 255.0);
    drop(i16s);
    let mut i8s = |count: usize, scale: f32| {
        let end = offset + count;
        let values = raw[offset..end]
            .iter()
            .map(|&byte| f32::from(byte as i8) / scale)
            .collect();
        offset = end;
        values
    };
    let threat = i8s(THREAT_DIMS * HL, 64.0);
    let l1_saved = i8s(BUCKETS * L2 * HL, 64.0);
    drop(i8s);
    let mut f32s = |count: usize| {
        let end = offset + count * 4;
        let values = raw[offset..end]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        offset = end;
        values
    };
    let l1_bias = f32s(BUCKETS * L2);
    let l2_saved = f32s(BUCKETS * L3 * L2);
    let l2_bias = f32s(BUCKETS * L3);
    let l3_saved = f32s(BUCKETS * L3);
    let l3_bias = f32s(BUCKETS);
    drop(f32s);
    let padding = &raw[offset..];
    assert!(padding.len() <= 63, "oversized quantized padding");
    assert!(
        padding
            .iter()
            .enumerate()
            .all(|(i, &byte)| byte == b"bullet"[i % 6]),
        "invalid quantized padding"
    );

    for (id, values) in [
        ("l0_pstw", pst),
        ("l0_pstb", pst_bias),
        ("l0_threatw", threat),
        ("l1w", transpose_saved(BUCKETS * L2, HL, l1_saved)),
        ("l1b", l1_bias),
        ("l2w", transpose_saved(BUCKETS * L3, L2, l2_saved)),
        ("l2b", l2_bias),
        ("l3w", transpose_saved(BUCKETS, L3, l3_saved)),
        ("l3b", l3_bias),
    ] {
        assert!(
            weights.set(id, TValue::F32(values)),
            "quant tensor shape {id}"
        );
    }
}

#[derive(Clone)]
struct PolicyDatum {
    features: [i16; SOURCE_FEATURES],
    moves: [i16; POLICY_MOVES],
    probabilities: [f32; POLICY_MOVES],
    eval: i16,
    result: f32,
}

#[derive(Clone)]
struct NpyPolicyReader {
    root: PathBuf,
    eval_name: String,
    chunk_rows: usize,
    rows_per_side: usize,
    seed: u64,
}

struct NpyFile {
    file: File,
    data_offset: u64,
    row_bytes: usize,
}

impl NpyFile {
    fn open(path: impl AsRef<Path>, row_bytes: usize) -> Self {
        let mut file = File::open(path.as_ref()).expect("open policy npy");
        let mut preamble = [0u8; 10];
        file.read_exact(&mut preamble).expect("read npy preamble");
        assert_eq!(&preamble[..6], b"\x93NUMPY", "bad npy magic");
        assert_eq!(preamble[6], 1, "only NumPy v1 files are supported");
        let header_len = u16::from_le_bytes([preamble[8], preamble[9]]) as u64;
        Self {
            file,
            data_offset: 10 + header_len,
            row_bytes,
        }
    }

    fn read(&mut self, row: usize, rows: usize) -> Vec<u8> {
        self.file
            .seek(SeekFrom::Start(
                self.data_offset + (row * self.row_bytes) as u64,
            ))
            .expect("seek policy npy");
        let mut bytes = vec![0; rows * self.row_bytes];
        self.file
            .read_exact(&mut bytes)
            .expect("read policy npy rows");
        bytes
    }
}

struct SideFiles {
    features: NpyFile,
    moves: NpyFile,
    probabilities: NpyFile,
    eval: NpyFile,
    result: NpyFile,
}

impl SideFiles {
    fn open(root: &Path, side: &str, eval_name: &str) -> Self {
        let dir = root.join(side);
        Self {
            features: NpyFile::open(dir.join("feats.npy"), SOURCE_FEATURES * 2),
            moves: NpyFile::open(dir.join("moves.npy"), POLICY_MOVES * 2),
            probabilities: NpyFile::open(dir.join("probs.npy"), POLICY_MOVES * 2),
            eval: NpyFile::open(dir.join(eval_name), 2),
            result: NpyFile::open(dir.join("result.npy"), 1),
        }
    }

    fn read(&mut self, row: usize, rows: usize) -> Vec<PolicyDatum> {
        let features = self.features.read(row, rows);
        let moves = self.moves.read(row, rows);
        let probabilities = self.probabilities.read(row, rows);
        let eval = self.eval.read(row, rows);
        let result = self.result.read(row, rows);
        (0..rows)
            .map(|r| {
                let mut datum = PolicyDatum {
                    features: [-1; SOURCE_FEATURES],
                    moves: [-1; POLICY_MOVES],
                    probabilities: [0.0; POLICY_MOVES],
                    eval: read_i16(&eval, r * 2),
                    result: f32::from(result[r]) / 2.0,
                };
                for i in 0..SOURCE_FEATURES {
                    datum.features[i] = read_i16(&features, (r * SOURCE_FEATURES + i) * 2);
                }
                for i in 0..POLICY_MOVES {
                    datum.moves[i] = read_i16(&moves, (r * POLICY_MOVES + i) * 2);
                    let bits = read_u16(&probabilities, (r * POLICY_MOVES + i) * 2);
                    datum.probabilities[i] = f16_to_f32(bits);
                }
                datum
            })
            .collect()
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let out = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut frac = fraction;
            let mut exp = 113u32;
            while frac & 0x0400 == 0 {
                frac <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((frac & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(out)
}

impl DataReader<PolicyDatum> for NpyPolicyReader {
    fn read_chunks<F: FnMut(&[PolicyDatum]) -> bool>(&self, skip_count: usize, mut f: F) {
        let mut sides = [
            SideFiles::open(&self.root, "white", &self.eval_name),
            SideFiles::open(&self.root, "black", &self.eval_name),
        ];
        let chunks = self.rows_per_side.div_ceil(self.chunk_rows);
        let mut sequence = 0usize;
        let mut skipped = 0usize;
        loop {
            let epoch = sequence / chunks;
            let slot = sequence % chunks;
            let mixed = splitmix64(self.seed ^ epoch as u64);
            let offset = mixed as usize % chunks;
            let mut stride = ((mixed >> 32) as usize % chunks).max(1);
            while gcd(stride, chunks) != 1 {
                stride += 1;
                if stride == chunks {
                    stride = 1;
                }
            }
            let chunk = (offset + slot * stride) % chunks;
            let row = chunk * self.chunk_rows;
            let rows = self.chunk_rows.min(self.rows_per_side - row);
            for side in &mut sides {
                let data = side.read(row, rows);
                if skipped + data.len() <= skip_count {
                    skipped += data.len();
                    continue;
                }
                let start = skip_count.saturating_sub(skipped);
                skipped += data.len();
                if f(&data[start..]) {
                    return;
                }
            }
            sequence += 1;
        }
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
        board.put_piece(
            Piece::new(color, piece_type),
            Square::new((feature % 64) as u8),
        );
    }
    board
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let data = PathBuf::from(
        std::env::var("D40_POLICY_DATA")
            .unwrap_or_else(|_| "data/lc0_nnue/policy_eval_ds_100m".to_string()),
    );
    let head = PolicyHead::from_env();
    let policy_rank = env_usize("D40_POLICY_RANK", DEFAULT_POLICY_RANK);
    assert!(
        policy_rank > 0 && policy_rank <= 64,
        "D40_POLICY_RANK must be in 1..=64"
    );
    let lambda = env_f32("D40_POLICY_LAMBDA", 1.0);
    let wdl = env_f32("D40_WDL", 0.3);
    let eval_target_scale = env_f32("D40_EVAL_TARGET_SCALE", DEFAULT_EVAL_TARGET_SCALE);
    assert!((0.0..=1.0).contains(&wdl), "D40_WDL must be in [0,1]");
    assert!(
        eval_target_scale > 0.0,
        "D40_EVAL_TARGET_SCALE must be positive"
    );
    let batch_size = env_usize("D40_BATCH", 8192);
    let batches_per_superbatch = env_usize("D40_BPS", 100);
    let end_superbatch = env_usize("D40_SUPERBATCHES", 1);
    let threads = env_usize("D40_THREADS", 6) as u8;
    let log_rate = env_usize("D40_LOG_RATE", 100);
    let save_rate = env_usize("D40_SAVE_RATE", 0);
    let rows_per_side = env_usize("D40_ROWS_PER_SIDE", DEFAULT_ROWS_PER_SIDE);
    let seed = env_usize("D40_SEED", 0x00D4_0901) as u64;
    let learning_rate = env_f32("D40_LR", 0.001);
    let final_lr_factor = env_f32("D40_LR_FINAL_FACTOR", 1.0);
    assert!(final_lr_factor > 0.0 && final_lr_factor <= 1.0);
    let out_dir = std::env::var("D40_OUT").unwrap_or_else(|_| "d40_policy_ckpt".to_string());
    let net_id = std::env::var("D40_ID").unwrap_or_else(|_| "d40-policy".to_string());
    std::fs::create_dir_all(&out_dir).expect("create D40_OUT");

    let eval_name = std::env::var("D40_EVAL_NAME").unwrap_or_else(|_| "eval.npy".to_string());
    let probe = SideFiles::open(&data, "white", &eval_name)
        .read(0, 1)
        .remove(0);
    let probe_board = board_from_features(&probe.features);
    let probe_pieces = probe_board.all_occupancy().count() as usize;
    let probability_sum: f32 = probe.probabilities.iter().sum();
    assert!((probability_sum - 1.0).abs() < 0.02);
    assert!((2..=32).contains(&probe_pieces));
    println!(
        "real dual-target reader: eval={}cp, eval_file={eval_name}, result={}, pieces={probe_pieces}, policy_sum={probability_sum:.6}, head={head:?}, policy_rank={policy_rank}, wdl={wdl}, eval_target_scale={eval_target_scale}, lr={learning_rate}->{}",
        probe.eval,
        probe.result,
        learning_rate * final_lr_factor
    );

    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_DIMS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("threat_stm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_DIMS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("buckets", (BUCKETS, 1), 1)
        .add_dense("eval_target", (1, 1))
        .add_dense("policy_target", (POLICY_DIM, 1));

    let mapper =
        ModelInputsMapper::build(
            &inputs,
            move |datum: &PolicyDatum,
                  _step,
                  (
                (((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), eval_target),
                policy_target,
            )| {
                let board = board_from_features(&datum.features);
                let mut hk_count = [0usize; 2];
                let mut th_count = [0usize; 2];
                for perspective in Color::ALL {
                    let p = perspective as usize;
                    let hk = if perspective == Color::White {
                        &mut *hk_stm
                    } else {
                        &mut *hk_ntm
                    };
                    for_each_halfka(&board, perspective, |idx| {
                        hk[hk_count[p]] = idx as i32;
                        hk_count[p] += 1;
                    });
                    hk[hk_count[p]..].fill(-1);

                    let threats = if perspective == Color::White {
                        &mut *th_stm
                    } else {
                        &mut *th_ntm
                    };
                    for_each_threat(&board, perspective, |idx| {
                        assert!(
                            th_count[p] < THREAT_MAX_ACTIVE,
                            "threat sparse capacity exceeded"
                        );
                        threats[th_count[p]] = idx as i32;
                        th_count[p] += 1;
                    });
                    threats[th_count[p]..].fill(-1);
                }
                buckets[0] = OUTPUT_BUCKETS_LAYOUT[board.all_occupancy().count() as usize] as i32;
                let value = 1.0 / (1.0 + (-f32::from(datum.eval) / eval_target_scale).exp());
                eval_target[0] = wdl * datum.result + (1.0 - wdl) * value;
                policy_target.fill(0.0);
                for i in 0..POLICY_MOVES {
                    if datum.moves[i] >= 0 {
                        let canonical = datum.moves[i] as usize;
                        let index = if head == PolicyHead::Bilinear {
                            // Bullet's 64x64 matmul is column-major but the
                            // subsequent reshape is a no-op. Feed its linear
                            // layout (to*64+from), so graph cell (from,to)
                            // receives the canonical from*64+to target.
                            (canonical % 64) * 64 + canonical / 64
                        } else {
                            canonical
                        };
                        policy_target[index] += datum.probabilities[i];
                    }
                }
            },
        );

    let defn =
        ModelDefinition::build(
            &inputs,
            move |builder,
                  (
                (((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), eval_target),
                policy_target,
            )| {
                let pst = builder.new_affine("l0_pst", HALFKA_DIMS, HL);
                pst.init_with_effective_input_size(32);
                let threat =
                    builder.new_weights("l0_threatw", (HL, THREAT_DIMS), InitSettings::Zeroed);
                let l1 = builder.new_affine("l1", HL, BUCKETS * L2);
                let l2 = builder.new_affine("l2", L2, BUCKETS * L3);
                let l3 = builder.new_affine("l3", L3, BUCKETS);

                let stm_acc = pst.forward(hk_stm) + threat.matmul(th_stm);
                let ntm_acc = pst.forward(hk_ntm) + threat.matmul(th_ntm);
                let trunk = stm_acc
                    .crelu()
                    .pairwise_mul()
                    .concat(ntm_acc.crelu().pairwise_mul());

                let eval_h1 = l1.forward(trunk).select(buckets).screlu();
                let eval_h2 = l2.forward(eval_h1).select(buckets).screlu();
                let eval_out = l3.forward(eval_h2).select(buckets);
                let eval_loss = eval_out.sigmoid().squared_error(eval_target);

                let policy_logits = match head {
                    PolicyHead::Eval => None,
                    PolicyHead::Full => Some(
                        builder
                            .new_affine("pol_full", HL, POLICY_DIM)
                            .forward(trunk),
                    ),
                    PolicyHead::Bilinear => {
                        let from = builder
                            .new_affine("pol_from", HL, 64 * policy_rank)
                            .forward(trunk)
                            .reshape((64, policy_rank));
                        let to = builder
                            .new_affine("pol_to", HL, 64 * policy_rank)
                            .forward(trunk)
                            .reshape((policy_rank, 64));
                        Some(from.matmul(to).reshape((POLICY_DIM, 1)))
                    }
                };
                match policy_logits {
                    Some(logits) => {
                        let policy_loss = logits
                            .softmax_crossentropy_loss(policy_target)
                            .reduce_sum_rows();
                        let loss = (eval_loss + lambda * policy_loss).reduce_sum_batch();
                        (
                            Some(loss),
                            vec![
                                ("eval".to_string(), eval_out),
                                ("policy".to_string(), logits),
                            ],
                        )
                    }
                    None => (
                        Some(eval_loss.reduce_sum_batch()),
                        vec![("eval".to_string(), eval_out)],
                    ),
                }
            },
        );

    let mut weights = ModelWeights::new(&defn, seed);
    if let Ok(path) = std::env::var("D40_INIT_QUANT") {
        assert_eq!(
            head,
            PolicyHead::Eval,
            "D40_INIT_QUANT currently supports eval-only nets"
        );
        load_eval_quantised_weights(&mut weights, &path);
        println!("initialised eval tensors from D40_INIT_QUANT={path}");
    }
    let mut optimiser = AdamW::new(
        defn,
        weights,
        DefaultDevice::new(0).expect("device"),
        AdamWParams::default(),
    )
    .expect("optimiser");
    let reader = NpyPolicyReader {
        root: data,
        eval_name,
        // Mix at least two independently selected chunks into normal 2K+
        // batches instead of presenting long game/tar-contiguous runs.
        chunk_rows: batch_size.clamp(256, 1024),
        rows_per_side,
        seed,
    };
    let loader = ReadMapLoader::new(reader, mapper, threads);
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
            final_lr
                + 0.5 * (learning_rate - final_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }),
    };
    println!(
        "training {head:?}: {end_superbatch} superbatches x {batches_per_superbatch} batches x {batch_size}"
    );
    let checkpoint_out = out_dir.clone();
    let checkpoint_id = net_id.clone();
    train(
        &mut optimiser,
        schedule,
        loader,
        |_, step, loss| {
            if step.batch() % log_rate == 0 {
                println!(
                    "  sb {} batch {} combined_loss={loss:.6}",
                    step.superbatch(),
                    step.batch()
                );
            }
        },
        move |optimiser, step| {
            let superbatch = step.superbatch();
            if save_rate > 0 && superbatch % save_rate == 0 && superbatch < end_superbatch {
                let bytes = optimiser
                    .cpu_weights()
                    .expect("sync checkpoint weights")
                    .to_quantised_buffer(&saved_format(head), true)
                    .expect("export intermediate checkpoint");
                let path = format!(
                    "{checkpoint_out}/{checkpoint_id}-sb{superbatch}-{head_name}-quant.bin",
                    head_name = match head {
                        PolicyHead::Eval => "eval",
                        PolicyHead::Bilinear => "bilinear",
                        PolicyHead::Full => "full",
                    }
                );
                std::fs::write(&path, &bytes).expect("write intermediate checkpoint");
                println!(
                    "saved intermediate checkpoint: {path} ({} bytes)",
                    bytes.len()
                );
            }
        },
    )
    .expect("dual-target training");

    let format = saved_format(head);
    let bytes = optimiser
        .cpu_weights()
        .expect("sync weights")
        .to_quantised_buffer(&format, true)
        .expect("export dual-target checkpoint");
    let head_name = match head {
        PolicyHead::Eval => "eval",
        PolicyHead::Bilinear => "bilinear",
        PolicyHead::Full => "full",
    };
    let path = format!("{out_dir}/{net_id}-{head_name}-quant.bin");
    std::fs::write(&path, &bytes).expect("write dual-target checkpoint");
    println!(
        "dual-target {head:?} checkpoint: {path} ({} bytes)",
        bytes.len()
    );
}
