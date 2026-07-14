/*
 * D40 real dual-target policy trainer (design/11 §12.5/§12.7).
 *
 * Reads data/lc0_nnue/policy_eval_ds_100m directly from its NumPy SoA files:
 * 32 mover-relative piece features, STM-relative eval cp, and up to 40 sparse
 * from×to probabilities. The piece features reconstruct the complete placement,
 * which is then fed through the engine's canonical HalfKAv2_hm + FullThreats
 * enumerators. No synthetic positions or policy labels remain.
 *
 * D40_POLICY_HEAD selects `bilinear` (rank 8) or `full` (4096 logits). Both
 * heads read the shared post-pairwise 768-wide trunk and optimize
 * eval_mse + D40_POLICY_LAMBDA * policy_cross_entropy.
 */
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
const POLICY_RANK: usize = 8;
const ROWS_PER_SIDE: usize = 50_000_000;
const SCALE: f32 = 400.0;

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
    Bilinear,
    Full,
}

impl PolicyHead {
    fn from_env() -> Self {
        match std::env::var("D40_POLICY_HEAD")
            .unwrap_or_else(|_| "bilinear".to_string())
            .as_str()
        {
            "bilinear" => Self::Bilinear,
            "full" => Self::Full,
            other => panic!("D40_POLICY_HEAD must be bilinear or full, got {other}"),
        }
    }
}

#[derive(Clone)]
struct PolicyDatum {
    features: [i16; SOURCE_FEATURES],
    moves: [i16; POLICY_MOVES],
    probabilities: [f32; POLICY_MOVES],
    eval: i16,
}

#[derive(Clone)]
struct NpyPolicyReader {
    root: PathBuf,
    chunk_rows: usize,
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
}

impl SideFiles {
    fn open(root: &Path, side: &str) -> Self {
        let dir = root.join(side);
        Self {
            features: NpyFile::open(dir.join("feats.npy"), SOURCE_FEATURES * 2),
            moves: NpyFile::open(dir.join("moves.npy"), POLICY_MOVES * 2),
            probabilities: NpyFile::open(dir.join("probs.npy"), POLICY_MOVES * 2),
            eval: NpyFile::open(dir.join("eval.npy"), 2),
        }
    }

    fn read(&mut self, row: usize, rows: usize) -> Vec<PolicyDatum> {
        let features = self.features.read(row, rows);
        let moves = self.moves.read(row, rows);
        let probabilities = self.probabilities.read(row, rows);
        let eval = self.eval.read(row, rows);
        (0..rows)
            .map(|r| {
                let mut datum = PolicyDatum {
                    features: [-1; SOURCE_FEATURES],
                    moves: [-1; POLICY_MOVES],
                    probabilities: [0.0; POLICY_MOVES],
                    eval: read_i16(&eval, r * 2),
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
            SideFiles::open(&self.root, "white"),
            SideFiles::open(&self.root, "black"),
        ];
        let mut side = (skip_count / self.chunk_rows) & 1;
        let mut row = (skip_count / 2) % ROWS_PER_SIDE;
        loop {
            let rows = self.chunk_rows.min(ROWS_PER_SIDE - row);
            let data = sides[side].read(row, rows);
            if f(&data) {
                break;
            }
            side ^= 1;
            if side == 0 {
                row = (row + rows) % ROWS_PER_SIDE;
            }
        }
    }
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
    let lambda = env_f32("D40_POLICY_LAMBDA", 1.0);
    let batch_size = env_usize("D40_BATCH", 8192);
    let batches_per_superbatch = env_usize("D40_BPS", 100);
    let end_superbatch = env_usize("D40_SUPERBATCHES", 1);
    let threads = env_usize("D40_THREADS", 6) as u8;
    let out_dir = std::env::var("D40_OUT").unwrap_or_else(|_| "d40_policy_ckpt".to_string());
    let net_id = std::env::var("D40_ID").unwrap_or_else(|_| "d40-policy".to_string());
    std::fs::create_dir_all(&out_dir).expect("create D40_OUT");

    let probe = SideFiles::open(&data, "white").read(0, 1).remove(0);
    let probe_board = board_from_features(&probe.features);
    let probability_sum: f32 = probe.probabilities.iter().sum();
    assert!((probability_sum - 1.0).abs() < 0.02);
    assert_eq!(probe_board.all_occupancy().count() as usize, 32);
    println!(
        "real dual-target reader: eval={}cp, pieces=32, policy_sum={probability_sum:.6}, head={head:?}",
        probe.eval
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
                eval_target[0] = 1.0 / (1.0 + (-f32::from(datum.eval) / SCALE).exp());
                policy_target.fill(0.0);
                for i in 0..POLICY_MOVES {
                    if datum.moves[i] >= 0 {
                        policy_target[datum.moves[i] as usize] += datum.probabilities[i];
                    }
                }
            },
        );

    let defn = ModelDefinition::build(
        &inputs,
        |builder,
         ((((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), eval_target), policy_target)| {
            let pst = builder.new_affine("l0_pst", HALFKA_DIMS, HL);
            pst.init_with_effective_input_size(32);
            let threat = builder.new_weights("l0_threatw", (HL, THREAT_DIMS), InitSettings::Zeroed);
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
                PolicyHead::Full => builder
                    .new_affine("pol_full", HL, POLICY_DIM)
                    .forward(trunk),
                PolicyHead::Bilinear => {
                    let from = builder
                        .new_affine("pol_from", HL, 64 * POLICY_RANK)
                        .forward(trunk)
                        .reshape((64, POLICY_RANK));
                    let to = builder
                        .new_affine("pol_to", HL, 64 * POLICY_RANK)
                        .forward(trunk)
                        .reshape((POLICY_RANK, 64));
                    from.matmul(to).reshape((POLICY_DIM, 1))
                }
            };
            let policy_loss = policy_logits
                .softmax_crossentropy_loss(policy_target)
                .reduce_sum_rows();
            let loss = (eval_loss + lambda * policy_loss).reduce_sum_batch();
            (
                Some(loss),
                vec![
                    ("eval".to_string(), eval_out),
                    ("policy".to_string(), policy_logits),
                ],
            )
        },
    );

    let weights = ModelWeights::new(&defn, 0x00D4_0901);
    let mut optimiser = AdamW::new(
        defn,
        weights,
        DefaultDevice::new(0).expect("device"),
        AdamWParams::default(),
    )
    .expect("optimiser");
    let reader = NpyPolicyReader {
        root: data,
        chunk_rows: batch_size.max(1024),
    };
    let loader = ReadMapLoader::new(reader, mapper, threads);
    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch,
        },
        log_rate: 1,
        lr_schedule: Box::new(|_: Step| 0.001),
    };
    println!(
        "training {head:?}: {end_superbatch} superbatches x {batches_per_superbatch} batches x {batch_size}"
    );
    train(
        &mut optimiser,
        schedule,
        loader,
        |_, step, loss| {
            println!(
                "  sb {} batch {} combined_loss={loss:.6}",
                step.superbatch(),
                step.batch()
            );
        },
        |_, _| {},
    )
    .expect("dual-target training");

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
    let bytes = optimiser
        .cpu_weights()
        .expect("sync weights")
        .to_quantised_buffer(&format, true)
        .expect("export dual-target checkpoint");
    let head_name = match head {
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
