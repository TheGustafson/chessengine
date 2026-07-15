/*
 * Policy-only trainer on a frozen classic CENN-v1 feature transformer.
 *
 * The evaluator is loaded from CLASSIC_EVAL and never receives gradients.
 * Only a compact position-embedding / legal-move policy head is trained. This
 * gives a bit-identical eval-only sister by construction and lets runtime reuse
 * its incremental 2x256 accumulator instead of refreshing a separate D40
 * trunk. Runtime computes the position embedding once and dots it only with
 * generated legal moves; it never materialises all 4,096 logits.
 */
use bullet_compiler::{model::InitSettings, tensor::TValue};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train},
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const INPUT: usize = 768;
const L1: usize = 256;
const POLICY_DIM: usize = 4096;
const SOURCE_FEATURES: usize = 32;
const DEFAULT_ROWS_PER_SIDE: usize = 10_000_000;

#[derive(Clone)]
struct PolicyDatum {
    features: [i16; SOURCE_FEATURES],
    moves: Vec<i16>,
    probabilities: Vec<f32>,
}

#[derive(Clone)]
struct NpyPolicyReader {
    root: PathBuf,
    chunk_rows: usize,
    rows_per_side: usize,
    policy_moves: usize,
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
        assert_eq!(&preamble[..6], b"\x93NUMPY");
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
    policy_moves: usize,
}

impl SideFiles {
    fn open(root: &Path, side: &str, policy_moves: usize) -> Self {
        let dir = root.join(side);
        Self {
            features: NpyFile::open(dir.join("feats.npy"), SOURCE_FEATURES * 2),
            moves: NpyFile::open(dir.join("moves.npy"), policy_moves * 2),
            probabilities: NpyFile::open(dir.join("probs.npy"), policy_moves * 2),
            policy_moves,
        }
    }

    fn read(&mut self, row: usize, rows: usize) -> Vec<PolicyDatum> {
        let features = self.features.read(row, rows);
        let moves = self.moves.read(row, rows);
        let probabilities = self.probabilities.read(row, rows);
        (0..rows)
            .map(|r| {
                let mut datum = PolicyDatum {
                    features: [-1; SOURCE_FEATURES],
                    moves: vec![-1; self.policy_moves],
                    probabilities: vec![0.0; self.policy_moves],
                };
                for i in 0..SOURCE_FEATURES {
                    datum.features[i] = read_i16(&features, (r * SOURCE_FEATURES + i) * 2);
                }
                for i in 0..self.policy_moves {
                    datum.moves[i] = read_i16(&moves, (r * self.policy_moves + i) * 2);
                    datum.probabilities[i] =
                        f16_to_f32(read_u16(&probabilities, (r * self.policy_moves + i) * 2));
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
            SideFiles::open(&self.root, "white", self.policy_moves),
            SideFiles::open(&self.root, "black", self.policy_moves),
        ];
        let mut side = (skip_count / self.chunk_rows) & 1;
        let mut row = (skip_count / 2) % self.rows_per_side;
        loop {
            let rows = self.chunk_rows.min(self.rows_per_side - row);
            let data = sides[side].read(row, rows);
            if f(&data) {
                break;
            }
            side ^= 1;
            if side == 0 {
                row = (row + rows) % self.rows_per_side;
            }
        }
    }
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

/// Load the quantized CENN-v1 feature transformer as normalized f32 weights.
/// Bullet matrices are column-major (`row + rows * column`), exactly matching
/// CENN's feature-major `[feature][neuron]` transformer payload. Do not
/// transpose this import: doing so trains the head against the wrong frozen
/// accumulator even though its loss can still decrease.
fn load_frozen_ft(path: &Path) -> (Vec<f32>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read CLASSIC_EVAL");
    assert!(bytes.len() >= 48 && &bytes[..4] == b"CENN", "not CENN-v1");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
        INPUT
    );
    assert_eq!(
        u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize,
        L1
    );
    assert_eq!(
        bytes[16], 1,
        "frozen policy trainer currently requires SCReLU"
    );
    assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 255);
    let payload = &bytes[48..];
    let mut weights = vec![0.0; L1 * INPUT];
    for feature in 0..INPUT {
        for neuron in 0..L1 {
            let offset = 2 * (feature * L1 + neuron);
            let value = i16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap());
            weights[feature * L1 + neuron] = f32::from(value) / 255.0;
        }
    }
    let bias_offset = 2 * INPUT * L1;
    let bias = (0..L1)
        .map(|neuron| {
            let offset = bias_offset + 2 * neuron;
            f32::from(i16::from_le_bytes(
                payload[offset..offset + 2].try_into().unwrap(),
            )) / 255.0
        })
        .collect();
    (weights, bias)
}

fn main() {
    let data = PathBuf::from(
        std::env::var("CLASSIC_POLICY_DATA")
            .unwrap_or_else(|_| "data/lc0_nnue/policy_eval_ds_100m".to_string()),
    );
    let eval = PathBuf::from(
        std::env::var("CLASSIC_EVAL").unwrap_or_else(|_| "data/nnue/nets/net-0.cenn".to_string()),
    );
    let rank = env_usize("CLASSIC_POLICY_RANK", 32);
    assert!((1..=64).contains(&rank));
    let batch_size = env_usize("CLASSIC_BATCH", 2048);
    let batches_per_superbatch = env_usize("CLASSIC_BPS", 1000);
    let end_superbatch = env_usize("CLASSIC_SUPERBATCHES", 10);
    let threads = env_usize("CLASSIC_THREADS", 2) as u8;
    let log_rate = env_usize("CLASSIC_LOG_RATE", 100);
    let rows_per_side = env_usize("CLASSIC_ROWS_PER_SIDE", DEFAULT_ROWS_PER_SIDE);
    let policy_moves = env_usize("CLASSIC_POLICY_MOVES", 40);
    assert!((1..=218).contains(&policy_moves));
    let seed = env_usize("CLASSIC_SEED", 0xC1A5_51C0) as u64;
    let learning_rate = env_f32("CLASSIC_LR", 0.001);
    let out_dir =
        std::env::var("CLASSIC_OUT").unwrap_or_else(|_| "classic_policy_ckpt".to_string());
    let net_id =
        std::env::var("CLASSIC_ID").unwrap_or_else(|_| "classic-frozen-policy".to_string());
    std::fs::create_dir_all(&out_dir).expect("create CLASSIC_OUT");

    let (frozen_weights, frozen_bias) = load_frozen_ft(&eval);
    println!(
        "frozen classic policy: eval={} rank={} rows/side={} schedule={}x{}x{}",
        eval.display(),
        rank,
        rows_per_side,
        end_superbatch,
        batches_per_superbatch,
        batch_size
    );

    let inputs = ModelInputs::default()
        .add_sparse("stm", (INPUT, 1), SOURCE_FEATURES)
        .add_sparse("ntm", (INPUT, 1), SOURCE_FEATURES)
        .add_dense("policy_target", (POLICY_DIM, 1))
        .add_dense("policy_mask", (POLICY_DIM, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |datum: &PolicyDatum, _step, (((stm, ntm), target), mask)| {
            let mut count = 0;
            for &feature in &datum.features {
                if feature < 0 {
                    break;
                }
                let feature = feature as usize;
                stm[count] = feature as i32;
                let slot = feature / 64;
                let piece_type = slot % 6;
                let color = slot / 6;
                let square = feature % 64;
                // Normalized corpus positions are always mover/White to move.
                // For the opponent accumulator, White pieces are the enemy and
                // every square is vertically mirrored.
                ntm[count] = (((color == 0) as usize * 6 + piece_type) * 64 + (square ^ 56)) as i32;
                count += 1;
            }
            stm[count..].fill(-1);
            ntm[count..].fill(-1);
            target.fill(0.0);
            // The sparse LC0 support contains the legal move list. Masking
            // everything else makes the objective match runtime, which only
            // asks the factorized head to rank generated legal moves.
            mask.fill(-30.0);
            for i in 0..datum.moves.len() {
                if datum.moves[i] >= 0 {
                    let canonical = datum.moves[i] as usize;
                    target[canonical] += datum.probabilities[i];
                    mask[canonical] = 0.0;
                }
            }
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        move |builder, (((stm, ntm), policy_target), policy_mask)| {
            let ftw = builder.new_weights(
                "frozen_ftw",
                (L1, INPUT),
                InitSettings::Custom(TValue::F32(frozen_weights.clone())),
            );
            let ftb = builder.new_weights(
                "frozen_ftb",
                (L1, 1),
                InitSettings::Custom(TValue::F32(frozen_bias.clone())),
            );
            let trunk = builder.with_no_grad(|| {
                (ftw.matmul(stm) + ftb)
                    .screlu()
                    .concat((ftw.matmul(ntm) + ftb).screlu())
            });
            let position = builder
                .new_affine("pol_pos", 2 * L1, rank)
                .forward(trunk)
                .screlu();
            let logits = builder
                .new_affine("pol_move", rank, POLICY_DIM)
                .forward(position);
            let loss = (logits + policy_mask)
                .softmax_crossentropy_loss(policy_target)
                .reduce_sum_rows()
                .reduce_sum_batch();
            (Some(loss), vec![("policy".to_string(), logits)])
        },
    );

    let weights = ModelWeights::new(&defn, seed);
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
        rows_per_side,
        policy_moves,
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
        lr_schedule: Box::new(move |_: Step| learning_rate),
    };
    train(
        &mut optimiser,
        schedule,
        loader,
        |_, step, loss| {
            if step.batch() % log_rate == 0 {
                println!(
                    "  sb {} batch {} policy_loss={loss:.6}",
                    step.superbatch(),
                    step.batch()
                );
            }
        },
        |_, _| {},
    )
    .expect("frozen policy training");

    let format = [
        SavedFormat::id("pol_posw").transpose(),
        SavedFormat::id("pol_posb"),
        SavedFormat::id("pol_movew").transpose(),
        SavedFormat::id("pol_moveb"),
    ];
    let bytes = optimiser
        .cpu_weights()
        .expect("sync weights")
        .to_quantised_buffer(&format, true)
        .expect("export policy head");
    let path = format!("{out_dir}/{net_id}-factorized-r{rank}-quant.bin");
    std::fs::write(&path, &bytes).expect("write policy checkpoint");
    println!(
        "frozen classic policy checkpoint: {path} ({} bytes)",
        bytes.len()
    );
}
