/* Protected value adaptation: frozen net-0 plus an initially-zero exact-SF
 * HalfKAv2_hm residual.  The loss sees the sum, but gradients and optimiser
 * updates exist only for the residual tensors.  This makes evaluator
 * preservation an architectural invariant rather than a soft loss term.
 */
use bullet_lib::{
    game::inputs::{Chess768, SparseInputType},
    nn::InitSettings,
    value::loader::DirectSequentialDataLoader,
};
use bullet_compiler::tensor::TValue;
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat},
    optimiser::adam::{AdamW, AdamWParams},
    reader::ReadMapLoader,
    run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train_frozen},
};
use bulletformat::ChessBoard;
use chessengine::eval::nnue::{load, sota_features::{HALFKA_DIMS, halfka_index}};
use chessengine::{Color, Piece, PieceType, Square};
use std::collections::BTreeSet;

const CLASSIC_INPUTS: usize = 768;
const HALFKA_INPUTS: usize = HALFKA_DIMS;
const MAX_ACTIVE: usize = 32;
const BASE_HIDDEN: usize = 256;
const RESIDUAL_HIDDEN: usize = 256;
const SCALE: f32 = 400.0;
const QA: i16 = 255;
const QB: i16 = 64;

const PIECE_TYPES: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn halfka_features(pos: &ChessBoard, mut emit: impl FnMut(usize, usize)) {
    let mut kings = [Square::A1; 2];
    let mut pieces = Vec::with_capacity(pos.occ().count_ones() as usize);
    for (code, square) in *pos {
        let color = Color::ALL[((code >> 3) & 1) as usize];
        let piece_type = PIECE_TYPES[(code & 7) as usize];
        let square = Square::new(square);
        if piece_type == PieceType::King { kings[color as usize] = square; }
        pieces.push((Piece::new(color, piece_type), square));
    }
    for (piece, square) in pieces {
        emit(
            halfka_index(Color::White, piece, square, kings[Color::White as usize]),
            halfka_index(Color::Black, piece, square, kings[Color::Black as usize]),
        );
    }
}

fn assert_frozen(weights: &ModelWeights, expected: &[(String, Vec<f32>)]) {
    for (name, values) in expected {
        assert_eq!(weights.get(name).values.f32(), values, "frozen tensor drifted: {name}");
    }
}

fn main() {
    let data = std::env::var("RES_DATA").expect("set RES_DATA");
    let base_path = std::env::var("RES_BASE").expect("set RES_BASE to net-0 CENN");
    let out_dir = std::env::var("RES_OUT").unwrap_or_else(|_| "halfka_residual_ckpt".into());
    let net_id = std::env::var("RES_ID").unwrap_or_else(|_| "halfka-residual".into());
    let end = env_usize("RES_SUPERBATCHES", 10);
    let bps = env_usize("RES_BPS", 282);
    let batch = env_usize("RES_BATCH", 16_384);
    let threads = env_usize("RES_THREADS", 4) as u8;
    let save_rate = env_usize("RES_SAVE_RATE", 2);
    let initial_lr = env_f32("RES_LR", 0.0001);
    let wdl = env_f32("RES_WDL", 0.3);
    let seed = env_usize("RES_SEED", 0xA4C0_0001) as u64;
    assert!((0.0..=1.0).contains(&wdl));
    std::fs::create_dir_all(&out_dir).expect("create RES_OUT");

    let inputs = ModelInputs::default()
        .add_sparse("classic_stm", (CLASSIC_INPUTS, 1), MAX_ACTIVE)
        .add_sparse("classic_ntm", (CLASSIC_INPUTS, 1), MAX_ACTIVE)
        .add_sparse("halfka_stm", (HALFKA_INPUTS, 1), MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_INPUTS, 1), MAX_ACTIVE)
        .add_dense("targets", (1, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |pos: &ChessBoard, _step, ((((classic_stm, classic_ntm), hk_stm), hk_ntm), targets)| {
            let mut classic_count = 0;
            Chess768.map_features(pos, |stm, ntm| {
                classic_stm[classic_count] = stm as i32;
                classic_ntm[classic_count] = ntm as i32;
                classic_count += 1;
            });
            classic_stm[classic_count..].fill(-1);
            classic_ntm[classic_count..].fill(-1);

            let mut hk_count = 0;
            halfka_features(pos, |stm, ntm| {
                hk_stm[hk_count] = stm as i32;
                hk_ntm[hk_count] = ntm as i32;
                hk_count += 1;
            });
            hk_stm[hk_count..].fill(-1);
            hk_ntm[hk_count..].fill(-1);

            let eval = sigmoid(f32::from(pos.score) / SCALE);
            let result = f32::from(pos.result) / 2.0;
            targets[0] = wdl * result + (1.0 - wdl) * eval;
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        |builder, ((((classic_stm, classic_ntm), hk_stm), hk_ntm), targets)| {
            let base_l0 = builder.new_affine("base_l0", CLASSIC_INPUTS, BASE_HIDDEN);
            let base_l1 = builder.new_affine("base_l1", 2 * BASE_HIDDEN, 1);
            let res_l0 = builder.new_affine("res_l0", HALFKA_INPUTS, RESIDUAL_HIDDEN);
            res_l0.init_with_effective_input_size(MAX_ACTIVE);
            let res_l1w = builder.new_weights(
                "res_l1w", (1, 2 * RESIDUAL_HIDDEN), InitSettings::Zeroed,
            );
            let res_l1b = builder.new_weights("res_l1b", (1, 1), InitSettings::Zeroed);

            let base_hidden = base_l0.forward(classic_stm).screlu()
                .concat(base_l0.forward(classic_ntm).screlu());
            let residual_hidden = res_l0.forward(hk_stm).screlu()
                .concat(res_l0.forward(hk_ntm).screlu());
            let base = base_l1.forward(base_hidden);
            let residual = res_l1w.matmul(residual_hidden) + res_l1b;
            let output = base + residual;
            let loss = output.sigmoid().squared_error(targets).reduce_sum_batch();
            (Some(loss), vec![("output".to_string(), output)])
        },
    );

    let base = load(&base_path).expect("load frozen classic evaluator");
    let mut weights = ModelWeights::new(&defn, seed);
    assert!(weights.set(
        "base_l0w",
        TValue::F32(base.ft_weights.iter().map(|&x| f32::from(x) / f32::from(QA)).collect()),
    ));
    assert!(weights.set(
        "base_l0b",
        TValue::F32(base.ft_bias.iter().map(|&x| f32::from(x) / f32::from(QA)).collect()),
    ));
    assert!(weights.set(
        "base_l1w",
        TValue::F32(base.out_weights.iter().map(|&x| f32::from(x) / f32::from(QB)).collect()),
    ));
    assert!(weights.set(
        "base_l1b",
        TValue::F32(vec![base.out_bias as f32 / f32::from(QA * QB)]),
    ));

    let frozen_names = ["base_l0w", "base_l0b", "base_l1w", "base_l1b"]
        .into_iter().map(str::to_string).collect::<BTreeSet<_>>();
    let frozen_expected = frozen_names.iter().map(|name| {
        (name.clone(), weights.get(name).values.f32().to_vec())
    }).collect::<Vec<_>>();

    let device = DefaultDevice::new(0).expect("device");
    let mut optimiser = AdamW::new(defn, weights, device, AdamWParams::default()).expect("optimiser");
    let reader = DirectSequentialDataLoader::new(&[data.as_str()]);
    let loader = ReadMapLoader::new(reader, mapper, threads);
    let final_lr = initial_lr * 0.3f32.powi(3);
    let lr_schedule: Box<dyn Fn(Step) -> f32> = Box::new(move |step| {
        if end <= 1 { return initial_lr; }
        let t = (step.superbatch() as f32 - 1.0) / (end as f32 - 1.0);
        final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + (std::f32::consts::PI * t).cos())
    });
    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size: batch, batches_per_superbatch: bps,
                               start_superbatch: 1, end_superbatch: end },
        log_rate: 32.min(bps),
        lr_schedule,
    };
    let residual_format = vec![
        SavedFormat::id("res_l0w").round().quantise::<i16>(QA),
        SavedFormat::id("res_l0b").round().quantise::<i16>(QA),
        SavedFormat::id("res_l1w").round().quantise::<i16>(QB),
        SavedFormat::id("res_l1b").round().quantise::<i16>(QA * QB),
    ];

    println!("protected base={base_path}; training exact-HalfKA residual only");
    let save_out = out_dir.clone();
    let save_id = net_id.clone();
    let expected = frozen_expected.clone();
    train_frozen(
        &mut optimiser,
        schedule,
        loader,
        &frozen_names,
        |_, _, _| {},
        move |opt, step| {
            let sb = step.superbatch();
            if sb % save_rate == 0 || sb == end {
                let current = opt.cpu_weights().expect("sync weights");
                assert_frozen(&current, &expected);
                let buf = current.to_quantised_buffer(&residual_format, true)
                    .expect("quantise residual checkpoint");
                let path = format!("{save_out}/{save_id}-sb{sb}-quant.bin");
                std::fs::write(&path, &buf).expect("write residual checkpoint");
                println!("saved {path}; frozen base exact");
                if sb == end {
                    let raw_path = format!("{save_out}/{save_id}-final-raw.bin");
                    let mut raw = std::fs::File::create(&raw_path).expect("create raw");
                    current.write_into(&mut raw).expect("write raw");
                }
            }
        },
    ).expect("protected residual training");
}
