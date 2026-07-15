/*
 * NNUE net-0 trainer (P13 step 3; design/11 §3.4.3/§3.4.7, §11).
 *
 * The FIRST real net: a 768->2x256->1 SCReLU value net trained on the current
 * SF-teacher corpus (data/nnue/train.bin), meant only to be strong enough to
 * beat PSQT at step 4 (design/11 §11: "smallest thing that proves the loop").
 * Unlike train_smoke.rs (a pipeline smoke on a 5k corpus), this is the tracked,
 * reproducible recipe for the shipped-provenance net series.
 *
 * Discardable scaffold (D15/D28): a `bullet` example built against the vendored
 * bullet clone. Kept in tools/nnue/ as the reproducible reference; the build
 * copies it into the bullet checkout's examples/ dir (see tools/nnue/TRAINING.md).
 *
 * Arch (pinned, design/11 §3.4.1/§3.4.3):
 *   input   Chess768        (768 = 2 x 6 x 64 perspective features)
 *   net     768 -> 2x256 -> 1, SCReLU, dual perspective, 1 output bucket
 *   quant   QA = 255, QB = 64      (save_format == canonical bullet 1_simple)
 *   scale   400
 *   wdl     0.3  (== doc-11 lambda 0.7 eval weight)
 *
 * THE OUTPUT-WEIGHT CLIP (net-0 precondition, design/11 §3.4.3):
 *   The AVX2 `mullo_epi16` SCReLU kernel is bit-exact vs the scalar reference
 *   ONLY while quantized |out_weight| <= 127, and the loader HARD-REJECTS any
 *   net violating it (NnueError::OutWeightOutOfClip). bullet's AdamW optimiser
 *   clips every weight to float [min_weight, max_weight] each step, and its
 *   DEFAULT is ±1.98 (crates/trainer/src/optimiser/adam.rs: AdamWParams::default
 *   + the `p = min(max(p, WMIN), WMAX)` in the update kernel). ValueTrainerBuilder
 *   constructs the optimiser with `Default::default()` (value/builder.rs), so
 *   this ±1.98 clip is active here. 1.98 * QB(64) = 126.72 -> round 127, so the
 *   default clip GUARANTEES quantized |out_weight| <= 127. netpack.py re-verifies
 *   the packed net before packing; do not weaken the clip.
 *
 * Env knobs (all optional except NET0_DATA):
 *   NET0_DATA          path to train.bin (DirectSequentialDataLoader)  [required]
 *   NET0_OUT           checkpoint output dir           [default: checkpoints]
 *   NET0_ID            net id / checkpoint name prefix          [default: net-0]
 *   NET0_SUPERBATCHES  end superbatch                             [default: 30]
 *   NET0_BPS           batches per superbatch                   [default: 1520]
 *                      (~1 epoch of a 25M corpus at batch 16384)
 *   NET0_BATCH         batch size                             [default: 16384]
 *   NET0_THREADS       host data-prep threads                      [default: 4]
 *   NET0_SAVE_RATE     save every N superbatches                  [default: 10]
 *   NET0_LR            initial learning rate                   [default: 0.001]
 *   NET0_LR_END_SB     cosine-decay horizon       [default: end superbatch]
 *   NET0_WDL           wdl proportion                            [default: 0.3]
 *   NET0_SEED          deterministic parameter seed        [default: 198273612]
 *   NET0_HIDDEN        accumulator width (256/512/1024)              [default: 256]
 *   NET0_INIT          optional 256-wide CENN-v1 evaluator to preserve/expand
 *   NET0_FORMAT        bulletformat | sfbinpack             [default: bulletformat]
 *   NET0_SF_BUFFER_MB  SF-binpack shuffle buffer MiB                 [default: 256]
 */
use bullet_compiler::tensor::TValue;
use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};
use bullet_trainer::model::ModelWeights;
use chessengine::eval::nnue::load;

fn sfbinpack_filter(entry: &loader::sfbinpack::TrainingDataEntry) -> bool {
    use loader::sfbinpack::{MoveType, PieceType};

    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let data = std::env::var("NET0_DATA").expect("set NET0_DATA to the train.bin path");
    let out = std::env::var("NET0_OUT").unwrap_or_else(|_| "checkpoints".to_string());
    let net_id = std::env::var("NET0_ID").unwrap_or_else(|_| "net-0".to_string());
    // Bullet creates the per-superbatch directory, but assumes its configured
    // output parent already exists when it writes the first checkpoint.
    std::fs::create_dir_all(&out).expect("create NET0_OUT");
    let end_superbatch = env_usize("NET0_SUPERBATCHES", 30);
    let bps = env_usize("NET0_BPS", 1520);
    let batch_size = env_usize("NET0_BATCH", 16_384);
    let threads = env_usize("NET0_THREADS", 4);
    let save_rate = env_usize("NET0_SAVE_RATE", 10);
    let initial_lr = env_f32("NET0_LR", 0.001);
    let lr_end = env_usize("NET0_LR_END_SB", end_superbatch);
    assert!(
        lr_end >= end_superbatch,
        "NET0_LR_END_SB must be >= NET0_SUPERBATCHES"
    );
    let wdl_value = env_f32("NET0_WDL", 0.3);
    let seed = env_usize("NET0_SEED", 198_273_612) as u64;
    let hidden_size = env_usize("NET0_HIDDEN", 256);
    assert!(
        matches!(hidden_size, 256 | 512 | 1024),
        "unsupported NET0_HIDDEN"
    );

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .seed(seed)
        .optimiser(AdamW) // AdamWParams::default() => weight clip ±1.98 (see header)
        .inputs(Chess768)
        // Quantised export == the CENN payload our engine loads (design/11 §3.4.3).
        // Layout matches the canonical bullet 1_simple example (no transpose):
        // l0w feature-major, l1w stm-half-first. netpack.py packs this verbatim.
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        // target = wdl * game_result + (1 - wdl) * sigmoid(score / SCALE)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, hidden_size);
            let l1 = builder.new_affine("l1", 2 * hidden_size, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    if let Ok(path) = std::env::var("NET0_INIT") {
        let teacher = load(&path).expect("load NET0_INIT");
        let mut l0w = vec![0.0f32; 768 * hidden_size];
        for feature in 0..768 {
            for neuron in 0..256 {
                l0w[feature * hidden_size + neuron] =
                    f32::from(teacher.ft_weights[feature * 256 + neuron]) / 255.0;
            }
            // New neurons are latent at initialization because their output
            // weights are exactly zero. Give their transformer rows small,
            // deterministic nonzero values so SCReLU does not leave a
            // function-preserving width expansion permanently dead.
            for neuron in 256..hidden_size {
                let mut z = (feature as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (neuron as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z ^= z >> 30;
                z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z ^= z >> 27;
                l0w[feature * hidden_size + neuron] = ((z % 2001) as f32 - 1000.0) / 100_000.0;
            }
        }
        let mut l0b = vec![0.0f32; hidden_size];
        for (dst, &src) in l0b[..256].iter_mut().zip(teacher.ft_bias.iter()) {
            *dst = f32::from(src) / 255.0;
        }
        l0b[256..].fill(0.05);
        let mut l1w = vec![0.0f32; 2 * hidden_size];
        for neuron in 0..256 {
            l1w[neuron] = f32::from(teacher.out_weights[neuron]) / 64.0;
            l1w[hidden_size + neuron] = f32::from(teacher.out_weights[256 + neuron]) / 64.0;
        }
        let mut init = ModelWeights::zeroed(trainer.optimiser.definition());
        assert!(init.set("l0w", TValue::F32(l0w)));
        assert!(init.set("l0b", TValue::F32(l0b)));
        assert!(init.set("l1w", TValue::F32(l1w)));
        assert!(init.set(
            "l1b",
            TValue::F32(vec![teacher.out_bias as f32 / (255.0 * 64.0)]),
        ));
        let init_path = format!("{out}/{net_id}-init.raw");
        init.write_into(&mut std::fs::File::create(&init_path).expect("create init raw"))
            .expect("write init raw");
        trainer
            .optimiser
            .load_weights_from_file(&init_path)
            .expect("install NET0_INIT");
        println!("fine-tune init from {path}");
    }

    let final_lr = initial_lr * 0.3f32.powi(3);
    let schedule = TrainingSchedule {
        net_id: net_id.clone(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: bps,
            start_superbatch: 1,
            end_superbatch,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_value },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: lr_end,
        },
        save_rate,
    };

    // bullet has no validation-loss support (value.rs warns it is unimplemented),
    // so `test_set: None`; the held-out signal is SPRT Elo vs PSQT (step 4).
    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: out.as_str(),
        batch_queue_size: 64,
    };

    let format = std::env::var("NET0_FORMAT").unwrap_or_else(|_| "bulletformat".to_string());
    match format.as_str() {
        "bulletformat" => {
            let data_loader = loader::DirectSequentialDataLoader::new(&[data.as_str()]);
            trainer.run(&schedule, &settings, &data_loader);
        }
        "sfbinpack" => {
            let buffer_mb = env_usize("NET0_SF_BUFFER_MB", 256);
            println!("SF binpack input: buffer={buffer_mb}MiB quiet-filter=standard");
            let data_loader =
                loader::SfBinpackLoader::new(data.as_str(), buffer_mb, threads, sfbinpack_filter);
            trainer.run(&schedule, &settings, &data_loader);
        }
        other => panic!("NET0_FORMAT must be bulletformat or sfbinpack, got {other}"),
    }
}
