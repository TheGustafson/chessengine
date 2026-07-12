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
 *   NET0_WDL           wdl proportion                            [default: 0.3]
 */
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

const HIDDEN_SIZE: usize = 256;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let data = std::env::var("NET0_DATA").expect("set NET0_DATA to the train.bin path");
    let out = std::env::var("NET0_OUT").unwrap_or_else(|_| "checkpoints".to_string());
    let net_id = std::env::var("NET0_ID").unwrap_or_else(|_| "net-0".to_string());
    let end_superbatch = env_usize("NET0_SUPERBATCHES", 30);
    let bps = env_usize("NET0_BPS", 1520);
    let batch_size = env_usize("NET0_BATCH", 16_384);
    let threads = env_usize("NET0_THREADS", 4);
    let save_rate = env_usize("NET0_SAVE_RATE", 10);
    let initial_lr = env_f32("NET0_LR", 0.001);
    let wdl_value = env_f32("NET0_WDL", 0.3);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
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
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

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
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: end_superbatch },
        save_rate,
    };

    // bullet has no validation-loss support (value.rs warns it is unimplemented),
    // so `test_set: None`; the held-out signal is SPRT Elo vs PSQT (step 4).
    let settings =
        LocalSettings { threads, test_set: None, output_directory: out.as_str(), batch_queue_size: 64 };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data.as_str()]);

    trainer.run(&schedule, &settings, &data_loader);
}
