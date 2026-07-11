/*
 * NNUE Tier-A pipeline smoke trainer (design/15 §3.2.4-5, §4 step 4).
 *
 * NOT a real net -- this trains a 768->2x256 SCReLU value net on the tiny
 * validated smoke corpus (data/nnue/raw/shard_43_*.txt -> train.bin) purely to
 * prove the gen->convert->train chain runs on this box's GPU and emits a
 * quantised net. Net quality is irrelevant; overfitting the tiny corpus is
 * expected and fine.
 *
 * This file is discardable scaffold (D15/D28): it is a `bullet` example, built
 * against the vendored bullet clone. It is kept in tools/nnue/ only as the
 * reproducible reference for the build; the actual build copies it into the
 * bullet checkout's examples/ dir (see tools/nnue/TRAINING.md).
 *
 * Arch (pinned to design/15 §3.2.4-5 / doc 11 §3.4.3):
 *   input   Chess768        (768 = 2 x 6 x 64 perspective features)
 *   net     768 -> 2x256 -> 1, SCReLU activation, dual perspective
 *   quant   QA = 255, QB = 64
 *   scale   400
 *   wdl     0.3  (== doc-11 lambda 0.7 eval weight; §3.2.5 complementarity)
 *
 * Env knobs (all optional except SMOKE_DATA):
 *   SMOKE_DATA          path to train.bin (DirectSequentialDataLoader)   [required]
 *   SMOKE_OUT           checkpoint output dir            [default: checkpoints]
 *   SMOKE_SUPERBATCHES  end superbatch                                [default: 10]
 *   SMOKE_BPS           batches per superbatch                        [default: 50]
 *   SMOKE_BATCH         batch size                                 [default: 16384]
 *   SMOKE_THREADS       host data-prep threads                         [default: 4]
 *   SMOKE_SAVE_RATE     save every N superbatches                       [default: 5]
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

fn main() {
    let data = std::env::var("SMOKE_DATA").expect("set SMOKE_DATA to the train.bin path");
    let out = std::env::var("SMOKE_OUT").unwrap_or_else(|_| "checkpoints".to_string());
    let end_superbatch = env_usize("SMOKE_SUPERBATCHES", 10);
    let bps = env_usize("SMOKE_BPS", 50);
    let batch_size = env_usize("SMOKE_BATCH", 16_384);
    let threads = env_usize("SMOKE_THREADS", 4);
    let save_rate = env_usize("SMOKE_SAVE_RATE", 5);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        // quantised export our future in-engine inference loads (doc 11 §3.4.3)
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

    let schedule = TrainingSchedule {
        net_id: "nnue_smoke".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: bps,
            start_superbatch: 1,
            end_superbatch,
        },
        // design/15 §3.2.5: bootstrap default --wdl 0.3
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: (end_superbatch / 2).max(1) },
        save_rate,
    };

    let settings =
        LocalSettings { threads, test_set: None, output_directory: out.as_str(), batch_queue_size: 64 };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data.as_str()]);

    trainer.run(&schedule, &settings, &data_loader);
}
