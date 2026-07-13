/*
 * D40 PRODUCTION eval-trunk trainer (task #28, design/11 §12.3/§12.6/§12.7).
 * Scratchpad-only scaffold (NOT a tracked engine recipe; discardable, D15/D28).
 *
 * The first HalfKA-shape D40-tail EVAL net: the native bullet trunk from the
 * feasibility spike (examples/d40_smoke.rs), promoted to a production recipe
 * with the two D40-specific details the smoke omitted:
 *
 *   1. the CUSTOM non-uniform output-bucket layout (design/11 §12.3
 *      `OUTPUT_BUCKETS_LAYOUT[popcount]`), impl'd via a custom
 *      `OutputBuckets::bucket()` (the smoke used the uniform `MaterialCount<8>`);
 *   2. the D40 accumulator width 768 (§12.2) so the pairwise trunk yields the
 *      exact 768 transformed features (384/perspective, STM-half first).
 *
 * Arch (design/11 §12.1-§12.3, native-bullet analogue):
 *   input   ChessBucketsMirrored (10 king buckets)  == closest native HalfKAv2_hm
 *   FT      inputs -> 768 (i16), .crelu().pairwise_mul()  == D40 pairwise trunk
 *   concat  STM-half first -> 768 transformed features
 *   tail    per output bucket: L1(768->16) screlu -> L2(16->32) screlu -> L3(32->1)
 *   buckets 8, CUSTOM OUTPUT_BUCKETS_LAYOUT (non-uniform, §12.3)
 *   quant   FT i16(255), L1 i8(64), L2/L3 Float          == D40 lean hybrid (§12.6)
 *   scale   400, wdl 0.3 (== doc-11 lambda 0.7 eval weight)
 *
 * NOTE vs the real D40 net: this uses bullet's native king-bucketed 768 input
 * (a HalfKA *analogue*), NOT the SF-exact HalfKAv2_hm + FullThreats feature set
 * (that is the custom SparseInputType scaffolded in examples/d40_features.rs +
 * the engine index math of task #29). This example proves the *eval trunk + tail
 * + custom-bucket + lean-quant* pipeline yields a usable net end to end.
 *
 * THE OUTPUT-WEIGHT CLIP (net-0 precondition, design/11 §3.4.3, carried to D40):
 *   bullet's AdamW default clips every weight to float [-1.98, 1.98] each step;
 *   1.98 * QB(64) = 126.72 -> round 127, so quantized |L1_weight| <= 127 (i8) is
 *   GUARANTEED and the i8/i16 save cannot overflow (save hard-checks anyway).
 *
 * Env knobs (all optional except D40_DATA):
 *   D40_DATA          path to train.bin (DirectSequentialDataLoader)   [required]
 *   D40_OUT           checkpoint output dir                    [default: d40_eval_ckpt]
 *   D40_ID            net id / checkpoint name prefix              [default: d40-eval-0]
 *   D40_SB            end superbatch                                    [default: 30]
 *   D40_BPS           batches per superbatch                          [default: 1520]
 *   D40_BATCH         batch size                                    [default: 16384]
 *   D40_HL            FT accumulator width, pre-pairwise (even)        [default: 768]
 *   D40_THREADS       host data-prep threads                             [default: 4]
 *   D40_SAVE_RATE     save every N superbatches                         [default: 10]
 *   D40_LR            initial learning rate                          [default: 0.001]
 *   D40_WDL           wdl proportion                                   [default: 0.3]
 */
use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::OutputBuckets,
    },
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};
use bulletformat::ChessBoard;

const SCALE: i32 = 400;
const QA: i16 = 255; // FT scale
const QB: i16 = 64; // L1 scale

// --- CUSTOM D40 output-bucket layout (design/11 §12.3) -----------------------
// OUTPUT_BUCKETS_LAYOUT[popcount(occ)] = [0x9, 1x4, 2x4, 3x3, 4x3, 5x3, 6x3, 7x4]
// Non-uniform: fine at mid-material, coarse at the extremes. 33 entries cover
// popcount 0..=32 (a legal board has 2..=32 pieces). "Copy the shape, retune."
#[rustfmt::skip]
const OUTPUT_BUCKETS_LAYOUT: [u8; 33] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, //  0..8   (9) -> bucket 0
    1, 1, 1, 1,                //  9..12  (4) -> bucket 1
    2, 2, 2, 2,                // 13..16  (4) -> bucket 2
    3, 3, 3,                   // 17..19  (3) -> bucket 3
    4, 4, 4,                   // 20..22  (3) -> bucket 4
    5, 5, 5,                   // 23..25  (3) -> bucket 5
    6, 6, 6,                   // 26..28  (3) -> bucket 6
    7, 7, 7, 7,                // 29..32  (4) -> bucket 7
];

#[derive(Clone, Copy, Default)]
struct D40OutputBuckets;

impl OutputBuckets<ChessBoard> for D40OutputBuckets {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let pieces = pos.occ().count_ones() as usize;
        OUTPUT_BUCKETS_LAYOUT[pieces.min(32)]
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let data = std::env::var("D40_DATA").expect("set D40_DATA to a bulletformat train.bin");
    let out = std::env::var("D40_OUT").unwrap_or_else(|_| "d40_eval_ckpt".to_string());
    let net_id = std::env::var("D40_ID").unwrap_or_else(|_| "d40-eval-0".to_string());
    let end_superbatch = env_usize("D40_SB", 30);
    let bps = env_usize("D40_BPS", 1520);
    let batch_size = env_usize("D40_BATCH", 16_384);
    let hl_size = env_usize("D40_HL", 768); // pre-pairwise accumulator width (even)
    let threads = env_usize("D40_THREADS", 4);
    let save_rate = env_usize("D40_SAVE_RATE", 10);
    let initial_lr = env_f32("D40_LR", 0.001);
    let wdl_value = env_f32("D40_WDL", 0.3);

    assert!(hl_size % 2 == 0, "FT accumulator width must be even (pairwise_mul halves it)");

    const NUM_OUTPUT_BUCKETS: usize = 8;
    const L2: usize = 16; // L1 output width (design/11 §12.3: L1 768->16)
    // Input king-bucket layout: 10 buckets (mirrored 32-entry -> 64), the closest
    // native analogue of HalfKAv2_hm's king bucketing.
    #[rustfmt::skip]
    const KING_BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
        9, 9, 9, 9,
        9, 9, 9, 9,
    ];
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&KING_BUCKET_LAYOUT);
    let inputs = 768 * NUM_INPUT_BUCKETS;

    println!(
        "D40 eval trunk: inputs={inputs} ({NUM_INPUT_BUCKETS} king buckets) FT={hl_size} \
         transformed={hl_size} tail=768->{L2}->32->1 x {NUM_OUTPUT_BUCKETS} custom buckets"
    );

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW) // AdamWParams::default() => weight clip +/-1.98 (i8 L1 safe)
        .inputs(ChessBucketsMirrored::new(KING_BUCKET_LAYOUT))
        // CUSTOM non-uniform output buckets (design/11 §12.3) -- the D40 detail.
        .output_buckets(D40OutputBuckets)
        // D40 lean hybrid quant (§12.6): FT i16, L1 i8, L2/L3 f32 (Float).
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(QB),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        // target = wdl * game_result + (1 - wdl) * sigmoid(score / SCALE)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(move |builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", inputs, hl_size);
            // CRITICAL for king-bucketed sparse inputs: initialise the FT weights
            // for the EFFECTIVE active-feature count (~32), not the 7680 nominal
            // input dim -- else stdev sqrt(2/7680)=0.016 vanishes the FT signal and
            // the net never learns (flat loss). The near-D40 template
            // examples/progression/4_multi_layer.rs does exactly this.
            l0.init_with_effective_input_size(32);
            let l1 = builder.new_affine("l1", hl_size, NUM_OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2, NUM_OUTPUT_BUCKETS * 32);
            let l3 = builder.new_affine("l3", 32, NUM_OUTPUT_BUCKETS);

            // D40 pairwise-mul FT trunk (§12.3): crelu then halve each perspective.
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden); // 768 transformed features

            // Per-output-bucket lean tail: int8 L1 (sparse trunk) then f32 L2/L3.
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
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

    let settings =
        LocalSettings { threads, test_set: None, output_directory: out.as_str(), batch_queue_size: 64 };

    let data_loader = DirectSequentialDataLoader::new(&[data.as_str()]);

    trainer.run(&schedule, &settings, &data_loader);
}
