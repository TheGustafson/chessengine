/* King-conditioned Chess768 transformer with the proven classic SCReLU value
 * head.  The 64 perspective-relative king buckets deliberately retain both
 * horizontal orientations: unlike SF HalfKA, this representation can be
 * initialised to a bit-exact expansion of an arbitrary Chess768 evaluator.
 */
use bullet_compiler::tensor::TValue;
use bullet_lib::{
    game::inputs::SparseInputType,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};
use bullet_trainer::model::ModelWeights;
use bulletformat::ChessBoard;
use chessengine::eval::nnue::{feature_index, load};
use chessengine::{Color, Piece, PieceType, Square};
const KING768_DIMS: usize = 64 * 768;

fn king768_index(perspective: Color, piece: Piece, square: Square, king: Square) -> usize {
    king.relative_to(perspective).index() as usize * 768 + feature_index(perspective, piece, square)
}

const QA: i16 = 255;
const QB: i16 = 64;
const SCALE: i32 = 400;

#[derive(Clone, Copy, Debug, Default)]
struct HalfKa;

impl SparseInputType for HalfKa {
    type RequiredDataType = ChessBoard;
    fn num_inputs(&self) -> usize {
        KING768_DIMS
    }
    fn max_active(&self) -> usize {
        32
    }
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut emit: F) {
        let mut kings = [Square::A1; 2];
        for (code, square) in *pos {
            if code & 7 == PieceType::King as u8 {
                let color = Color::ALL[((code >> 3) & 1) as usize];
                kings[color as usize] = Square::new(square);
            }
        }
        for (code, square) in *pos {
            let color = Color::ALL[((code >> 3) & 1) as usize];
            let piece_type = [
                PieceType::Pawn,
                PieceType::Knight,
                PieceType::Bishop,
                PieceType::Rook,
                PieceType::Queen,
                PieceType::King,
            ][(code & 7) as usize];
            let piece = Piece::new(color, piece_type);
            let square = Square::new(square);
            emit(
                king768_index(Color::White, piece, square, kings[Color::White as usize]),
                king768_index(Color::Black, piece, square, kings[Color::Black as usize]),
            );
        }
    }
    fn shorthand(&self) -> String {
        "KingBucket768-49152".into()
    }
    fn description(&self) -> String {
        "64 king buckets x Chess768, classic dual SCReLU head".into()
    }
}

fn usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn f32_env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn sfbinpack_filter(entry: &loader::sfbinpack::TrainingDataEntry) -> bool {
    use loader::sfbinpack::{MoveType, PieceType};

    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn main() {
    let data = std::env::var("HKA_DATA").expect("set HKA_DATA");
    let out = std::env::var("HKA_OUT").unwrap_or_else(|_| "halfka_classic_ckpt".into());
    let id = std::env::var("HKA_ID").unwrap_or_else(|_| "halfka-classic".into());
    // Bullet creates the per-superbatch directory, but assumes its configured
    // output parent already exists when it writes the first checkpoint.
    std::fs::create_dir_all(&out).expect("create HKA_OUT");
    let hidden = usize_env("HKA_HIDDEN", 768);
    assert!(matches!(hidden, 256 | 768), "HKA_HIDDEN must be 256 or 768");
    let end = usize_env("HKA_SUPERBATCHES", 30);
    let bps = usize_env("HKA_BPS", 1520);
    let batch = usize_env("HKA_BATCH", 16_384);
    let threads = usize_env("HKA_THREADS", 4);
    let save_rate = usize_env("HKA_SAVE_RATE", 5);
    let initial_lr = f32_env("HKA_LR", 0.001);
    let lr_end = usize_env("HKA_LR_END_SB", end);
    assert!(lr_end >= end, "HKA_LR_END_SB must be >= HKA_SUPERBATCHES");
    let wdl_value = f32_env("HKA_WDL", 0.3);
    let seed = usize_env("HKA_SEED", 198_273_612) as u64;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .seed(seed)
        .optimiser(AdamW)
        .inputs(HalfKa)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm, ntm| {
            let ft = builder.new_affine("l0", KING768_DIMS, hidden);
            let out = builder.new_affine("l1", 2 * hidden, 1);
            out.forward(ft.forward(stm).screlu().concat(ft.forward(ntm).screlu()))
        });
    if let Ok(path) = std::env::var("HKA_INIT_NET0") {
        assert_eq!(
            hidden, 256,
            "net-0 function-preserving init requires HKA_HIDDEN=256"
        );
        let teacher = load(&path).expect("load HKA_INIT_NET0");
        let mut l0w = vec![0.0f32; KING768_DIMS * hidden];
        for king_bucket in 0..64 {
            for classic in 0..768 {
                for i in 0..hidden {
                    l0w[(king_bucket * 768 + classic) * hidden + i] =
                        f32::from(teacher.ft_weights[classic * hidden + i]) / 255.0;
                }
            }
        }
        let mut init = ModelWeights::zeroed(trainer.optimiser.definition());
        assert!(init.set("l0w", TValue::F32(l0w)));
        assert!(init.set(
            "l0b",
            TValue::F32(
                teacher
                    .ft_bias
                    .iter()
                    .map(|&v| f32::from(v) / 255.0)
                    .collect()
            )
        ));
        assert!(init.set(
            "l1w",
            TValue::F32(
                teacher
                    .out_weights
                    .iter()
                    .map(|&v| f32::from(v) / 64.0)
                    .collect()
            )
        ));
        assert!(init.set(
            "l1b",
            TValue::F32(vec![teacher.out_bias as f32 / (255.0 * 64.0)])
        ));
        let init_path = format!("{out}/{id}-net0-init.raw");
        init.write_into(&mut std::fs::File::create(&init_path).expect("create init raw"))
            .expect("write init raw");
        trainer
            .optimiser
            .load_weights_from_file(&init_path)
            .expect("install net0 init");
        println!("function-preserving HalfKA init from {path}");
    }
    let final_lr = initial_lr * 0.3f32.powi(3);
    let schedule = TrainingSchedule {
        net_id: id,
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: batch,
            batches_per_superbatch: bps,
            start_superbatch: 1,
            end_superbatch: end,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_value },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: lr_end,
        },
        save_rate,
    };
    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: &out,
        batch_queue_size: 64,
    };
    let format = std::env::var("HKA_FORMAT").unwrap_or_else(|_| "bulletformat".into());
    match format.as_str() {
        "bulletformat" => {
            let reader = loader::DirectSequentialDataLoader::new(&[&data]);
            trainer.run(&schedule, &settings, &reader);
        }
        "sfbinpack" => {
            let buffer_mb = usize_env("HKA_SF_BUFFER_MB", 256);
            println!("SF binpack input: buffer={buffer_mb}MiB quiet-filter=standard");
            let reader =
                loader::SfBinpackLoader::new(data.as_str(), buffer_mb, threads, sfbinpack_filter);
            trainer.run(&schedule, &settings, &reader);
        }
        other => panic!("HKA_FORMAT must be bulletformat or sfbinpack, got {other}"),
    }
}
