/*
 * D40 custom-feature scaffold (task #28 part 2; design/11 §12.1/§12.6/§12.7).
 * Scratchpad-only scaffold (discardable, D15/D28).
 *
 * TWO deliverables, both compiling + running on a real bulletformat corpus:
 *
 *  A. `HalfKaThreats`: the custom `SparseInputType` over the CONCATENATED
 *     0..83,248 index space (HalfKAv2_hm 22,528 in [0,22528) + FullThreats 60,720
 *     in [22528,83248)). It calls the engine's canonical, unit-tested
 *     `halfka_index` and `threat_index` functions directly. Only enumeration is
 *     local, and startup parity-checks its exact index sets against the engine's
 *     canonical enumerators over fixtures plus real corpus records.
 *
 *  B. The TWO-summed-tensor FT graph (design/11 §12.1/§12.6): the HalfKA weights
 *     (i16) and threat weights (i8) live in SEPARATE affine tensors that are
 *     summed in the graph (`node + node`) into one accumulator, so they (1)
 *     quantize separately (i16 / i8 -- the D40 lean hybrid) and (2) match the
 *     inference head's SEPARATE pst/threat accumulators (§12.3). Built on the
 *     low-level `ModelDefinition` + `AdamW::new` + `run::train` path (the same
 *     path `ValueTrainer` builds internally; see examples/tests/test4.rs).
 *
 * This is the real C2 feature front end; trainer and inference index math share
 * one implementation through the chessengine path dependency.
 */
use bullet_compiler::tensor::TValue;
use bullet_lib::{
    game::inputs::SparseInputType,
    nn::InitSettings,
    value::loader::{self, DirectSequentialDataLoader},
};
use bullet_trainer::{
    model::{
        ModelDefinition, ModelEvaluator, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat,
    },
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train},
};
use bulletformat::ChessBoard;
use chessengine::attacks::{attacks, pawn_attacks};
use chessengine::eval::nnue::sota_features::{
    HALFKA_DIMS, THREAT_DIMS, for_each_halfka, for_each_threat, halfka_index, threat_index,
};
use chessengine::{Bitboard, Board, Color, Piece, PieceType, Square};
use std::sync::OnceLock;

type SfReader = loader::SfBinpackLoader<fn(&loader::sfbinpack::TrainingDataEntry) -> bool>;

#[derive(Clone)]
enum D40Reader {
    Bullet(DirectSequentialDataLoader),
    Sf(SfReader),
}

impl DataReader<ChessBoard> for D40Reader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip_count: usize, f: F) {
        match self {
            Self::Bullet(reader) => DataReader::<ChessBoard>::read_chunks(reader, skip_count, f),
            Self::Sf(reader) => DataReader::<ChessBoard>::read_chunks(reader, skip_count, f),
        }
    }
}

/// Official-style random position skip (decorrelates game-contiguous
/// binpack streams). `D40_RANDOM_SKIP_PCT` (0-99, default 0 = historical
/// behavior). Deterministic FNV over occupancy/ply/score, same construction
/// as train_net0's NET0_RANDOM_SKIP_PCT.
static RANDOM_SKIP_PCT: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    let pct = std::env::var("D40_RANDOM_SKIP_PCT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    assert!(pct < 100, "D40_RANDOM_SKIP_PCT must be 0-99");
    if pct > 0 {
        println!("SF binpack random skip: {pct}%");
    }
    pct
});

fn position_hash(entry: &loader::sfbinpack::TrainingDataEntry) -> u64 {
    use loader::sfbinpack::Color;
    let mut h = 0xcbf29ce484222325u64;
    for v in [
        entry.pos.pieces_bb(Color::White).bits(),
        entry.pos.pieces_bb(Color::Black).bits(),
        u64::from(entry.ply),
        entry.score as u64,
    ] {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn sfbinpack_filter(entry: &loader::sfbinpack::TrainingDataEntry) -> bool {
    use loader::sfbinpack::{MoveType, PieceType};

    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        && (*RANDOM_SKIP_PCT == 0 || position_hash(entry) % 100 >= *RANDOM_SKIP_PCT)
}

// --- Feature-space layout (design/11 §12.1) ---------------------------------
const HALFKA_INPUTS: usize = HALFKA_DIMS;
const THREAT_INPUTS: usize = THREAT_DIMS;
const TOTAL_INPUTS: usize = HALFKA_INPUTS + THREAT_INPUTS; // 83_248
const HALFKA_MAX_ACTIVE: usize = 32; // <=32 HalfKA features / perspective
const THREAT_MAX_ACTIVE: usize = 160;
const MAX_ACTIVE: usize = HALFKA_MAX_ACTIVE + THREAT_MAX_ACTIVE; // 192
const DEFAULT_SCALE: f32 = 400.0;
const DEFAULT_WDL: f32 = 0.3;
const NUM_BUCKETS: usize = 8;
const L2: usize = 16;
const L3: usize = 32;
const HL: usize = 768;

fn halfka_only() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| env_usize("D40_HALFKA_ONLY", 0) != 0)
}

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

const PIECE_TYPES: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

struct Decoded {
    mailbox: [Option<(Color, PieceType)>; 64],
    pieces: [[Bitboard; 6]; 2],
    occupancy: Bitboard,
    kings: [Square; 2],
}

fn decode(pos: &ChessBoard) -> Decoded {
    let mut mailbox = [None; 64];
    let mut pieces = [[Bitboard::EMPTY; 6]; 2];
    let mut kings = [Square::A1; 2];

    for (code, sq) in *pos {
        let color = Color::ALL[((code >> 3) & 1) as usize];
        let piece_type = PIECE_TYPES[(code & 7) as usize];
        let square = Square::new(sq);
        mailbox[sq as usize] = Some((color, piece_type));
        pieces[color as usize][piece_type as usize] |= Bitboard::from_square(square);
        if piece_type == PieceType::King {
            kings[color as usize] = square;
        }
    }

    Decoded {
        mailbox,
        pieces,
        occupancy: Bitboard::from_u64(pos.occ()),
        kings,
    }
}

fn emit_features<F: FnMut(usize, usize)>(pos: &ChessBoard, mut f: F) {
    let decoded = decode(pos);
    let [white_king, black_king] = decoded.kings;

    for square in decoded.occupancy {
        let (color, piece_type) =
            decoded.mailbox[square.index() as usize].expect("occupied square must contain a piece");
        let piece = Piece::new(color, piece_type);
        f(
            halfka_index(Color::White, piece, square, white_king),
            halfka_index(Color::Black, piece, square, black_king),
        );
    }

    if halfka_only() {
        return;
    }

    let both = |piece_type: PieceType| {
        decoded.pieces[Color::White as usize][piece_type as usize]
            | decoded.pieces[Color::Black as usize][piece_type as usize]
    };
    let all_pawns = both(PieceType::Pawn);
    let pawn_targets = all_pawns | both(PieceType::Knight) | both(PieceType::Rook);
    let minor_slider_targets = pawn_targets | both(PieceType::Bishop);
    let queen_targets = minor_slider_targets | both(PieceType::Queen);

    let mut emit_threat =
        |attacker_color: Color, attacker_type: PieceType, from: Square, to: Square| {
            let attacker = Piece::new(attacker_color, attacker_type);
            let (victim_color, victim_type) =
                decoded.mailbox[to.index() as usize].expect("threat target must contain a piece");
            let victim = Piece::new(victim_color, victim_type);

            if let Some(white) = threat_index(Color::White, attacker, from, to, victim, white_king)
            {
                let black = threat_index(Color::Black, attacker, from, to, victim, black_king)
                    .or_else(|| threat_index(Color::Black, victim, to, from, attacker, black_king))
                    .expect("a White-active threat must have a Black-perspective partner");
                f(HALFKA_INPUTS + white, HALFKA_INPUTS + black);
            }
        };

    for color in Color::ALL {
        for from in decoded.pieces[color as usize][PieceType::Pawn as usize] {
            for to in pawn_attacks(color, from) & pawn_targets {
                emit_threat(color, PieceType::Pawn, from, to);
            }
            if let Some(to) = from.shifted(color.pawn_push())
                && all_pawns.contains(to)
            {
                emit_threat(color, PieceType::Pawn, from, to);
            }
        }

        for piece_type in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let targets = if matches!(piece_type, PieceType::Knight | PieceType::Queen) {
                queen_targets
            } else {
                minor_slider_targets
            };
            for from in decoded.pieces[color as usize][piece_type as usize] {
                for to in attacks(piece_type, from, decoded.occupancy) & targets {
                    emit_threat(color, piece_type, from, to);
                }
            }
        }
    }
}

// ============================================================================
// A. Custom SparseInputType: HalfKAv2_hm + FullThreats over 0..83,248.
// ============================================================================
#[derive(Clone, Copy, Debug, Default)]
pub struct HalfKaThreats;

impl SparseInputType for HalfKaThreats {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TOTAL_INPUTS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE
    }

    /// Emit (stm_index, ntm_index) pairs over the concatenated index space.
    ///
    /// Each emitted pair is entirely within ONE subspace: a HalfKA pair has both
    /// indices in [0, HALFKA_INPUTS); a threat pair has both in
    /// [HALFKA_INPUTS, TOTAL_INPUTS). The downstream mapper routes on that.
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        emit_features(pos, f);
    }

    fn shorthand(&self) -> String {
        format!("HalfKA+Threats({TOTAL_INPUTS})")
    }

    fn description(&self) -> String {
        "SF HalfKAv2_hm (22528) + FullThreats (60720), two separately-quantised FT tensors"
            .to_string()
    }
}

// ============================================================================
// A parsed fixture used only for the feature-space structural self-check below.
// Training data always comes from D40_DATA through bullet's supported direct
// bulletformat loader.
// ============================================================================
fn self_check_position() -> ChessBoard {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5"
        .parse()
        .expect("valid self-check position")
}

fn reconstruct_board(pos: &ChessBoard) -> Board {
    let decoded = decode(pos);
    let mut board = Board::empty();
    for square in decoded.occupancy {
        let (color, piece_type) =
            decoded.mailbox[square.index() as usize].expect("occupied square must contain a piece");
        board.put_piece(Piece::new(color, piece_type), square);
    }
    board
}

fn verify_feature_parity(reader: &D40Reader, requested_rows: usize) {
    assert!(requested_rows > 0, "D40_VERIFY_ROWS must be positive");
    let mut max_halfka = 0usize;
    let mut max_threats = 0usize;
    let mut checked = 0usize;
    let mut check = |pos: &ChessBoard| {
        let board = reconstruct_board(pos);
        let mut trainer = [Vec::<usize>::new(), Vec::<usize>::new()];
        emit_features(pos, |stm, ntm| {
            trainer[Color::White as usize].push(stm);
            trainer[Color::Black as usize].push(ntm);
        });

        for perspective in Color::ALL {
            let mut engine = Vec::new();
            for_each_halfka(&board, perspective, |idx| engine.push(idx));
            if !halfka_only() {
                for_each_threat(&board, perspective, |idx| engine.push(HALFKA_INPUTS + idx));
            }

            let actual = &mut trainer[perspective as usize];
            actual.sort_unstable();
            engine.sort_unstable();
            assert_eq!(
                *actual, engine,
                "trainer/engine feature mismatch at sampled position {checked}, {perspective:?}"
            );

            let halfka = actual.partition_point(|&idx| idx < HALFKA_INPUTS);
            max_halfka = max_halfka.max(halfka);
            max_threats = max_threats.max(actual.len() - halfka);
        }
        checked += 1;
    };

    check(&self_check_position());
    let target = requested_rows;
    let mut corpus_checked = 0usize;
    reader.read_chunks(0, |chunk: &[ChessBoard]| {
        let take = chunk.len().min(target - corpus_checked);
        for pos in &chunk[..take] {
            check(pos);
        }
        corpus_checked += take;
        corpus_checked == target
    });
    assert_eq!(corpus_checked, target, "feature scan ended early");

    assert!(
        max_halfka <= HALFKA_MAX_ACTIVE,
        "HalfKA sparse capacity is too small"
    );
    assert!(
        max_threats <= THREAT_MAX_ACTIVE,
        "threat sparse capacity is too small"
    );
    println!(
        "feature parity: {} positions exact; max active HalfKA={max_halfka}, threats={max_threats}",
        checked
    );
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

fn load_raw_weights(weights: &mut ModelWeights, path: &str) {
    let file = std::fs::File::open(path).expect("open D40_PROBE_RAW");
    weights.load_from(file).expect("load named raw tensors");
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

fn load_quantised_weights(weights: &mut ModelWeights, path: &str) {
    let raw = std::fs::read(path).expect("read D40_PROBE_QUANT");
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
    let pst = i16s(HALFKA_INPUTS * HL, 255.0);
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
    let threat = i8s(THREAT_INPUTS * HL, 64.0);
    let l1_saved = i8s(NUM_BUCKETS * L2 * HL, 64.0);
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
    let l1_bias = f32s(NUM_BUCKETS * L2);
    let l2_saved = f32s(NUM_BUCKETS * L3 * L2);
    let l2_bias = f32s(NUM_BUCKETS * L3);
    let l3_saved = f32s(NUM_BUCKETS * L3);
    let l3_bias = f32s(NUM_BUCKETS);
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

    let tensors = [
        ("l0_pstw", pst),
        ("l0_pstb", pst_bias),
        ("l0_threatw", threat),
        ("l1w", transpose_saved(NUM_BUCKETS * L2, HL, l1_saved)),
        ("l1b", l1_bias),
        ("l2w", transpose_saved(NUM_BUCKETS * L3, L2, l2_saved)),
        ("l2b", l2_bias),
        ("l3w", transpose_saved(NUM_BUCKETS, L3, l3_saved)),
        ("l3b", l3_bias),
    ];
    for (id, values) in tensors {
        assert!(
            weights.set(id, TValue::F32(values)),
            "quant tensor shape {id}"
        );
    }
}

fn main() {
    let data = std::env::var("D40_DATA")
        .expect("set D40_DATA to a bulletformat ChessBoard train.bin path");
    let feats = HalfKaThreats;
    let format = std::env::var("D40_FORMAT").unwrap_or_else(|_| "bulletformat".into());
    let threads = env_usize("D40_THREADS", 6) as u8;
    let reader = match format.as_str() {
        "bulletformat" => D40Reader::Bullet(DirectSequentialDataLoader::new(&[data.as_str()])),
        "sfbinpack" => {
            let buffer_mb = env_usize("D40_SF_BUFFER_MB", 256);
            println!("SF binpack input: buffer={buffer_mb}MiB quiet-filter=standard");
            D40Reader::Sf(loader::SfBinpackLoader::new(
                data.as_str(),
                buffer_mb,
                usize::from(threads),
                sfbinpack_filter,
            ))
        }
        other => panic!("D40_FORMAT must be bulletformat or sfbinpack, got {other}"),
    };
    println!("=== D40 feature scaffold ===");
    println!("input type   : {}", feats.description());
    println!(
        "num_inputs   : {} (expected {TOTAL_INPUTS})",
        feats.num_inputs()
    );
    println!(
        "max_active   : {} (expected {MAX_ACTIVE})",
        feats.max_active()
    );
    assert_eq!(feats.num_inputs(), TOTAL_INPUTS);
    assert_eq!(feats.max_active(), MAX_ACTIVE);
    let verify_rows = env_usize("D40_VERIFY_ROWS", 64);
    verify_feature_parity(&reader, verify_rows);
    if env_usize("D40_VERIFY_ONLY", 0) != 0 {
        println!("D40_VERIFY_ONLY complete");
        return;
    }

    // Scaffold self-check: features split cleanly into the two subspaces.
    let pos = self_check_position();
    let (mut n_hk, mut n_th) = (0usize, 0usize);
    feats.map_features(&pos, |stm, ntm| {
        assert!(
            stm < TOTAL_INPUTS && ntm < TOTAL_INPUTS,
            "index out of range"
        );
        let stm_is_hk = stm < HALFKA_INPUTS;
        let ntm_is_hk = ntm < HALFKA_INPUTS;
        assert_eq!(stm_is_hk, ntm_is_hk, "a pair must lie in ONE subspace");
        if stm_is_hk { n_hk += 1 } else { n_th += 1 }
    });
    println!("startpos     : {n_hk} HalfKA + {n_th} threat features active");
    assert!(n_hk <= HALFKA_MAX_ACTIVE && n_th <= THREAT_MAX_ACTIVE);

    // ------------------------------------------------------------------
    // B. Two-summed-tensor FT graph (low-level path).
    // ------------------------------------------------------------------
    let out_dir = std::env::var("D40_OUT").unwrap_or_else(|_| "d40_feat_ckpt".to_string());
    let net_id = std::env::var("D40_ID").unwrap_or_else(|_| "d40-feat-0".to_string());
    let start_superbatch = env_usize("D40_START_SUPERBATCH", 1);
    let end_superbatch = env_usize("D40_SUPERBATCHES", 30);
    let batches_per_superbatch = env_usize("D40_BPS", 1520);
    let batch_size = env_usize("D40_BATCH", 16_384);
    let save_rate = env_usize("D40_SAVE_RATE", 10);
    let initial_lr = env_f32("D40_LR", 0.001);
    let lr_end_superbatch = env_usize("D40_LR_END_SB", end_superbatch);
    assert!(
        (1..=end_superbatch).contains(&start_superbatch),
        "D40_START_SUPERBATCH must be in 1..=D40_SUPERBATCHES"
    );
    assert!(
        lr_end_superbatch >= end_superbatch,
        "D40_LR_END_SB must be >= D40_SUPERBATCHES"
    );
    let target_scale = env_f32("D40_EVAL_TARGET_SCALE", DEFAULT_SCALE);
    let wdl = env_f32("D40_WDL", DEFAULT_WDL);
    let score_multiplier = env_f32("D40_SCORE_MULTIPLIER", 1.0);
    assert!(target_scale > 0.0 && target_scale.is_finite());
    assert!((0.0..=1.0).contains(&wdl));
    assert!(score_multiplier > 0.0 && score_multiplier.is_finite());
    std::fs::create_dir_all(&out_dir).expect("create D40_OUT directory");

    // Four sparse inputs: HalfKA (i16 tensor) + threats (i8 tensor), each stm/ntm.
    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (HALFKA_INPUTS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_INPUTS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("threat_stm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("buckets", (NUM_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |pos: &ChessBoard,
              _step,
              (((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), targets)| {
            let (mut hk, mut th) = (0usize, 0usize);
            feats.map_features(pos, |stm, ntm| {
                if stm < HALFKA_INPUTS {
                    assert!(
                        hk < HALFKA_MAX_ACTIVE,
                        "HalfKA sparse capacity exceeded: position has more than \
                         {HALFKA_MAX_ACTIVE} active features"
                    );
                    hk_stm[hk] = stm as i32;
                    hk_ntm[hk] = ntm as i32;
                    hk += 1;
                } else {
                    assert!(
                        th < THREAT_MAX_ACTIVE,
                        "FullThreats sparse capacity exceeded: position has more than \
                         {THREAT_MAX_ACTIVE} active features"
                    );
                    th_stm[th] = (stm - HALFKA_INPUTS) as i32;
                    th_ntm[th] = (ntm - HALFKA_INPUTS) as i32;
                    th += 1;
                }
            });
            // -1 sentinel fills the remainder of every sparse stream (skip marker).
            hk_stm[hk..].iter_mut().for_each(|s| *s = -1);
            hk_ntm[hk..].iter_mut().for_each(|s| *s = -1);
            th_stm[th..].iter_mut().for_each(|s| *s = -1);
            th_ntm[th..].iter_mut().for_each(|s| *s = -1);
            buckets[0] = OUTPUT_BUCKETS_LAYOUT[(pos.occ.count_ones() as usize).min(32)] as i32;
            let eval = sigmoid(f32::from(pos.score) * score_multiplier / target_scale);
            let result = f32::from(pos.result) / 2.0;
            targets[0] = wdl * result + (1.0 - wdl) * eval;
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), targets)| {
            // Two SEPARATE FT tensors (quantised i16 / i8 respectively). Per
            // design/11 §12.2 the threats transformer is seeded from ZERO and
            // carries NO bias (the shared FT bias lives on the HalfKA tensor);
            // that also keeps the two bias gradients from colliding into an
            // un-eliminable CSE copy in the backward graph.
            let l0_pst = builder.new_affine("l0_pst", HALFKA_INPUTS, HL);
            l0_pst.init_with_effective_input_size(32);
            let l0_threat_w =
                builder.new_weights("l0_threatw", (HL, THREAT_INPUTS), InitSettings::Zeroed);
            let l1 = builder.new_affine("l1", HL, NUM_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2, NUM_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, NUM_BUCKETS);

            // node + node: the two-transformer sum into one accumulator/perspective.
            let acc_stm = l0_pst.forward(hk_stm) + l0_threat_w.matmul(th_stm);
            let acc_ntm = l0_pst.forward(hk_ntm) + l0_threat_w.matmul(th_ntm);

            // D40 pairwise trunk.
            let stm_hidden = acc_stm.crelu().pairwise_mul();
            let ntm_hidden = acc_ntm.crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);

            let hl2 = l1.forward(hl1).select(buckets).screlu();
            let hl3 = l2.forward(hl2).select(buckets).screlu();
            let out = l3.forward(hl3).select(buckets);

            let loss = out.sigmoid().squared_error(targets).reduce_sum_batch();
            (Some(loss), vec![("output".to_string(), out)])
        },
    );

    let device = DefaultDevice::new(0).expect("device");
    if let Ok(fen_path) = std::env::var("D40_PROBE_FENS")
        && (std::env::var("D40_PROBE_RAW").is_ok() || std::env::var("D40_PROBE_QUANT").is_ok())
    {
        let mut weights = ModelWeights::zeroed(&defn);
        let prefix = if let Ok(path) = std::env::var("D40_PROBE_QUANT") {
            load_quantised_weights(&mut weights, &path);
            "D40_PROBE_QUANT"
        } else {
            load_raw_weights(&mut weights, &std::env::var("D40_PROBE_RAW").unwrap());
            "D40_PROBE_RAW"
        };
        let mut evaluator = ModelEvaluator::new(&defn, device.clone()).expect("raw evaluator");
        evaluator.load_weights(&weights).expect("raw model weights");
        let fens = std::fs::read_to_string(fen_path).expect("read D40_PROBE_FENS");
        for line in fens.lines() {
            let fen = line.trim();
            if fen.is_empty() || fen.starts_with('#') {
                continue;
            }
            let position: ChessBoard = format!("{fen} | 0 | 0.5").parse().expect("probe FEN");
            let host = mapper.map(&[position], Step::default(), 1);
            let input = host.to_device(&device).expect("probe inputs to device");
            let output = evaluator.evaluate(&input).expect("probe evaluate");
            let TValue::F32(value) = output
                .get("output")
                .expect("probe output")
                .to_host()
                .expect("probe output to host")
            else {
                panic!("probe output must be f32");
            };
            println!("{prefix}\t{:.9}\t{fen}", value[0]);
        }
        return;
    }

    let seed = env_usize("D40_SEED", 0x00D4_0FEA) as u64;
    let mut weights = ModelWeights::new(&defn, seed);
    if let Ok(path) = std::env::var("D40_INIT_QUANT") {
        load_quantised_weights(&mut weights, &path);
        println!("initialised from quantized checkpoint {path}");
    } else if let Ok(path) = std::env::var("D40_INIT_RAW") {
        load_raw_weights(&mut weights, &path);
        println!("initialised from raw checkpoint {path}");
    }
    // AdamW default clip +/-1.98 => i8 L1/threat weights * 64 <= 127 (no overflow).
    let params = AdamWParams::default();
    let mut optimiser = AdamW::new(defn, weights, device, params).expect("optimiser");

    let loader = ReadMapLoader::new(reader, mapper, threads);

    let final_lr = initial_lr * 0.3f32.powi(3);
    let lr_schedule: Box<dyn Fn(Step) -> f32> = Box::new(move |step| {
        if lr_end_superbatch <= 1 {
            return initial_lr;
        }
        let t =
            ((step.superbatch() as f32 - 1.0) / (lr_end_superbatch as f32 - 1.0)).clamp(0.0, 1.0);
        final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + (std::f32::consts::PI * t).cos())
    });

    let log_every = env_usize("D40_LOG_EVERY", 128)
        .max(1)
        .min(batches_per_superbatch);
    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch,
            end_superbatch,
        },
        log_rate: log_every,
        lr_schedule,
    };

    let fmt = vec![
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

    println!(
        "--- D40 training: superbatches {start_superbatch}..={end_superbatch} x {batches_per_superbatch} batches x {batch_size} ---"
    );
    println!(
        "--- D40 targets: scale={target_scale} wdl={wdl} score_multiplier={score_multiplier} seed={seed} ---"
    );
    println!("--- D40 HalfKA-only ablation: {} ---", halfka_only());
    let save_out = out_dir.clone();
    let save_id = net_id.clone();
    train(
        &mut optimiser,
        schedule,
        loader,
        move |_, step, err| {
            if step.batch() % log_every == 0 {
                println!(
                    "  sb {} batch {}: loss {err:.6}",
                    step.superbatch(),
                    step.batch()
                );
            }
        },
        move |opt, step| {
            let sb = step.superbatch();
            if sb % save_rate == 0 || sb == end_superbatch {
                let weights = opt.cpu_weights().expect("sync CPU weights");
                let buf = weights
                    .to_quantised_buffer(&fmt, true)
                    .expect("quantise D40 checkpoint");
                let path = format!("{save_out}/{save_id}-sb{sb}-quant.bin");
                std::fs::write(&path, &buf).expect("write D40 checkpoint");
                println!("  saved {path} ({} bytes)", buf.len());
                if sb == end_superbatch {
                    let raw_path = format!("{save_out}/{save_id}-final-raw.bin");
                    let mut raw = std::fs::File::create(&raw_path).expect("create raw checkpoint");
                    weights
                        .write_into(&mut raw)
                        .expect("write named raw checkpoint");
                    println!("  saved {raw_path} (trainer-parity source)");
                }
            }
        },
    )
    .expect("D40 training");
    println!("=== d40_features: production-shaped checkpoint complete ===");
}
