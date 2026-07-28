/*
 * D40-ATTN: KQV self-attention eval head (Stage-0 offline probe).
 * Scratchpad-only scaffold (discardable, D15/D28). OFFLINE ONLY.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * `docs/research/transformer-head-exploration.md` §5 Stage 0 asks one cheap,
 * decisive question: on the SAME inputs and value target, does an attention
 * head fit the target BETTER than our D40 MLP head (`768->16->32->1`, per 8
 * buckets)?  This trainer is the attention arm of that A/B.  It is a byte-clone
 * of `d40_features.rs`'s FRONT END (HalfKAv2_hm + FullThreats sparse inputs,
 * the two separately-quantised FT tensors `l0_pst`/`l0_threatw` summed into one
 * 768-wide accumulator per perspective, the identical sigmoid+squared_error
 * value loss and target handling) and REPLACES ONLY the head.
 *
 * STAGE-0 SCOPE (read before wiring anything downstream)
 * ------------------------------------------------------
 *  - NO engine export, NO quantisation, NO `.cenn` pack.  `netpack` expects the
 *    l1/l2/l3 MLP tensors; an attention head cannot be packed by it.  Inference
 *    in ARCHON is a LATER stage (design doc §5 Stage 1: a hand-written Rust
 *    scalar forward, only if this offline gate is positive).  This trainer only
 *    (1) trains, (2) reports running training loss + an optional held-out
 *    validation loss (the "does it fit the target better" metric), and
 *    (3) saves the RAW bullet checkpoint (`ModelWeights::write_into`).
 *  - Single-head, single-layer.  `D40_ATTN_HEADS`/`D40_ATTN_LAYERS` are read but
 *    asserted == 1; multi-head/-layer is a follow-up (needs a per-head QKDIM
 *    split + per-head softmax + concat, straightforward but out of Stage-0).
 *
 * ARM IMPLEMENTED: A ("accumulator-chunk attention") — design doc §1.A.
 *   The 768-wide per-perspective accumulator (after `crelu`) is reshaped into
 *   T tokens x d (T = D40_ATTN_TOKENS, default 12 -> d = 64, assert 768 % T==0).
 *   Tokens carry NO chess semantics — attention just learns cross-feature
 *   second-order interactions (our `pairwise_mul` does a crude version of this).
 *   Shared Wq/Wk/Wv project each token; a per-token gate scores each token;
 *   softmax over tokens; value-weighted token-sum -> pooled vector; both
 *   perspectives pooled + concatenated -> a small bucketed output MLP -> scalar.
 *   The 8-bucket `.select(buckets)` structure is kept so the value-target/loss
 *   path is byte-identical to d40_features.
 *
 * THE Q.K^T TRANSPOSE PROBLEM (the crux — honest resolution)
 * ---------------------------------------------------------
 *   A textbook single-head self-attention forms scores = Q.K^T / sqrt(dk), a
 *   [T x T] matrix, which contracts Q [T,dk] against K^T [dk,T] — i.e. one of
 *   Q/K MUST be transposed so tokens land on OPPOSITE axes of the matmul.
 *
 *   bullet's MODEL builder (the `ModelDefinition::build` closure, i.e.
 *   `bullet_compiler::model::builder::ModelNode`) exposes NO transpose and NO
 *   plain softmax:
 *     - The only graph transpose lives one IR layer down, on the tensor-IR
 *       `MatrixLayout::transpose`
 *       (compiler/src/tensor/operation/linear/matmul.rs:16), and the model-level
 *       `Matmul` op HARDCODES `col_mjr: true` for both operands with no
 *       transpose field (compiler/src/model/operations/linear.rs:130,134,138).
 *       `ModelNode` has no `.transpose()` (compiler/src/model/builder.rs).
 *     - `reshape` lowers to a NO-OP buffer reinterpretation
 *       (compiler/src/model/operations/pointwise.rs:191 returns inputs[0]), so a
 *       col-major [T,d] reshaped to [d,T] is NOT the transpose — it reorders
 *       values.  Reshape cannot manufacture a transpose.
 *     - The only `softmax` is on the tensor-IR `TNode`
 *       (compiler/src/tensor/builder.rs:159), unreachable from the model builder;
 *       `ModelNode` has only the FUSED `softmax_crossentropy_loss`
 *       (compiler/src/model/builder.rs:360), not a standalone softmax activation.
 *
 *   CONCLUSION: a genuine [T x T] Q.K^T self-attention is NOT expressible with
 *   bullet's model-builder ops.  Per the design brief we implement the closest
 *   expressible KQV variant and say so plainly:
 *
 *   DIAGONAL SELF-GATING KQV ATTENTION (what this file actually computes):
 *     Q, K, V = Wq/Wk/Wv @ X   (all three projections present + used)
 *     per-token logit_t = (Q_t . K_t) / sqrt(dk)      <- the DIAGONAL of Q.K^T
 *     w = softmax_t(logits)                            <- over the T tokens
 *     pooled = sum_t w_t * V_t                         <- value-weighted pool
 *   This is exactly the design brief's proposed fallback ("per-token importance
 *   = softmax(MLP(token)) then weighted token-sum — no [T x T] transpose"): the
 *   bilinear Q_t.K_t IS a learned per-token importance MLP.  It keeps the full
 *   KQV parameterisation and every projection, and it needs no transpose.  We
 *   compose the softmax by hand from `exp` + `reduce_sum_cols` + `repeat`
 *   (reciprocal); there is no reduce-MAX at the model level, so the softmax
 *   omits max-subtraction — safe here because `crelu` bounds the accumulator to
 *   [0,1] and QKDIM is small, so logits cannot overflow `exp`.
 *
 *   The full off-diagonal cross-token [T x T] attention is left for the Stage-1
 *   Rust forward (which has a free hand-written transpose) OR a future bullet
 *   patch exposing a model-level transpose.
 *
 * ARM C IS A FOLLOW-UP (design doc §1.C — the strong candidate).
 *   Arm C (per-square FT embeddings: keep 64 per-square accumulators instead of
 *   SUMMING them, attention over 64 square-tokens) is NOT in this file: our FT
 *   sums ALL active features into ONE accumulator (see `acc_stm` below), so Arm
 *   C needs an FT restructure (64 separate per-square feature transformers, a
 *   different SparseInputType + graph).  That is a distinct trainer; this file
 *   is Arm A only.
 *
 * ENV KNOBS
 * ---------
 *   Shared with d40_features.rs (identical semantics):
 *     D40_DATA               (required) bulletformat/sfbinpack train path
 *     D40_FORMAT             "bulletformat" (default) | "sfbinpack"
 *     D40_THREADS            mapper threads (default 6)
 *     D40_SF_BUFFER_MB       sfbinpack ring buffer MiB (default 256)
 *     D40_RANDOM_SKIP_PCT    sfbinpack random position skip 0-99 (default 0)
 *     D40_HALFKA_ONLY        1 = drop threat features (ablation, default 0)
 *     D40_VERIFY_ROWS        feature-parity self-check rows (default 64)
 *     D40_VERIFY_ONLY        1 = run parity check then exit
 *     D40_OUT                checkpoint dir (default d40_attn_ckpt)
 *     D40_ID                 net id prefix (default d40-attn-0)
 *     D40_START_SUPERBATCH   resume start (default 1)
 *     D40_SUPERBATCHES       end superbatch (default 30)
 *     D40_BPS                batches / superbatch (default 1520)
 *     D40_BATCH              batch size (default 16384)
 *     D40_SAVE_RATE          save + held-out-eval every N superbatches (def 10)
 *     D40_LR                 initial LR (default 0.001, cosine to LR*0.027)
 *     D40_LR_END_SB          LR schedule end superbatch (default = SUPERBATCHES)
 *     D40_EVAL_TARGET_SCALE  target sigmoid scale (default 400)
 *     D40_WDL                WDL blend 0..1 (default 0.3)
 *     D40_SCORE_MULTIPLIER   score pre-scale (default 1.0)
 *     D40_SEED               weight init seed (default 0x00D40FEA)
 *     D40_LOG_EVERY          batch log interval (default 128)
 *     D40_LOSS_POWER         loss |pred-tgt|^p (default 2 = squared_error)
 *     D40_INIT_RAW           raw d40 checkpoint to warm-start the FT from
 *                            (loads ONLY l0_pstw/l0_pstb/l0_threatw; the
 *                            attention head stays randomly initialised)
 *   Attention head (new):
 *     D40_ATTN_TOKENS        T tokens; d = 768/T (default 12 -> d=64; 768%T==0)
 *     D40_ATTN_QKDIM         Wq/Wk output dim dk (default 32)
 *     D40_ATTN_VDIM          Wv output dim dv (default = d)
 *     D40_ATTN_OUT_HID       bucketed output MLP hidden width (default 16)
 *     D40_ATTN_HEADS         asserted == 1 (Stage-0)
 *     D40_ATTN_LAYERS        asserted == 1 (Stage-0)
 *     D40_ATTN_FREEZE_FT     1 = freeze l0_pstw/l0_pstb/l0_threatw (train ONLY
 *                            the attention head; requires D40_INIT_RAW).  This
 *                            is the cleanest "same accumulator, MLP head vs
 *                            attention head" isolation.
 *   Held-out validation (the Stage-0 metric):
 *     D40_ATTN_VAL_DATA      optional disjoint val slice; if set, mean
 *                            (sigmoid(out)-target)^2 is reported at each save
 *                            point (identical loss fn to training -> directly
 *                            comparable to the MLP-head run on the same slice)
 *     D40_ATTN_VAL_FORMAT    val format (default = D40_FORMAT)
 *     D40_ATTN_VAL_MAX       max val positions held in memory (default 2048)
 */
use bullet_compiler::tensor::TValue;
use bullet_lib::{
    game::inputs::SparseInputType,
    nn::InitSettings,
    value::loader::{self, DirectSequentialDataLoader},
};
use bullet_trainer::{
    model::{
        ModelDefinition, ModelEvaluator, ModelInputs, ModelInputsMapper, ModelWeights,
    },
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train_frozen},
};
use bulletformat::ChessBoard;
use chessengine::attacks::{attacks, pawn_attacks};
use chessengine::eval::nnue::sota_features::{
    HALFKA_DIMS, THREAT_DIMS, for_each_halfka, for_each_threat, halfka_index, threat_index,
};
use chessengine::{Bitboard, Board, Color, Piece, PieceType, Square};
use std::collections::BTreeSet;
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

/// Official-style random position skip (decorrelates game-contiguous binpack
/// streams). `D40_RANDOM_SKIP_PCT` (0-99, default 0). Deterministic FNV over
/// occupancy/ply/score — identical construction to d40_features.rs.
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

// --- Feature-space layout (design/11 §12.1) — identical to d40_features.rs ---
const HALFKA_INPUTS: usize = HALFKA_DIMS;
const THREAT_INPUTS: usize = THREAT_DIMS;
const TOTAL_INPUTS: usize = HALFKA_INPUTS + THREAT_INPUTS; // 83_248
const HALFKA_MAX_ACTIVE: usize = 32;
const THREAT_MAX_ACTIVE: usize = 160;
const MAX_ACTIVE: usize = HALFKA_MAX_ACTIVE + THREAT_MAX_ACTIVE; // 192
const DEFAULT_SCALE: f32 = 400.0;
const DEFAULT_WDL: f32 = 0.3;
const NUM_BUCKETS: usize = 8;
const HL: usize = 768;

/// The three FT tensors shared with d40_features.rs / the raw checkpoint.
/// (`new_affine("l0_pst", ..)` emits `l0_pstw`+`l0_pstb`; the threat transformer
/// is `l0_threatw`.)  These are the tensors we warm-start and optionally freeze.
const FT_TENSORS: [&str; 3] = ["l0_pstw", "l0_pstb", "l0_threatw"];

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
// Custom SparseInputType: HalfKAv2_hm + FullThreats over 0..83,248.
// Byte-identical to d40_features.rs — the two heads MUST share one front end.
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

    assert!(max_halfka <= HALFKA_MAX_ACTIVE, "HalfKA sparse capacity is too small");
    assert!(max_threats <= THREAT_MAX_ACTIVE, "threat sparse capacity is too small");
    println!(
        "feature parity: {} positions exact; max active HalfKA={max_halfka}, threats={max_threats}",
        checked
    );
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

/// Selectively load ONLY the three FT tensors from a raw d40 `ModelWeights`
/// checkpoint into `weights`, ignoring every other record (the raw d40 file also
/// holds l1/l2/l3 MLP tensors that DON'T exist in the attention graph — a plain
/// `ModelWeights::load_from` would panic on those unknown ids,
/// crates/trainer/src/model/weights.rs:120).  Same raw record format as
/// d41_alloc11.rs `load_trunk_into`: [id ascii + '\n'][u64 len LE][len x f32 LE].
fn load_ft_from_raw(weights: &mut ModelWeights, path: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read D40_INIT_RAW {path}: {e}"));
    let mut off = 0usize;
    let mut loaded = 0usize;
    while off < bytes.len() {
        let nl = off
            + bytes[off..].iter().position(|&c| c == b'\n').expect("raw record id newline");
        let id = std::str::from_utf8(&bytes[off..nl]).expect("raw id ascii").to_string();
        off = nl + 1;
        let len = usize::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        let vals: Vec<f32> = bytes[off..off + len * 4]
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        off += len * 4;
        if FT_TENSORS.contains(&id.as_str()) {
            assert!(
                weights.set(&id, TValue::F32(vals)),
                "FT tensor {id} shape/dtype mismatch vs the attention graph"
            );
            loaded += 1;
        }
    }
    assert_eq!(loaded, FT_TENSORS.len(), "did not find all FT tensors in {path}");
    println!("warm-start FT: loaded {loaded} tensors from {path}");
}

fn build_reader(data: &str, format: &str, threads: u8) -> D40Reader {
    match format {
        "bulletformat" => D40Reader::Bullet(DirectSequentialDataLoader::new(&[data])),
        "sfbinpack" => {
            let buffer_mb = env_usize("D40_SF_BUFFER_MB", 256);
            println!("SF binpack input: buffer={buffer_mb}MiB quiet-filter=standard");
            D40Reader::Sf(loader::SfBinpackLoader::new(
                data,
                buffer_mb,
                usize::from(threads),
                sfbinpack_filter,
            ))
        }
        other => panic!("D40_FORMAT must be bulletformat or sfbinpack, got {other}"),
    }
}

/// Recompute the exact training value target for a position (must match the
/// mapper below, so a held-out MSE is directly comparable to training loss).
fn value_target(pos: &ChessBoard, score_multiplier: f32, target_scale: f32, wdl: f32) -> f32 {
    let eval = sigmoid(f32::from(pos.score) * score_multiplier / target_scale);
    let result = f32::from(pos.result) / 2.0;
    wdl * result + (1.0 - wdl) * eval
}

fn main() {
    let data = std::env::var("D40_DATA")
        .expect("set D40_DATA to a bulletformat ChessBoard train.bin path");
    let feats = HalfKaThreats;
    let format = std::env::var("D40_FORMAT").unwrap_or_else(|_| "bulletformat".into());
    let threads = env_usize("D40_THREADS", 6) as u8;
    let reader = build_reader(&data, &format, threads);

    println!("=== D40-ATTN (Stage-0 offline attention head) ===");
    println!("input type   : {}", feats.description());
    assert_eq!(feats.num_inputs(), TOTAL_INPUTS);
    assert_eq!(feats.max_active(), MAX_ACTIVE);
    let verify_rows = env_usize("D40_VERIFY_ROWS", 64);
    verify_feature_parity(&reader, verify_rows);
    if env_usize("D40_VERIFY_ONLY", 0) != 0 {
        println!("D40_VERIFY_ONLY complete");
        return;
    }

    // ------------------------------------------------------------------
    // Attention head configuration.
    // ------------------------------------------------------------------
    let tokens = env_usize("D40_ATTN_TOKENS", 12);
    assert!(tokens > 0 && HL % tokens == 0, "D40_ATTN_TOKENS must divide {HL}");
    let d = HL / tokens; // per-token model dim (default 64)
    let qk_dim = env_usize("D40_ATTN_QKDIM", 32);
    let v_dim = env_usize("D40_ATTN_VDIM", d);
    let out_hid = env_usize("D40_ATTN_OUT_HID", 16);
    assert!(qk_dim > 0 && v_dim > 0 && out_hid > 0);
    let heads = env_usize("D40_ATTN_HEADS", 1);
    let layers = env_usize("D40_ATTN_LAYERS", 1);
    assert_eq!(heads, 1, "Stage-0 is single-head; multi-head is a follow-up");
    assert_eq!(layers, 1, "Stage-0 is single-layer; multi-layer is a follow-up");
    let attn_scale = 1.0f32 / (qk_dim as f32).sqrt();

    // Same 6 model inputs + mapper as d40_features.rs (unchanged target path).
    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (HALFKA_INPUTS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_INPUTS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("threat_stm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("buckets", (NUM_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let out_dir = std::env::var("D40_OUT").unwrap_or_else(|_| "d40_attn_ckpt".to_string());
    let net_id = std::env::var("D40_ID").unwrap_or_else(|_| "d40-attn-0".to_string());
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
    assert!(lr_end_superbatch >= end_superbatch, "D40_LR_END_SB must be >= D40_SUPERBATCHES");
    let target_scale = env_f32("D40_EVAL_TARGET_SCALE", DEFAULT_SCALE);
    let wdl = env_f32("D40_WDL", DEFAULT_WDL);
    let score_multiplier = env_f32("D40_SCORE_MULTIPLIER", 1.0);
    assert!(target_scale > 0.0 && target_scale.is_finite());
    assert!((0.0..=1.0).contains(&wdl));
    assert!(score_multiplier > 0.0 && score_multiplier.is_finite());
    std::fs::create_dir_all(&out_dir).expect("create D40_OUT directory");

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |pos: &ChessBoard,
              _step,
              (((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), targets)| {
            let (mut hk, mut th) = (0usize, 0usize);
            feats.map_features(pos, |stm, ntm| {
                if stm < HALFKA_INPUTS {
                    assert!(hk < HALFKA_MAX_ACTIVE, "HalfKA sparse capacity exceeded");
                    hk_stm[hk] = stm as i32;
                    hk_ntm[hk] = ntm as i32;
                    hk += 1;
                } else {
                    assert!(th < THREAT_MAX_ACTIVE, "FullThreats sparse capacity exceeded");
                    th_stm[th] = (stm - HALFKA_INPUTS) as i32;
                    th_ntm[th] = (ntm - HALFKA_INPUTS) as i32;
                    th += 1;
                }
            });
            hk_stm[hk..].iter_mut().for_each(|s| *s = -1);
            hk_ntm[hk..].iter_mut().for_each(|s| *s = -1);
            th_stm[th..].iter_mut().for_each(|s| *s = -1);
            th_ntm[th..].iter_mut().for_each(|s| *s = -1);
            buckets[0] = OUTPUT_BUCKETS_LAYOUT[(pos.occ.count_ones() as usize).min(32)] as i32;
            targets[0] = value_target(pos, score_multiplier, target_scale, wdl);
        },
    );

    // ------------------------------------------------------------------
    // Graph: identical FT front end + DIAGONAL SELF-GATING KQV head (Arm A).
    // ------------------------------------------------------------------
    let defn = ModelDefinition::build(
        &inputs,
        move |builder, (((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), targets)| {
            // --- FT front end: byte-identical to d40_features.rs ---
            let l0_pst = builder.new_affine("l0_pst", HALFKA_INPUTS, HL);
            l0_pst.init_with_effective_input_size(32);
            let l0_threat_w =
                builder.new_weights("l0_threatw", (HL, THREAT_INPUTS), InitSettings::Zeroed);
            let acc_stm = l0_pst.forward(hk_stm) + l0_threat_w.matmul(th_stm);
            let acc_ntm = l0_pst.forward(hk_ntm) + l0_threat_w.matmul(th_ntm);

            // --- Shared attention projections (bias-free: a per-token bias
            //     [dk,1] would not broadcast across the T token-columns).
            //     Wq/Wk: d->dk, Wv: d->dv.  new_weights shape is (rows, cols).
            //     InitSettings is Clone (not Copy), so build one per projection. ---
            let init_stdev = (1.0 / d as f32).sqrt();
            let proj_init = || InitSettings::Normal { mean: 0.0, stdev: init_stdev };
            let wq = builder.new_weights("attn_qw", (qk_dim, d), proj_init());
            let wk = builder.new_weights("attn_kw", (qk_dim, d), proj_init());
            let wv = builder.new_weights("attn_vw", (v_dim, d), proj_init());
            // Bucketed output MLP: (2*dv) -> out_hid -> 1, per 8 buckets.
            let o1 = builder.new_affine("attn_o1", 2 * v_dim, NUM_BUCKETS * out_hid);
            let o2 = builder.new_affine("attn_o2", out_hid, NUM_BUCKETS);

            // Attention over one perspective's 768-acc.  Reshape [768,1] ->
            // [d, T] (col-major): each of the T columns is a contiguous d-block
            // = one token.  Wq/Wk/Wv left-multiply, applying ONE shared weight
            // to every token (weight sharing across tokens AND perspectives).
            //   Q,K,V   = W @ X            [dk|dv, T]
            //   logit_t = (Q_t . K_t)/sqrt(dk)   -> reduce over the dk rows -> [1,T]
            //   w       = softmax_t(logits)      -> exp / sum over T cols
            //   pooled  = V @ w^T                [dv, 1]   (batched act@act matmul)
            // Inlined per perspective (rather than a closure) to keep the
            // model-node lifetimes trivial for the compiler.
            let acc_x_stm = acc_stm.crelu().reshape((d, tokens));
            let q_stm = wq.matmul(acc_x_stm);
            let k_stm = wk.matmul(acc_x_stm);
            let v_stm = wv.matmul(acc_x_stm);
            let logits_stm = (q_stm * k_stm).reduce_sum_rows() * attn_scale; // [1, T]
            let e_stm = logits_stm.exp();
            let w_stm = e_stm * (1.0 / e_stm.reduce_sum_cols()).repeat(tokens); // [1, T]
            let pooled_stm = v_stm.matmul(w_stm.reshape((tokens, 1))); // [dv, 1]

            let acc_x_ntm = acc_ntm.crelu().reshape((d, tokens));
            let q_ntm = wq.matmul(acc_x_ntm);
            let k_ntm = wk.matmul(acc_x_ntm);
            let v_ntm = wv.matmul(acc_x_ntm);
            let logits_ntm = (q_ntm * k_ntm).reduce_sum_rows() * attn_scale;
            let e_ntm = logits_ntm.exp();
            let w_ntm = e_ntm * (1.0 / e_ntm.reduce_sum_cols()).repeat(tokens);
            let pooled_ntm = v_ntm.matmul(w_ntm.reshape((tokens, 1)));

            // Concat both perspectives -> bucketed output MLP -> scalar.
            let pooled = pooled_stm.concat(pooled_ntm); // [2*dv, 1]
            let hidden = o1.forward(pooled).select(buckets).screlu();
            let out = o2.forward(hidden).select(buckets); // [1, 1]

            // Identical value loss to d40_features.rs (default squared_error).
            let sig = out.sigmoid();
            let loss = match std::env::var("D40_LOSS_POWER").as_deref() {
                Ok(p) if p != "2" && p != "2.0" => {
                    let power: f32 = p.parse().expect("D40_LOSS_POWER must be a float");
                    sig.power_error(targets, power).reduce_sum_batch()
                }
                _ => sig.squared_error(targets).reduce_sum_batch(),
            };
            (Some(loss), vec![("output".to_string(), out)])
        },
    );

    let device = DefaultDevice::new(0).expect("device");
    let seed = env_usize("D40_SEED", 0x00D4_0FEA) as u64;
    let mut weights = ModelWeights::new(&defn, seed);

    // Warm-start / freeze the FT.  FREEZE_FT requires a warm-started FT (freezing
    // a randomly-initialised FT would defeat the isolation).
    let freeze_ft = env_usize("D40_ATTN_FREEZE_FT", 0) != 0;
    let mut frozen: BTreeSet<String> = BTreeSet::new();
    if let Ok(path) = std::env::var("D40_INIT_RAW") {
        load_ft_from_raw(&mut weights, &path);
        if freeze_ft {
            frozen = FT_TENSORS.iter().map(|s| s.to_string()).collect();
        }
    } else {
        assert!(!freeze_ft, "D40_ATTN_FREEZE_FT=1 requires D40_INIT_RAW (a warm-started FT)");
    }

    let params = AdamWParams::default();
    let mut optimiser = AdamW::new(defn, weights, device.clone(), params).expect("optimiser");
    let loader = ReadMapLoader::new(reader, mapper.clone(), threads);

    let final_lr = initial_lr * 0.3f32.powi(3);
    let lr_schedule: Box<dyn Fn(Step) -> f32> = Box::new(move |step| {
        if lr_end_superbatch <= 1 {
            return initial_lr;
        }
        let t =
            ((step.superbatch() as f32 - 1.0) / (lr_end_superbatch as f32 - 1.0)).clamp(0.0, 1.0);
        final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + (std::f32::consts::PI * t).cos())
    });

    let log_every = env_usize("D40_LOG_EVERY", 128).max(1).min(batches_per_superbatch);
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

    // ------------------------------------------------------------------
    // Optional held-out validation slice (the Stage-0 "fit the target" metric).
    // Collect a disjoint sample into memory; scored one position at a time
    // because ModelEvaluator is compiled for batch size 1
    // (crates/trainer/src/model/evaluator.rs:23 -> lower_forward(1)).
    // ------------------------------------------------------------------
    let val_positions: Vec<ChessBoard> = match std::env::var("D40_ATTN_VAL_DATA") {
        Ok(val_data) => {
            let val_format =
                std::env::var("D40_ATTN_VAL_FORMAT").unwrap_or_else(|_| format.clone());
            let val_max = env_usize("D40_ATTN_VAL_MAX", 2048);
            let val_reader = build_reader(&val_data, &val_format, threads);
            let mut positions: Vec<ChessBoard> = Vec::with_capacity(val_max);
            val_reader.read_chunks(0, |chunk: &[ChessBoard]| {
                let take = chunk.len().min(val_max.saturating_sub(positions.len()));
                positions.extend_from_slice(&chunk[..take]);
                positions.len() >= val_max
            });
            println!("held-out val: {} positions from {val_data} ({val_format})", positions.len());
            positions
        }
        Err(_) => Vec::new(),
    };

    // Evaluator borrows the definition only for its own construction; it is then
    // owned and moved into the superbatch callback (same pattern as
    // d41_alloc11.rs), so `train_frozen(&mut optimiser, ..)` can borrow the
    // optimiser mutably without conflict.
    let mut evaluator = if val_positions.is_empty() {
        None
    } else {
        Some(ModelEvaluator::new(optimiser.definition(), device.clone()).expect("evaluator"))
    };
    let eval_mapper = mapper.clone();
    let eval_device = device.clone();

    println!(
        "--- D40-ATTN: superbatches {start_superbatch}..={end_superbatch} x {batches_per_superbatch} x {batch_size} ---"
    );
    println!(
        "--- head: ARM A diagonal self-gating KQV | tokens={tokens} d={d} qkdim={qk_dim} vdim={v_dim} out_hid={out_hid} ---"
    );
    println!(
        "--- targets: scale={target_scale} wdl={wdl} score_mult={score_multiplier} seed={seed} | frozen_ft={} ---",
        !frozen.is_empty()
    );

    let save_out = out_dir.clone();
    let save_id = net_id.clone();
    train_frozen(
        &mut optimiser,
        schedule,
        loader,
        &frozen,
        move |_, step, err| {
            if step.batch() % log_every == 0 {
                println!("  sb {} batch {}: loss {err:.6}", step.superbatch(), step.batch());
            }
        },
        move |opt, step| {
            let sb = step.superbatch();
            if sb % save_rate != 0 && sb != end_superbatch {
                return;
            }
            // Save the RAW bullet checkpoint (Stage-0: raw only, no quant/pack).
            let raw_path = format!("{save_out}/{save_id}-sb{sb}-raw.bin");
            {
                let mut raw = std::fs::File::create(&raw_path).expect("create raw checkpoint");
                opt.cpu_weights()
                    .expect("sync CPU weights")
                    .write_into(&mut raw)
                    .expect("write raw checkpoint");
            }
            println!("  saved {raw_path}");
            if sb == end_superbatch {
                let final_path = format!("{save_out}/{save_id}-final-raw.bin");
                let mut raw = std::fs::File::create(&final_path).expect("create final raw");
                opt.cpu_weights()
                    .expect("sync CPU weights")
                    .write_into(&mut raw)
                    .expect("write final raw checkpoint");
                println!("  saved {final_path} (final)");
            }

            // Held-out validation loss (mean (sigmoid(out)-target)^2), if enabled.
            if let Some(ev) = evaluator.as_mut() {
                ev.load_weights(&opt.cpu_weights().expect("weights for held-out eval"))
                    .expect("load held-out eval weights");
                let mut sse = 0.0f64;
                for pos in &val_positions {
                    let host = eval_mapper.map(std::slice::from_ref(pos), Step::default(), 1);
                    let input = host.to_device(&eval_device).expect("val inputs to device");
                    let out = ev.evaluate(&input).expect("val forward");
                    let TValue::F32(v) =
                        out.get("output").expect("val output").to_host().expect("val output to host")
                    else {
                        panic!("val output must be f32");
                    };
                    let pred = sigmoid(v[0]);
                    let tgt = value_target(pos, score_multiplier, target_scale, wdl);
                    sse += f64::from((pred - tgt).powi(2));
                }
                let mse = sse / val_positions.len() as f64;
                println!("  [sb {sb}] held-out val loss {mse:.6} over {} positions", val_positions.len());
            }
        },
    )
    .expect("D40-ATTN training");
    println!("=== d40_attn: Stage-0 raw checkpoint(s) written to {out_dir} ===");
}
