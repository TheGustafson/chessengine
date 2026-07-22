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
    run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train, train_frozen},
};
use std::collections::BTreeSet;
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
    /// Fold a small "aux" bulletformat .bin (self-play / SF18 hard-negs) into the
    /// streamed main binpack at `mix_pct`% of positions (the flywheel fold-in, D40).
    /// The aux is read as random CONTIGUOUS windows straight off disk each injection:
    /// the OS page-caches it when it fits RAM (→ RAM speed) and pages it when it does
    /// not — so there is NO RAM ceiling and the fold-in set can grow with the flywheel.
    /// Requires a PRE-SHUFFLED aux .bin (contiguous windows must be decorrelated).
    Mixed { main: Box<SfReader>, aux_path: String, aux_len: usize, mix_pct: f64 },
}

impl DataReader<ChessBoard> for D40Reader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip_count: usize, mut f: F) {
        match self {
            Self::Bullet(reader) => DataReader::<ChessBoard>::read_chunks(reader, skip_count, f),
            Self::Sf(reader) => DataReader::<ChessBoard>::read_chunks(reader, skip_count, f),
            Self::Mixed { main, aux_path, aux_len, mix_pct } => {
                use std::io::{Read, Seek, SeekFrom};
                let (aux_len, pct) = (*aux_len, *mix_pct);
                let mut file = std::fs::File::open(aux_path).expect("open D40_MIX_DATA");
                // xorshift64 seeded off the aux size (deterministic, decorrelated windows).
                let mut seed: u64 = 0x9E37_79B9_7F4A_7C15 ^ (aux_len as u64).wrapping_mul(2_654_435_761);
                // aligned scratch (Vec<ChessBoard> => 8-byte aligned; read raw bytes into it).
                let mut aux: Vec<ChessBoard> = Vec::new();
                DataReader::<ChessBoard>::read_chunks(main.as_ref(), skip_count, |chunk| {
                    if f(chunk) {
                        return true;
                    }
                    if pct <= 0.0 || aux_len == 0 {
                        return false;
                    }
                    // inject an aux window sized so aux is exactly pct% of positions.
                    let n = (((chunk.len() as f64) * pct / (100.0 - pct)) as usize).min(aux_len);
                    if n == 0 {
                        return false;
                    }
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let start = (seed as usize) % (aux_len - n + 1);
                    aux.resize(n, ChessBoard::default());
                    let bytes = unsafe {
                        std::slice::from_raw_parts_mut(
                            aux.as_mut_ptr() as *mut u8,
                            n * std::mem::size_of::<ChessBoard>(),
                        )
                    };
                    file.seek(SeekFrom::Start((start * std::mem::size_of::<ChessBoard>()) as u64))
                        .and_then(|_| file.read_exact(bytes))
                        .expect("read aux window");
                    f(&aux)
                });
            }
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

/// Held-out band (2026-07-22): `D40_HELDOUT_BAND="lo:hi"` selects rows with
/// position_hash % 100 in [lo, hi) INSTEAD of the training keep-test
/// (hash % 100 >= SKIP). Disjoint from every training run whenever
/// hi <= D40_RANDOM_SKIP_PCT — the shadow population no net has trained on.
static HELDOUT_BAND: std::sync::LazyLock<Option<(u64, u64)>> =
    std::sync::LazyLock::new(|| {
        let raw = std::env::var("D40_HELDOUT_BAND").ok()?;
        let (lo, hi) = raw.split_once(':').expect("D40_HELDOUT_BAND must be lo:hi");
        let (lo, hi): (u64, u64) =
            (lo.parse().expect("band lo"), hi.parse().expect("band hi"));
        assert!(lo < hi && hi <= 100, "D40_HELDOUT_BAND must satisfy 0 <= lo < hi <= 100");
        println!("held-out band: hash%100 in [{lo},{hi})");
        Some((lo, hi))
    });

/// D47: policy enabled? (read once; the mapper must not pay movegen when off).
static POL_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("D40_POL_LAMBDA").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0) > 0.0
});
/// D47: legal-move mask on? Default ON (the spec is legal-rows-only); turning
/// it off is a deliberate ablation, not an accident.
static POL_MASK_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("D40_POL_LEGAL_MASK").map(|v| v != "0").unwrap_or(true)
});

/// D47: rebuild a chessengine `Board` from the STORED (stm-canonicalized)
/// bulletformat position. The stored frame has the mover as White on a
/// vertically-normalized board, so the reconstructed board is always
/// White-to-move; castling/ep are NOT stored (bulletformat drops them), so
/// the legal-move mask is conservatively missing castling/ep — harmless: the
/// sfbinpack filter guarantees the TARGET move is Normal non-capture, and a
/// mask that omits castling/ep only removes rows from the CE normalization.
fn stored_board(pos: &ChessBoard) -> Board {
    const PT: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];
    let mut board = Board::empty();
    let mut occ = pos.occ;
    let mut idx = 0usize;
    while occ != 0 {
        let sq = occ.trailing_zeros() as u8;
        occ &= occ - 1;
        let nibble = (pos.pcs[idx / 2] >> (4 * (idx & 1))) & 0xF;
        idx += 1;
        let color = if nibble & 8 == 0 { Color::White } else { Color::Black };
        let piece_type = PT[(nibble & 7) as usize];
        board.put_piece(Piece::new(color, piece_type), Square::new(sq));
    }
    board
}

fn sfbinpack_filter(entry: &loader::sfbinpack::TrainingDataEntry) -> bool {
    use loader::sfbinpack::{MoveType, PieceType};

    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        && match *HELDOUT_BAND {
            // Held-out extraction: band REPLACES the skip test (quiet-filter
            // criteria above stay identical — trainer-parity distribution).
            Some((lo, hi)) => {
                let h = position_hash(entry) % 100;
                h >= lo && h < hi
            }
            None => *RANDOM_SKIP_PCT == 0 || position_hash(entry) % 100 >= *RANDOM_SKIP_PCT,
        }
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
// D47 SHARED-BOTTLENECK MULTIHEAD ARCH (user-directed, 2026-07-22):
// accumulators 1536/perspective (threats retained), pairwise -> concat(1536)
// -> L1 -> 256 shared bottleneck (THE branch point for heads) -> value tail
// 256->32->1x8buckets. Policy = 1858 embedding rows off the 256 (D47 vocab,
// legal-rows-only). Engine consts in sota_head.rs must match (1536/256/32).
const L2: usize = 128;
const L3: usize = 32;
const HL: usize = 1280;

fn halfka_only() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| env_usize("D40_HALFKA_ONLY", 0) != 0)
}

/// King-bucket FACTORIZATION (task #27, `D40_FACTORIZE=1`). A pure
/// data-efficiency device: during training the 32 king buckets share a gradient
/// through a bucket-agnostic 704-wide factoriser, folded back into the real FT
/// at export so the engine format + inference cost are UNCHANGED. Default is 0
/// (the existing 240sb/720sb recipes reproduce byte-identically when unset).
fn factorize() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| env_usize("D40_FACTORIZE", 0) != 0)
}

/// The bucket-agnostic factoriser width: SF's `PS_NB` piece-square stride, of
/// which `HALFKA_INPUTS` is exactly 32 tiled copies (one per king bucket). Each
/// real HalfKA feature `f = bucket*704 + piece_square` shares the factor
/// `HALFKA_INPUTS + (f % 704)` — the king-bucket-free piece-square weight.
const FACTOR_INPUTS: usize = 704;
const _: () = assert!(
    HALFKA_INPUTS == 32 * FACTOR_INPUTS,
    "HalfKA is 32 king buckets × 704 piece-square slots"
);

/// Fold the king-bucket factoriser into the real HalfKA transformer at export
/// (`D40_FACTORIZE=1`). This is the manual reimplementation of bullet's
/// `Factorised::merge_factoriser` for the two-FT custom path (design/11 §12.6).
///
/// `weights` is the trained `l0_pstw` in bullet's internal, **feature-major**
/// layout `w[feature*HL + neuron]` over the widened input space
/// `[HALFKA_INPUTS + FACTOR_INPUTS]`: real features `[0, HALFKA_INPUTS)` first,
/// then the 704 bucket-agnostic factor columns. It returns the engine-format
/// `l0_pstw` of shape `[HALFKA_INPUTS × HL]` (22528 features) where
/// `merged[f] = real[f] + factor[f % 704]` — the exact sum the factorised
/// accumulator computed during training, so the emitted quant checkpoint and
/// `.cenn` are shape-identical to a non-factorized net (netpack.py unchanged).
///
/// Runs as a `SavedFormat` transform BEFORE `round()/quantise::<i16>(255)`, so
/// the merge happens in float and quantisation sees the final engine weights.
fn merge_factoriser(_store: &ModelWeights, weights: Vec<f32>) -> Vec<f32> {
    assert_eq!(
        weights.len(),
        (HALFKA_INPUTS + FACTOR_INPUTS) * HL,
        "factorised l0_pstw has the wrong size"
    );
    let mut merged = vec![0.0f32; HALFKA_INPUTS * HL];
    for feature in 0..HALFKA_INPUTS {
        let real = feature * HL;
        let factor = (HALFKA_INPUTS + feature % FACTOR_INPUTS) * HL;
        for n in 0..HL {
            merged[real + n] = weights[real + n] + weights[factor + n];
        }
    }
    merged
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

fn take_i16s(raw: &[u8], offset: &mut usize, count: usize, scale: f32) -> Vec<f32> {
    let end = *offset + count * 2;
    let values = raw[*offset..end]
        .chunks_exact(2)
        .map(|bytes| f32::from(i16::from_le_bytes(bytes.try_into().unwrap())) / scale)
        .collect();
    *offset = end;
    values
}

fn take_i8s(raw: &[u8], offset: &mut usize, count: usize, scale: f32) -> Vec<f32> {
    let end = *offset + count;
    let values = raw[*offset..end]
        .iter()
        .map(|&byte| f32::from(byte as i8) / scale)
        .collect();
    *offset = end;
    values
}

fn take_f32s(raw: &[u8], offset: &mut usize, count: usize) -> Vec<f32> {
    let end = *offset + count * 4;
    let values = raw[*offset..end]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    *offset = end;
    values
}

fn load_quantised_weights(weights: &mut ModelWeights, path: &str) {
    let raw = std::fs::read(path).expect("read D40_PROBE_QUANT");
    let mut offset = 0usize;
    let pst = take_i16s(&raw, &mut offset, HALFKA_INPUTS * HL, 255.0);
    let pst_bias = take_i16s(&raw, &mut offset, HL, 255.0);
    let threat = take_i8s(&raw, &mut offset, THREAT_INPUTS * HL, 64.0);
    // D47 quant layout: when the policy head is on, polmovew (i8, transposed
    // [1858][L2]) + polmoveb (f32) ride BETWEEN the threat FT and l1 (the fmt
    // push order at save time). Gate on the same *POL_ON as defn/mapper so the
    // graph and the byte layout always agree; the padding tripwire below
    // catches any mismatch.
    let polmove = if *POL_ON {
        let w = take_i8s(&raw, &mut offset, 1858 * L2, 64.0);
        let b = take_f32s(&raw, &mut offset, 1858);
        Some((w, b))
    } else {
        None
    };
    let l1_saved = take_i8s(&raw, &mut offset, NUM_BUCKETS * L2 * HL, 64.0);
    let l1_bias = take_f32s(&raw, &mut offset, NUM_BUCKETS * L2);
    let l2_saved = take_f32s(&raw, &mut offset, NUM_BUCKETS * L3 * L2);
    let l2_bias = take_f32s(&raw, &mut offset, NUM_BUCKETS * L3);
    let l3_saved = take_f32s(&raw, &mut offset, NUM_BUCKETS * L3);
    let l3_bias = take_f32s(&raw, &mut offset, NUM_BUCKETS);
    let padding = &raw[offset..];
    assert!(padding.len() <= 63, "oversized quantized padding");
    assert!(
        padding
            .iter()
            .enumerate()
            .all(|(i, &byte)| byte == b"bullet"[i % 6]),
        "invalid quantized padding"
    );

    let mut tensors = vec![
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
    if let Some((w, b)) = polmove {
        tensors.push(("polmovew", transpose_saved(1858, L2, w)));
        tensors.push(("polmoveb", b));
    }
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
        "bulletformat" => {
            // The bulletformat path applies NO load-time filter (sfbinpack_filter
            // guards only the binpack path), so the corpus MUST be pre-filtered
            // to trainer parity (built by a *_tight builder / export-value's
            // default filter — docs/research/data-recipe-contract.md). An
            // unfiltered corpus here reproduces the canon −582 class of failure.
            println!(
                "bulletformat input: NO load-time filter — corpus must be pre-filtered \
                 (trainer-parity 5-criterion; see data-recipe-contract.md): {data}"
            );
            D40Reader::Bullet(DirectSequentialDataLoader::new(&[data.as_str()]))
        }
        "sfbinpack" => {
            let buffer_mb = env_usize("D40_SF_BUFFER_MB", 256);
            println!("SF binpack input: buffer={buffer_mb}MiB quiet-filter=standard");
            // coerce the fn-item to a fn pointer so `main` has the `SfReader` type.
            let filter: fn(&loader::sfbinpack::TrainingDataEntry) -> bool = sfbinpack_filter;
            let main: SfReader =
                loader::SfBinpackLoader::new(data.as_str(), buffer_mb, usize::from(threads), filter);
            // D40_MIX_DATA=<pre-shuffled aux .bin> folds self-play/SF18 hard-negs into
            // the stream at D40_MIX_PCT% (default 5). Empty/unset => plain sfbinpack.
            match std::env::var("D40_MIX_DATA") {
                Ok(aux_path) if !aux_path.is_empty() => {
                    let sz = std::fs::metadata(&aux_path).expect("stat D40_MIX_DATA").len() as usize;
                    let aux_len = sz / std::mem::size_of::<ChessBoard>();
                    let mix_pct: f64 =
                        std::env::var("D40_MIX_PCT").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
                    assert!((0.0..100.0).contains(&mix_pct), "D40_MIX_PCT must be 0-99 (float ok)");
                    assert!(aux_len > 0, "D40_MIX_DATA is empty: {aux_path}");
                    // Surface the aux-EPOCH count (the real overfit knob): with a small aux
                    // a fixed % over-presents it (5% of 1.7M over 720sb ~= 521 epochs). Prefer
                    // targeting ~2-6 aux epochs; grow the aux via the flywheel for more effect.
                    let total = (env_usize("D40_SUPERBATCHES", 240)
                        * env_usize("D40_BPS", 1520)
                        * env_usize("D40_BATCH", 16384)) as f64;
                    let aux_epochs = (total * mix_pct / 100.0) / aux_len as f64;
                    println!(
                        "FOLD-IN: {aux_len} aux @ {mix_pct}% => ~{aux_epochs:.1} aux epochs (on-demand windows, no RAM cap)"
                    );
                    D40Reader::Mixed { main: Box::new(main), aux_path, aux_len, mix_pct }
                }
                _ => D40Reader::Sf(main),
            }
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

    // Held-out dump (2026-07-22): stream the band-filtered reader to a raw
    // bulletformat .bin — the fixed shadow-population probe set. Trainer-parity
    // by construction: same loader, same filter body, same stm-canonicalized
    // conversion (records carry score/result/packed move, so the eval mode
    // below reproduces the exact training targets from them).
    if let Ok(dump_path) = std::env::var("D40_HELDOUT_DUMP") {
        assert!(
            HELDOUT_BAND.is_some(),
            "D40_HELDOUT_DUMP requires D40_HELDOUT_BAND (e.g. 0:5)"
        );
        let rows_cap = env_usize("D40_HELDOUT_ROWS", 1_000_000);
        use std::io::Write;
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&dump_path).expect("create D40_HELDOUT_DUMP"),
        );
        let mut written = 0usize;
        reader.read_chunks(0, |chunk: &[ChessBoard]| {
            let take = chunk.len().min(rows_cap - written);
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr() as *const u8,
                    take * std::mem::size_of::<ChessBoard>(),
                )
            };
            out.write_all(bytes).expect("write held-out rows");
            written += take;
            written == rows_cap
        });
        out.flush().expect("flush held-out dump");
        println!("held-out dump: {written} rows -> {dump_path}");
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

    // King-bucket factorization (D40_FACTORIZE): widen the HalfKA FT input by the
    // 704-wide factoriser and double its active count (each real feature also
    // emits its bucket-agnostic factor). Folded away at export (merge_factoriser),
    // so the engine format + inference cost are unchanged. Threats are king-
    // agnostic and are NOT factorized.
    let factorize = factorize();
    let halfka_ft_inputs = HALFKA_INPUTS + if factorize { FACTOR_INPUTS } else { 0 };
    let halfka_nnz = HALFKA_MAX_ACTIVE * if factorize { 2 } else { 1 };

    // Four sparse inputs: HalfKA (i16 tensor) + threats (i8 tensor), each stm/ntm.
    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (halfka_ft_inputs, 1), halfka_nnz)
        .add_sparse("halfka_ntm", (halfka_ft_inputs, 1), halfka_nnz)
        .add_sparse("threat_stm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("buckets", (NUM_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1))
        // D41 rich-net: 3-way game-result one-hot [L,D,W] (stm-relative), for the
        // co-trained WDL output head. Always present; inert unless D40_WDL_LAMBDA>0
        // (λ=0 => wdl head gets no gradient, value tensors identical => CENN parity).
        .add_dense("wdl_target", (3, 1))
        // D41 rich-net: teacher best-move from/to one-hots (stm-relative), for the
        // co-trained POLICY aux head (the H-AUX bet: shape the trunk, drop at deploy).
        // Sourced from the packed move in ChessBoard.extra. Inert unless D40_POL_LAMBDA>0.
        // D47: complete-move policy target over the 1,858-class vocabulary
        // (chessengine::policy_vocab — the shared single implementation) plus
        // the additive legal-move mask (0.0 legal / -30.0 illegal — soft enough that exp() never underflows to an exact 0; -1e4 NaN'd the CE via 0*log(0)), so the CE
        // normalizes over legal rows only. Both inert unless D40_POL_LAMBDA>0.
        .add_dense("pol_target", (1858, 1))
        .add_dense("pol_mask", (1858, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |pos: &ChessBoard,
              _step,
              ((((((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), targets), wdl_target), pol_target), pol_mask)| {
            let (mut hk, mut th) = (0usize, 0usize);
            feats.map_features(pos, |stm, ntm| {
                if stm < HALFKA_INPUTS {
                    // Factorization emits the real feature AND its bucket-agnostic
                    // factor into the SAME halfka stream, so the accumulator sums
                    // real_weight[f] + factor_weight[f % 704] — exact factorization.
                    let needed = if factorize { 2 } else { 1 };
                    assert!(
                        hk + needed <= halfka_nnz,
                        "HalfKA sparse capacity exceeded: position has more than \
                         {halfka_nnz} active halfka slots"
                    );
                    hk_stm[hk] = stm as i32;
                    hk_ntm[hk] = ntm as i32;
                    hk += 1;
                    if factorize {
                        hk_stm[hk] = (HALFKA_INPUTS + stm % FACTOR_INPUTS) as i32;
                        hk_ntm[hk] = (HALFKA_INPUTS + ntm % FACTOR_INPUTS) as i32;
                        hk += 1;
                    }
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
            // WDL head target: one-hot over [L, D, W], stm-relative (pos.result ∈ {0,1,2}).
            let r = (pos.result as usize).min(2);
            wdl_target[0] = 0.0;
            wdl_target[1] = 0.0;
            wdl_target[2] = 0.0;
            wdl_target[r] = 1.0;
            // D47 policy target: complete-move class over the 1858 vocabulary,
            // from the loader-packed ORIENTED from/to (extra[0..2]) + promo code
            // (extra[2]). Rows with no packed move (bf-converted bins, aux rows:
            // extra all zero => pf==pt) contribute an all-zero target and an
            // all-zero mask => zero CE. The legal mask (0.0 legal / -1e4
            // illegal) is generated from the stored position via the ENGINE's
            // movegen and mapped through the SAME canonical frame; the
            // every-target-is-legal assert below is the frame tripwire.
            pol_target.iter_mut().for_each(|x| *x = 0.0);
            pol_mask.iter_mut().for_each(|x| *x = 0.0);
            if *POL_ON {
                use chessengine::eval::nnue::policy_vocab::{self, POLICY_VOCAB};
                let (pf, pt) = (pos.extra[0], pos.extra[1]);
                let promo = match pos.extra[2] {
                    1 => Some(PieceType::Knight),
                    2 => Some(PieceType::Bishop),
                    3 => Some(PieceType::Rook),
                    _ => None,
                };
                let target_id = if pf < 64 && pt < 64 && pf != pt {
                    policy_vocab::policy_move_id(pf, pt, promo)
                } else {
                    None
                };
                if let Some(id) = target_id {
                    pol_target[id as usize] = 1.0;
                }
                if *POL_MASK_ON {
                    // stored frame == mover frame; only the _hm file-mirror
                    // remains, decided by the STORED king file (invariant to
                    // the vertical flip, so it matches the pack-time orient).
                    let board = stored_board(pos);
                    let ksq = Square::new(pos.ksq & 63);
                    let orient = policy_vocab::canonical_orient(false, ksq);
                    pol_mask.iter_mut().for_each(|x| *x = -30.0);
                    let mut list = chessengine::moves::MoveList::new();
                    chessengine::movegen::generate_legal(&board, &mut list);
                    for &m in list.iter() {
                        let (fc, tc) =
                            policy_vocab::canonical_from_to(m.from(), m.to(), orient);
                        if let Some(mid) =
                            policy_vocab::policy_move_id(fc, tc, m.promotion_piece())
                        {
                            pol_mask[mid as usize] = 0.0;
                        }
                    }
                    // Frame tripwire: the packed target must be a legal row.
                    if let Some(id) = target_id {
                        assert!(
                            pol_mask[id as usize] == 0.0,
                            "D47 frame bug: target id {id} not in the legal mask \
                             (pf {pf} pt {pt} promo {:?} ksq {})",
                            promo,
                            pos.ksq
                        );
                    }
                    let _ = POLICY_VOCAB;
                }
            }
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        |builder, ((((((((hk_stm, hk_ntm), th_stm), th_ntm), buckets), targets), wdl_target), pol_target), pol_mask)| {
            // Two SEPARATE FT tensors (quantised i16 / i8 respectively). Per
            // design/11 §12.2 the threats transformer is seeded from ZERO and
            // carries NO bias (the shared FT bias lives on the HalfKA tensor);
            // that also keeps the two bias gradients from colliding into an
            // un-eliminable CSE copy in the backward graph.
            // Factorization widens this FT by FACTOR_INPUTS (the extra 704 columns
            // are the bucket-agnostic factoriser). The effective input size tracks
            // the ACTIVE count (32 real, or 64 real+factor) so the initial
            // accumulator variance — and the merged weight distribution at t=0 —
            // match the non-factorized net (each column ~N(0, 2/nnz); the merged
            // real+factor sum has the same variance as a single non-factorized
            // column). Default path is unchanged (nnz == 32).
            let l0_pst = builder.new_affine("l0_pst", halfka_ft_inputs, HL);
            l0_pst.init_with_effective_input_size(halfka_nnz);
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
            let sig = out.sigmoid();

            // D41 rich-net: co-trained WDL output head. L3->3 logits (bucket-specific
            // via hl3), softmax-CE on the game-result one-hot. ADDITIVE loss
            // (value_MSE + λ·CE), NOT the multiplicative NLL that distorted the mean
            // in EXP-UNC-1. λ=0 (default) => the head is neither output-reachable nor
            // in the loss => dropped from the raw ckpt => value CENN byte-identical.
            // The WDL / policy heads are created ONLY inside their λ>0 loss branches
            // below. An affine created but NOT reachable in the loss is an orphaned
            // weight with no registered gradient → the backward pass panics
            // (definition.rs unwrap). λ=0 (default) ⇒ no head tensors exist ⇒ the
            // value net + graph are byte-identical to today.
            let wdl_lambda = env_f32("D40_WDL_LAMBDA", 0.0);
            let pol_lambda = env_f32("D40_POL_LAMBDA", 0.0);

            // EXP-UNC-1 (search study bet 1): heteroscedastic-NLL uncertainty head.
            // A PARALLEL log-variance head `l3_var` (same L3->NUM_BUCKETS shape as
            // the mean head) makes the value output (mean = `out`, log_var). Loss =
            // 0.5*(exp(-s)*(sig - t)^2 + s), the Gaussian NLL in sigmoid space with
            // s = log(sigma^2). The MEAN head + its save tensors (l3w/l3b) are
            // UNCHANGED, so the mean-only netpack for the Gate-1 value-parity screen
            // is byte-structurally identical to the control. logvar is exposed as a
            // SAVED OUTPUT so l3_var lands in the raw checkpoint — the raw save only
            // persists output-reachable weights, so a loss-only l3_var is dropped
            // (verified: without this the smoke raw ckpt had no l3_varw/l3_varb).
            // The runtime uncertainty sidecar (tools/nnue/extract_unc_head.py) reads
            // l3_varw/l3_varb from the raw ckpt. Default path (flag unset) stays
            // squared_error -> existing recipes reproduce byte-identically.
            let mut outputs = vec![("output".to_string(), out)];
            let loss = if std::env::var("D40_FROZEN_VAR").is_ok() {
                // EXP-UNC-1 FROZEN-mean post-hoc variance — the bulletproof fix
                // after coupled (g2 −179) AND detach (−116) both leaked the
                // variance gradient into the mean (with_no_grad did NOT isolate it).
                // Here the mean/trunk are FROZEN via `train_frozen` (warm-started
                // from the incumbent raw via D40_INIT_RAW), so ONLY l3_var trains —
                // no stop-grad needed, freezing isolates it perfectly. l3_var learns
                // to predict the FROZEN mean's squared residual (a clean post-hoc σ̂
                // predictor); the mean stays == the incumbent BY CONSTRUCTION, so
                // there is no Gate-1 mean parity to pass. Requires D40_INIT_RAW set
                // AND the train-call site to use train_frozen with the mean tensors.
                let l3_var = builder.new_affine("l3_var", L3, NUM_BUCKETS);
                let logvar = l3_var.forward(hl3).select(buckets);
                outputs.push(("logvar".to_string(), logvar));
                let sq = sig.squared_error(targets);
                let inv_var = (-logvar).exp();
                let nll = (inv_var * sq + logvar) * 0.5;
                nll.reduce_sum_batch()
            } else if std::env::var("D40_HETERO_NLL_DETACH").is_ok() {
                // EXP-UNC-1 DETACH variant (D33 reviewer B): protect the MEAN from
                // the heteroscedastic-NLL reweighting (Seitzer). The mean+trunk
                // train on PURE MSE, so the mean == the MSE control by construction
                // (Gate 1 automatic + TRUE, not "carried over"); the variance head
                // trains on the DETACHED residual off a DETACHED trunk — its own
                // weights still train (created outside with_no_grad, so they keep
                // requires_grad), but no gradient reaches the mean or the shared
                // trunk. σ̂ is then a pure post-hoc residual predictor. `x*1.0`
                // under with_no_grad yields a stop-grad copy (model.rs: an op node
                // under stop_grad gets no requires_grad, so backward halts there).
                let l3_var = builder.new_affine("l3_var", L3, NUM_BUCKETS);
                let hl3_sg = builder.with_no_grad(|| hl3 * 1.0);
                let logvar = l3_var.forward(hl3_sg).select(buckets);
                outputs.push(("logvar".to_string(), logvar));
                let sig_sg = builder.with_no_grad(|| sig * 1.0);
                let resid = sig_sg.squared_error(targets); // (detached sig - t)^2
                let inv_var = (-logvar).exp();
                let nll_var = (inv_var * resid + logvar) * 0.5;
                let mse = sig.squared_error(targets); // mean/trunk gradient == control
                (mse + nll_var).reduce_sum_batch()
            } else if std::env::var("D40_HETERO_NLL").is_ok() {
                let l3_var = builder.new_affine("l3_var", L3, NUM_BUCKETS);
                let logvar = l3_var.forward(hl3).select(buckets);
                outputs.push(("logvar".to_string(), logvar));
                let sq = sig.squared_error(targets); // (sig - t)^2, per position
                let inv_var = (-logvar).exp(); // exp(-log_var) = 1/sigma^2
                let nll = (inv_var * sq + logvar) * 0.5;
                nll.reduce_sum_batch()
            } else {
                // D40_LOSS_POWER (study 2026-07, experiment #2): official
                // nnue-pytorch trains |pt-qf|^2.5; bullet's default is power 2.
                // Default path stays squared_error so existing recipes reproduce
                // byte-identically; any other value routes through power_error.
                match std::env::var("D40_LOSS_POWER").as_deref() {
                    Ok(p) if p != "2" && p != "2.0" => {
                        let power: f32 = p.parse().expect("D40_LOSS_POWER must be a float");
                        sig.power_error(targets, power).reduce_sum_batch()
                    }
                    _ => sig.squared_error(targets).reduce_sum_batch(),
                }
            };
            // D47 granular metrics: report the raw components beside the fused
            // loss (run.rs reads outputs named "metric:*" per batch).
            outputs.push(("metric:value".to_string(), loss));
            // D41 rich-net: fold the WDL cross-entropy into the loss (additive,
            // λ-weighted). Gradients ADD — the WDL head does not reweight the eval
            // gradient (contrast EXP-UNC-1's exp(−s) NLL). λ→0 recovers the control.
            // D41 rich-net WDL head — created HERE (only when used): L3->3 logits off
            // the bucket-selected hl3, softmax-CE on the game-result one-hot, ADDITIVE
            // (no eval-gradient reweighting, contrast EXP-UNC-1's exp(−s) NLL). Pushed
            // as a SAVED output so l3_wdl lands in the raw ckpt for extraction.
            let loss = if wdl_lambda > 0.0 {
                let l3_wdl = builder.new_affine("l3_wdl", L3, 3);
                let wdl_logits = l3_wdl.forward(hl3);
                outputs.push(("wdl".to_string(), wdl_logits));
                let wdl_ce = wdl_logits
                    .softmax_crossentropy_loss(wdl_target)
                    .reduce_sum_rows()
                    .reduce_sum_batch();
                loss + wdl_ce * wdl_lambda
            } else {
                loss
            };
            // D41 rich-net POLICY aux head (H-AUX) — created HERE (only when used):
            // two HL->64 heads off the shared trunk hl1 forcing it to encode the
            // teacher's move preference; additive from+to CE, deploy EVAL-ONLY.
            let loss = if pol_lambda > 0.0 {
                // D47 vocabulary policy head: 1858 embedding rows dotted with
                // the 256 shared bottleneck (hl2 — THE branch point), additive
                // legal mask (-1e4 on illegal rows => the CE normalizes over
                // legal moves only), softmax-CE vs the one-hot move_id target.
                // λ (D40_POL_LAMBDA) is calibrated for POLICY≈EVAL loss parity
                // (D47; decay later). Saved to quant (polmovew/b) so netpack
                // emits the .cpol companion — the engine orders the
                // policy-argmax move first via selected-row int8 inference.
                let l_polmove = builder.new_affine("polmove", L2, 1858);
                let pol_logits = l_polmove.forward(hl2) + pol_mask;
                let pol_ce = pol_logits
                    .softmax_crossentropy_loss(pol_target)
                    .reduce_sum_rows()
                    .reduce_sum_batch();
                outputs.push(("metric:pol_ce".to_string(), pol_ce));
                loss + pol_ce * pol_lambda
            } else {
                loss
            };
            (Some(loss), outputs)
        },
    );

    let device = DefaultDevice::new(0).expect("device");

    // Held-out eval (2026-07-22): run the EXACT training graph (same mapper,
    // same loss nodes) over a dumped shadow-population .bin and report the
    // per-row value MSE + policy CE — the overfit detector. Weights from
    // D40_PROBE_RAW (f32-faithful) or D40_PROBE_QUANT (deploy-faithful).
    // For a policy net set D40_POL_LAMBDA>0 (gates the graph + quant layout);
    // leave it unset for value-only nets (the twin).
    if let Ok(dump_path) = std::env::var("D40_HELDOUT_EVAL")
        && (std::env::var("D40_PROBE_RAW").is_ok() || std::env::var("D40_PROBE_QUANT").is_ok())
    {
        let mut weights = ModelWeights::zeroed(&defn);
        let which = if let Ok(path) = std::env::var("D40_PROBE_QUANT") {
            load_quantised_weights(&mut weights, &path);
            path
        } else {
            let path = std::env::var("D40_PROBE_RAW").unwrap();
            load_raw_weights(&mut weights, &path);
            path
        };
        let mut evaluator = ModelEvaluator::new(&defn, device.clone()).expect("heldout evaluator");
        evaluator.load_weights(&weights).expect("heldout weights");
        let bytes = std::fs::read(&dump_path).expect("read D40_HELDOUT_EVAL");
        let rec = std::mem::size_of::<ChessBoard>();
        assert!(
            !bytes.is_empty() && bytes.len() % rec == 0,
            "held-out dump must be a non-empty multiple of the record size"
        );
        let n = bytes.len() / rec;
        let mut rows: Vec<ChessBoard> = vec![ChessBoard::default(); n];
        unsafe {
            std::slice::from_raw_parts_mut(rows.as_mut_ptr() as *mut u8, bytes.len())
                .copy_from_slice(&bytes)
        };
        let eval_batch = env_usize("D40_HELDOUT_BATCH", 16_384);
        let (mut value_sum, mut pol_sum) = (0f64, 0f64);
        for batch in rows.chunks(eval_batch) {
            let host = mapper.map(batch, Step::default(), batch.len());
            let input = host.to_device(&device).expect("heldout inputs to device");
            let output = evaluator.evaluate(&input).expect("heldout evaluate");
            let scalar = |name: &str| -> f64 {
                let TValue::F32(v) = output
                    .get(name)
                    .unwrap_or_else(|| panic!("missing output {name}"))
                    .to_host()
                    .expect("metric to host")
                else {
                    panic!("metric {name} must be f32");
                };
                f64::from(v[0])
            };
            value_sum += scalar("metric:value");
            if *POL_ON {
                pol_sum += scalar("metric:pol_ce");
            }
        }
        let pol = if *POL_ON { pol_sum / n as f64 } else { f64::NAN };
        println!(
            "heldout\trows {n}\tvalue_mse {:.9}\tpol_ce {pol:.9}\tckpt {which}",
            value_sum / n as f64
        );
        return;
    }

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
    // Warm-start interaction (D40_FACTORIZE): a checkpoint's l0_pstw must match
    // the graph's l0_pstw size. A factorized graph expects the WIDENED tensor
    // ((HALFKA_INPUTS + 704) × HL), so it can only resume from a factorized raw;
    // warming from a NON-factorized net (l0_pstw == HALFKA_INPUTS × HL, or a
    // merged quant) size-mismatches and is rejected below by ModelWeights. To
    // warm a factorized run from a non-factorized net you would load the real
    // columns and zero-init the 704 factoriser columns (not wired — the
    // factorized recipe trains FRESH). D40_INIT_QUANT is unusable with
    // D40_FACTORIZE (its l0_pstw reader is the merged, non-widened shape).
    if let Ok(path) = std::env::var("D40_INIT_QUANT") {
        load_quantised_weights(&mut weights, &path);
        println!("initialised from quantized checkpoint {path}");
    } else if let Ok(path) = std::env::var("D40_INIT_RAW") {
        load_raw_weights(&mut weights, &path);
        println!("initialised from raw checkpoint {path}");
    }
    // AdamW default clip +/-1.98 => i8 L1/threat weights * 64 <= 127 (no overflow).
    // D40_WEIGHT_DECAY overrides the default 0.01 (e.g. 0 for a finetune that must
    // not shrink weights / output scale). Default preserves from-scratch behaviour.
    let mut params = AdamWParams::default();
    if let Ok(wd) = std::env::var("D40_WEIGHT_DECAY") {
        params.decay = wd.parse().expect("D40_WEIGHT_DECAY must be f32");
        println!("weight_decay override: {}", params.decay);
    }
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

    // Factorization folds the 704-wide factoriser back into the real HalfKA FT
    // here, BEFORE quantisation, so the emitted quant checkpoint (and the .cenn
    // netpack.py packs from it) are shape-identical to a non-factorized net — no
    // netpack change, unchanged inference. Default path is byte-identical.
    let l0_pstw_saved = if factorize {
        SavedFormat::id("l0_pstw")
            .transform(merge_factoriser)
            .round()
            .quantise::<i16>(255)
    } else {
        SavedFormat::id("l0_pstw").round().quantise::<i16>(255)
    };
    // HalfKA-only nets (D41 drop-threats arch) never save the threat transformer.
    // The tensor is still built and "trained" inert — its inputs are all inactive
    // (line 285 emits no threat features), so it gets zero gradient and stays at its
    // zero init — but OMITTING it from the saved quant.bin is what makes the packed
    // .cenn genuinely threatless: netpack (D40_HALFKA_ONLY=1) and the engine loader
    // (flags=0) both expect NO threat bytes, and the engine then skips the threat
    // accumulator entirely (the NPS win). Saving a zero threat tensor instead would
    // both misalign netpack's payload AND still cost the full threat-update NPS.
    let mut fmt = vec![
        l0_pstw_saved,
        SavedFormat::id("l0_pstb").round().quantise::<i16>(255),
    ];
    if !halfka_only() {
        fmt.push(SavedFormat::id("l0_threatw").round().quantise::<i8>(64));
    }
    // D47: the deployed policy head rides the quant ckpt so netpack can emit
    // the .cpol companion. Transposed to row-major [1858][L2]; int8 q64 —
    // ordering needs relative scores only, so the exact scale is uncritical.
    if *POL_ON {
        fmt.push(SavedFormat::id("polmovew").transpose().round().quantise::<i8>(64));
        fmt.push(SavedFormat::id("polmoveb"));
    }
    fmt.push(SavedFormat::id("l1w").transpose().round().quantise::<i8>(64));
    fmt.push(SavedFormat::id("l1b"));
    fmt.push(SavedFormat::id("l2w").transpose());
    fmt.push(SavedFormat::id("l2b"));
    fmt.push(SavedFormat::id("l3w").transpose());
    fmt.push(SavedFormat::id("l3b"));

    println!(
        "--- D40 training: superbatches {start_superbatch}..={end_superbatch} x {batches_per_superbatch} batches x {batch_size} ---"
    );
    println!(
        "--- D40 targets: scale={target_scale} wdl={wdl} score_multiplier={score_multiplier} pol_lambda={} seed={seed} ---",
        env_f32("D40_POL_LAMBDA", 0.0)
    );
    println!("--- D40 HalfKA-only ablation: {} ---", halfka_only());
    if factorize {
        println!(
            "--- D40 king-bucket FACTORIZATION: ON (halfka FT {HALFKA_INPUTS}+{FACTOR_INPUTS} \
             inputs, {halfka_nnz} active/persp; folded to {HALFKA_INPUTS} at export) ---"
        );
    }
    let save_out = out_dir.clone();
    let save_id = net_id.clone();
    // EXP-UNC-1 FROZEN_VAR: freeze the mean/trunk tensors so ONLY l3_var trains
    // (warm-started from the incumbent → mean == incumbent). Empty set otherwise,
    // which makes train_frozen behave exactly like train (byte-identical).
    let frozen_tensors: BTreeSet<String> = if std::env::var("D40_FROZEN_VAR").is_ok() {
        ["l0_pstw", "l0_pstb", "l0_threatw", "l1w", "l1b", "l2w", "l2b", "l3w", "l3b"]
            .iter()
            .map(|name| name.to_string())
            .collect()
    } else {
        BTreeSet::new()
    };
    train_frozen(
        &mut optimiser,
        schedule,
        loader,
        &frozen_tensors,
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
                // T5 (2026-07-17): raw f32 at EVERY save point, not only the
                // end — raw is the faithful resume source (D40_INIT_RAW +
                // D40_START_SUPERBATCH; quantised init loses precision).
                // Keep only the newest intermediate raw: they're ~4x the
                // quant size and only the latest matters for resume.
                let raw_path = format!("{save_out}/{save_id}-sb{sb}-raw.bin");
                let mut raw = std::fs::File::create(&raw_path).expect("create sb raw checkpoint");
                weights.write_into(&mut raw).expect("write sb raw checkpoint");
                println!("  saved {raw_path} (resume source)");
                if sb > save_rate {
                    let prev = format!("{save_out}/{save_id}-sb{}-raw.bin", sb - save_rate);
                    let _ = std::fs::remove_file(&prev);
                }
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
