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
use bullet_lib::{
    game::inputs::SparseInputType, nn::InitSettings, value::loader::DirectSequentialDataLoader,
};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};
use bulletformat::ChessBoard;
use chessengine::attacks::{attacks, pawn_attacks};
use chessengine::eval::nnue::sota_features::{
    HALFKA_DIMS, THREAT_DIMS, for_each_halfka, for_each_threat, halfka_index, threat_index,
};
use chessengine::{Bitboard, Board, Color, Piece, PieceType, Square};

// --- Feature-space layout (design/11 §12.1) ---------------------------------
const HALFKA_INPUTS: usize = HALFKA_DIMS;
const THREAT_INPUTS: usize = THREAT_DIMS;
const TOTAL_INPUTS: usize = HALFKA_INPUTS + THREAT_INPUTS; // 83_248
const HALFKA_MAX_ACTIVE: usize = 32; // <=32 HalfKA features / perspective
const THREAT_MAX_ACTIVE: usize = 160;
const MAX_ACTIVE: usize = HALFKA_MAX_ACTIVE + THREAT_MAX_ACTIVE; // 192

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

fn verify_feature_parity(data: &str) {
    let reader = DirectSequentialDataLoader::new(&[data]);
    let mut positions = vec![self_check_position()];
    reader.read_chunks(0, |chunk: &[ChessBoard]| {
        positions.extend_from_slice(&chunk[..chunk.len().min(64)]);
        true
    });

    let mut max_halfka = 0usize;
    let mut max_threats = 0usize;
    for (position_number, pos) in positions.iter().enumerate() {
        let board = reconstruct_board(pos);
        let mut trainer = [Vec::<usize>::new(), Vec::<usize>::new()];
        emit_features(pos, |stm, ntm| {
            trainer[Color::White as usize].push(stm);
            trainer[Color::Black as usize].push(ntm);
        });

        for perspective in Color::ALL {
            let mut engine = Vec::new();
            for_each_halfka(&board, perspective, |idx| engine.push(idx));
            for_each_threat(&board, perspective, |idx| engine.push(HALFKA_INPUTS + idx));

            let actual = &mut trainer[perspective as usize];
            actual.sort_unstable();
            engine.sort_unstable();
            assert_eq!(
                *actual, engine,
                "trainer/engine feature mismatch at sampled position {position_number}, {perspective:?}"
            );

            let halfka = actual.partition_point(|&idx| idx < HALFKA_INPUTS);
            max_halfka = max_halfka.max(halfka);
            max_threats = max_threats.max(actual.len() - halfka);
        }
    }

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
        positions.len()
    );
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let data = std::env::var("D40_DATA")
        .expect("set D40_DATA to a bulletformat ChessBoard train.bin path");
    let feats = HalfKaThreats;
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
    verify_feature_parity(&data);

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
    let hl_size = 768usize; // D40 accumulator width (even; pairwise halves it)
    let batch_size = 256usize;

    // Four sparse inputs: HalfKA (i16 tensor) + threats (i8 tensor), each stm/ntm.
    let inputs = ModelInputs::default()
        .add_sparse("halfka_stm", (HALFKA_INPUTS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("halfka_ntm", (HALFKA_INPUTS, 1), HALFKA_MAX_ACTIVE)
        .add_sparse("threat_stm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_sparse("threat_ntm", (THREAT_INPUTS, 1), THREAT_MAX_ACTIVE)
        .add_dense("targets", (1, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |pos: &ChessBoard, _step, ((((hk_stm, hk_ntm), th_stm), th_ntm), targets)| {
            let (mut hk, mut th) = (0usize, 0usize);
            feats.map_features(pos, |stm, ntm| {
                if stm < HALFKA_INPUTS {
                    if hk < HALFKA_MAX_ACTIVE {
                        hk_stm[hk] = stm as i32;
                        hk_ntm[hk] = ntm as i32;
                        hk += 1;
                    }
                } else if th < THREAT_MAX_ACTIVE {
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
            // Real corpus eval target = sigmoid(score / 400).
            targets[0] = sigmoid(f32::from(pos.score) / 400.0);
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        |builder, ((((hk_stm, hk_ntm), th_stm), th_ntm), targets)| {
            // Two SEPARATE FT tensors (quantised i16 / i8 respectively). Per
            // design/11 §12.2 the threats transformer is seeded from ZERO and
            // carries NO bias (the shared FT bias lives on the HalfKA tensor);
            // that also keeps the two bias gradients from colliding into an
            // un-eliminable CSE copy in the backward graph.
            let l0_pst = builder.new_affine("l0_pst", HALFKA_INPUTS, hl_size);
            let l0_threat_w =
                builder.new_weights("l0_threatw", (hl_size, THREAT_INPUTS), InitSettings::Zeroed);
            let l1 = builder.new_affine("l1", hl_size, 16);
            let l2 = builder.new_affine("l2", 16, 32);
            let l3 = builder.new_affine("l3", 32, 1);

            // node + node: the two-transformer sum into one accumulator/perspective.
            let acc_stm = l0_pst.forward(hk_stm) + l0_threat_w.matmul(th_stm);
            let acc_ntm = l0_pst.forward(hk_ntm) + l0_threat_w.matmul(th_ntm);

            // D40 pairwise trunk.
            let stm_hidden = acc_stm.crelu().pairwise_mul();
            let ntm_hidden = acc_ntm.crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);

            let hl2 = l1.forward(hl1).screlu();
            let hl3 = l2.forward(hl2).screlu();
            let out = l3.forward(hl3);

            let loss = out.sigmoid().squared_error(targets).reduce_sum_batch();
            (Some(loss), vec![("output".to_string(), out)])
        },
    );

    let weights = ModelWeights::new(&defn, 0x00D4_0FEA);
    let device = DefaultDevice::new(0).expect("device");
    // AdamW default clip +/-1.98 => i8 L1/threat weights * 64 <= 127 (no overflow).
    let params = AdamWParams::default();
    let mut optimiser = AdamW::new(defn, weights, device, params).expect("optimiser");

    let reader = DirectSequentialDataLoader::new(&[data.as_str()]);
    let loader = ReadMapLoader::new(reader, mapper, 2);

    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: 2,
            start_superbatch: 1,
            end_superbatch: 1,
        },
        log_rate: 1,
        lr_schedule: Box::new(|_| 0.001),
    };

    println!("--- training the two-summed-tensor graph (1 superbatch x 2 batches) ---");
    let mut steps = 0u32;
    train(
        &mut optimiser,
        schedule,
        loader,
        |_, step, err| {
            steps += 1;
            println!(
                "  step {} (sb {}, batch {}): loss {err:.6}",
                steps,
                step.superbatch(),
                step.batch()
            );
        },
        |_, _| {},
    )
    .expect("training step");

    // Prove the two FT tensors quantise SEPARATELY (i16 pst + i8 threats), no overflow.
    let fmt = vec![
        SavedFormat::id("l0_pstw").round().quantise::<i16>(255),
        SavedFormat::id("l0_pstb").round().quantise::<i16>(255),
        SavedFormat::id("l0_threatw").round().quantise::<i8>(64),
        SavedFormat::id("l1w"),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w"),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w"),
        SavedFormat::id("l3b"),
    ];
    let buf = optimiser
        .cpu_weights()
        .expect("sync cpu weights")
        .to_quantised_buffer(&fmt, true)
        .expect("separate quant (overflow-checked)");
    println!(
        "separate-quant buffer: {} bytes (i16 HalfKA-FT + i8 threat-FT + f32 tail)",
        buf.len()
    );
    println!("=== d40_features: STRUCTURE + two-tensor graph + separate quant OK ===");
}
