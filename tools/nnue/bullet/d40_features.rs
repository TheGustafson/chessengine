/*
 * D40 custom-feature scaffold (task #28 part 2; design/11 §12.1/§12.6/§12.7).
 * Scratchpad-only scaffold (discardable, D15/D28).
 *
 * TWO deliverables, both compiling + running on a tiny sample:
 *
 *  A. `HalfKaThreats`: the custom `SparseInputType` skeleton over the CONCATENATED
 *     0..83,248 index space (HalfKAv2_hm 22,528 in [0,22528) + FullThreats 60,720
 *     in [22528,83248)). The EXACT index math is a documented TODO -- the canonical
 *     HalfKAv2_hm + FullThreats index functions (OrientTBL polarities, 32 king
 *     buckets, threat LUT geometry) are being implemented engine-side (task #29,
 *     design/11 §12.9) and will be ported in / shared. The placeholder emits
 *     structurally-valid indices in the two subspaces so the graph runs on real
 *     sparse gathers.
 *
 *  B. The TWO-summed-tensor FT graph (design/11 §12.1/§12.6): the HalfKA weights
 *     (i16) and threat weights (i8) live in SEPARATE affine tensors that are
 *     summed in the graph (`node + node`) into one accumulator, so they (1)
 *     quantize separately (i16 / i8 -- the D40 lean hybrid) and (2) match the
 *     inference head's SEPARATE pst/threat accumulators (§12.3). Built on the
 *     low-level `ModelDefinition` + `AdamW::new` + `run::train` path (the same
 *     path `ValueTrainer` builds internally; see examples/tests/test4.rs).
 *
 * This is NOT a real net: the placeholder index math means the learned weights
 * are meaningless. It proves the STRUCTURE + graph + separate quant compile+run.
 */
use bullet_lib::game::inputs::{Chess768, SparseInputType};
use bullet_lib::nn::InitSettings;
use bulletformat::ChessBoard;
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{train, DefaultDevice, TrainingSchedule, TrainingSteps},
};

// --- Feature-space layout (design/11 §12.1) ---------------------------------
const HALFKA_INPUTS: usize = 22_528; // 64 * 704 / 2
const THREAT_INPUTS: usize = 60_720; // FullThreats
const TOTAL_INPUTS: usize = HALFKA_INPUTS + THREAT_INPUTS; // 83_248
const HALFKA_MAX_ACTIVE: usize = 32; // <=32 HalfKA features / perspective
const THREAT_MAX_ACTIVE: usize = 128; // <=128 threat features / perspective
const MAX_ACTIVE: usize = HALFKA_MAX_ACTIVE + THREAT_MAX_ACTIVE; // 160

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
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // ================= TODO(task #29 / design/11 §12.9) =================
        // Replace this ENTIRE body with the SF-exact index math, shared with the
        // engine inference front-end:
        //   HalfKAv2_hm:  make_index(persp) =
        //       (sq ^ OrientTBL[ksq] ^ 56*persp) + PieceSquareIndex[persp][pc]
        //       + KingBuckets[ksq ^ 56*persp]                 (PS_NB=704, 32 buckets)
        //   FullThreats:  attacker-block + (victim-colour,relation) sub-block
        //       + from-square base + ordinal-of-`to`          (own OrientTBL polarity)
        // Placeholder below derives in-range indices from the 768 base features so
        // both subspaces activate with real gathers (weights are meaningless).
        // ===================================================================
        Chess768.map_features(pos, |stm768, ntm768| {
            // HalfKA subspace [0, HALFKA_INPUTS): one feature / piece.
            f(stm768 % HALFKA_INPUTS, ntm768 % HALFKA_INPUTS);
            // Threat subspace [HALFKA_INPUTS, TOTAL_INPUTS): one placeholder / piece.
            let th_stm = HALFKA_INPUTS + (stm768.wrapping_mul(31) % THREAT_INPUTS);
            let th_ntm = HALFKA_INPUTS + (ntm768.wrapping_mul(31) % THREAT_INPUTS);
            f(th_stm, th_ntm);
        });
    }

    fn shorthand(&self) -> String {
        format!("HalfKA+Threats({TOTAL_INPUTS})")
    }

    fn description(&self) -> String {
        "SF HalfKAv2_hm (22528) + FullThreats (60720), two separately-quantised FT tensors".to_string()
    }
}

// ============================================================================
// Synthetic data: a few real positions cycled to fill tiny batches.
// ============================================================================
#[derive(Clone)]
struct CyclingReader {
    data: Vec<ChessBoard>,
    chunk: usize,
}

impl DataReader<ChessBoard> for CyclingReader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _skip_count: usize, mut f: F) {
        let mut i = 0;
        loop {
            let chunk: Vec<ChessBoard> = (0..self.chunk)
                .map(|_| {
                    let d = self.data[i % self.data.len()];
                    i += 1;
                    d
                })
                .collect();
            if f(&chunk) {
                break;
            }
        }
    }
}

fn synthetic_positions() -> Vec<ChessBoard> {
    // fen | white-relative-score-cp | result(0/0.5/1)
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 | 25 | 0.5",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5 | 15 | 1.0",
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1 | 320 | 1.0",
    ];
    fens.iter().map(|s| s.parse::<ChessBoard>().expect("valid synthetic fen")).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let feats = HalfKaThreats;
    println!("=== D40 feature scaffold ===");
    println!("input type   : {}", feats.description());
    println!("num_inputs   : {} (expected {TOTAL_INPUTS})", feats.num_inputs());
    println!("max_active   : {} (expected {MAX_ACTIVE})", feats.max_active());
    assert_eq!(feats.num_inputs(), TOTAL_INPUTS);
    assert_eq!(feats.max_active(), MAX_ACTIVE);

    // Scaffold self-check: features split cleanly into the two subspaces.
    let pos = synthetic_positions()[0];
    let (mut n_hk, mut n_th) = (0usize, 0usize);
    feats.map_features(&pos, |stm, ntm| {
        assert!(stm < TOTAL_INPUTS && ntm < TOTAL_INPUTS, "index out of range");
        let stm_is_hk = stm < HALFKA_INPUTS;
        let ntm_is_hk = ntm < HALFKA_INPUTS;
        assert_eq!(stm_is_hk, ntm_is_hk, "a pair must lie in ONE subspace");
        if stm_is_hk {
            n_hk += 1
        } else {
            n_th += 1
        }
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
            // Synthetic eval target = sigmoid(score / 400).
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

    let reader = CyclingReader { data: synthetic_positions(), chunk: batch_size };
    let loader = ReadMapLoader::new(reader, mapper, 2);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size, batches_per_superbatch: 2, start_superbatch: 1, end_superbatch: 1 },
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
            println!("  step {} (sb {}, batch {}): loss {err:.6}", steps, step.superbatch(), step.batch());
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
    println!("separate-quant buffer: {} bytes (i16 HalfKA-FT + i8 threat-FT + f32 tail)", buf.len());
    println!("=== d40_features: STRUCTURE + two-tensor graph + separate quant OK ===");
}
