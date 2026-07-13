/*
 * D40 policy-trainer scaffold (task #28 part 3; design/11 §12.5/§12.7).
 * Scratchpad-only scaffold (discardable, D15/D28).
 *
 * Drops from the convenience `ValueTrainer` to the LOW-LEVEL trainer path
 * (`ModelDefinition` + `AdamW::new` + `run::train`) -- the same path
 * `ValueTrainer` builds internally (crates/bullet_lib/src/value.rs is the
 * template; examples/tests/test4.rs is a minimal worked example) -- to add:
 *
 *   - a from x to (4096) POLICY head off the SHARED pairwise trunk (§12.5:
 *     the policy head reads the post-pairwise 768 transformed features, so its
 *     gradient shapes the trunk -- the "eval+full" grid member);
 *   - a COMBINED loss  = eval_loss + lambda * softmax_crossentropy(policy_target)
 *     over a named `policy_target` dense data input (via `ModelInputs.add_dense`).
 *
 * Proven here on a TINY SYNTHETIC policy target (hand-made distributions over a
 * few positions): the combined-loss graph compiles and runs training steps.
 * The REAL Lc0 soft-policy corpus + its DataReader are task #27 (in progress);
 * this scaffold deliberately does NOT block on it -- swap the synthetic reader
 * for the real one when it lands.
 */
use bullet_lib::game::inputs::{Chess768, SparseInputType};
use bulletformat::ChessBoard;
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, ReadMapLoader},
    run::{train, DefaultDevice, TrainingSchedule, TrainingSteps},
};

const NNZ: usize = 32; // Chess768 max active
const POLICY_DIM: usize = 4096; // from x to
const LAMBDA_POLICY: f32 = 1.0; // eval_loss + LAMBDA_POLICY * policy_ce (an SPSA knob)

// A position with an attached synthetic Lc0-style soft-policy target.
#[derive(Clone)]
struct PolicyDatum {
    board: ChessBoard,
    /// (from*64 + to, probability); a proper distribution (sums to ~1).
    policy: Vec<(usize, f32)>,
}

#[derive(Clone)]
struct CyclingReader {
    data: Vec<PolicyDatum>,
    chunk: usize,
}

impl DataReader<PolicyDatum> for CyclingReader {
    fn read_chunks<F: FnMut(&[PolicyDatum]) -> bool>(&self, _skip_count: usize, mut f: F) {
        let mut i = 0;
        loop {
            let chunk: Vec<PolicyDatum> = (0..self.chunk)
                .map(|_| {
                    let d = self.data[i % self.data.len()].clone();
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

fn ft(from: usize, to: usize) -> usize {
    from * 64 + to
}

fn synthetic_data() -> Vec<PolicyDatum> {
    // fen | white-relative-score-cp | result. Policy = hand-made soft distribution
    // over a few (from,to) moves (legality irrelevant for a graph scaffold).
    let make = |fen: &str, score: i32, res: f32, policy: Vec<(usize, f32)>| PolicyDatum {
        board: format!("{fen} | {score} | {res}").parse::<ChessBoard>().expect("valid fen"),
        policy,
    };
    vec![
        make(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0,
            0.5,
            // e2e4, d2d4, g1f3
            vec![(ft(12, 28), 0.5), (ft(11, 27), 0.3), (ft(6, 21), 0.2)],
        ),
        make(
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            25,
            0.5,
            // f1b5, f1c4, b1c3
            vec![(ft(5, 33), 0.45), (ft(5, 26), 0.35), (ft(1, 18), 0.2)],
        ),
        make(
            "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
            320,
            1.0,
            // e2e4, e3d4, e3f4
            vec![(ft(12, 28), 0.6), (ft(20, 27), 0.2), (ft(20, 29), 0.2)],
        ),
    ]
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let hl_size = 768usize; // shared FT accumulator width (even; pairwise halves it)
    let batch_size = 256usize;

    println!("=== D40 policy trainer scaffold ===");
    println!("trunk FT={hl_size} -> pairwise 768 transformed features (shared)");
    println!("eval head: 768->16->32->1   policy head: 768->{POLICY_DIM} (from x to)");
    println!("combined loss: eval_mse + {LAMBDA_POLICY} * softmax_ce(policy_target)");

    // stm/nstm sparse (Chess768) + dense eval target + dense POLICY target.
    let inputs = ModelInputs::default()
        .add_sparse("stm", (768, 1), NNZ)
        .add_sparse("nstm", (768, 1), NNZ)
        .add_dense("targets", (1, 1))
        .add_dense("policy_target", (POLICY_DIM, 1));

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |datum: &PolicyDatum, _step, (((stm, nstm), targets), policy)| {
            let mut cnt = 0;
            Chess768.map_features(&datum.board, |our, opp| {
                stm[cnt] = our as i32;
                nstm[cnt] = opp as i32;
                cnt += 1;
            });
            stm[cnt..].iter_mut().for_each(|s| *s = -1);
            nstm[cnt..].iter_mut().for_each(|s| *s = -1);

            // Eval target = sigmoid(score / 400) (WDL blend omitted for the scaffold).
            targets[0] = sigmoid(f32::from(datum.board.score) / 400.0);

            // Policy target = the soft distribution (DenseInput is pre-zeroed).
            for &(idx, prob) in &datum.policy {
                policy[idx] = prob;
            }
        },
    );

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((stm, nstm), targets), policy_target)| {
            let l0 = builder.new_affine("l0", 768, hl_size);
            let l1 = builder.new_affine("l1", hl_size, 16);
            let l2 = builder.new_affine("l2", 16, 32);
            let l3 = builder.new_affine("l3", 32, 1);
            // from x to policy head off the shared trunk.
            let policy = builder.new_affine("policy", hl_size, POLICY_DIM);

            // Shared pairwise trunk (STM-half first).
            let stm_hidden = l0.forward(stm).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(nstm).crelu().pairwise_mul();
            let trunk = stm_hidden.concat(ntm_hidden); // 768 transformed features

            // Eval head + MSE-on-sigmoid loss (per sample, (1,1)).
            let hl2 = l1.forward(trunk).screlu();
            let hl3 = l2.forward(hl2).screlu();
            let eval_out = l3.forward(hl3);
            let eval_loss = eval_out.sigmoid().squared_error(targets);

            // Policy head + softmax cross-entropy to the soft target.
            // softmax_crossentropy_loss returns the per-logit -t*log(softmax); sum
            // over the 4096 rows for the per-sample CE, then (1,1).
            let policy_logits = policy.forward(trunk);
            let policy_loss = policy_logits.softmax_crossentropy_loss(policy_target).reduce_sum_rows();

            // Combined loss (operator-overload combine), then reduce over the batch.
            let loss = (eval_loss + LAMBDA_POLICY * policy_loss).reduce_sum_batch();

            (Some(loss), vec![("output".to_string(), eval_out), ("policy".to_string(), policy_logits)])
        },
    );

    let weights = ModelWeights::new(&defn, 0x00D4_0901);
    let device = DefaultDevice::new(0).expect("device");
    let mut optimiser = AdamW::new(defn, weights, device, AdamWParams::default()).expect("optimiser");

    let reader = CyclingReader { data: synthetic_data(), chunk: batch_size };
    let loader = ReadMapLoader::new(reader, mapper, 2);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size, batches_per_superbatch: 4, start_superbatch: 1, end_superbatch: 1 },
        log_rate: 1,
        lr_schedule: Box::new(|_| 0.001),
    };

    println!("--- training the combined eval+policy graph (1 superbatch x 4 batches) ---");
    let mut steps = 0u32;
    train(
        &mut optimiser,
        schedule,
        loader,
        |_, step, err| {
            steps += 1;
            println!("  step {} (sb {}, batch {}): combined loss {err:.6}", steps, step.superbatch(), step.batch());
        },
        |_, _| {},
    )
    .expect("training step");

    assert!(steps >= 1, "no training step ran");
    println!("=== d40_policy: combined eval+policy graph compiled + ran {steps} steps OK ===");
}
