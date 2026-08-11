//! The move-ordering ARM BENCH (feature `arm70`, never shipped). Set with
//! `ARCHON_A70="lva=8,qsort=1"`.
//!
//! Every knob defaults to the shipped value, so `arm70` with no environment set
//! must reproduce the shipped node count exactly — check that null before
//! reading any arm.

use std::sync::LazyLock;

#[derive(Debug, Clone, Copy)]
pub struct Arm {
    /// Least-valuable-attacker tiebreak on captures: subtract
    /// `attacker_value / lva` from the capture ordering score. 0 = off.
    pub lva: i32,
    /// Force a FULL quiet sort instead of the depth-scaled partial one.
    pub qsort_full: bool,
    /// Override for `QUIET_SORT_DEPTH_FACTOR` (0 = keep shipped).
    pub qsort_factor: i32,
    /// Multiplier (in 16ths) on the first continuation-history plane in the
    /// quiet score. 16 = shipped.
    pub c1_w: i32,
    /// Multiplier (in 16ths) on the butterfly history term. 16 = shipped
    /// (already a weight of 2).
    pub main_w: i32,
    /// Add capture history to evasion captures (shipped: pure MVV).
    pub evasion_caphist: bool,
    /// Score evasion quiets with the full continuation stack + pawn history
    /// (shipped: main + cont1 only).
    pub evasion_full: bool,
    /// Per-16ths multiplier on the quiet threat-escape/entry term.
    pub threat_w: i32,
    /// Per-16ths multiplier on the quiet offense term.
    pub offense_w: i32,
    /// **The null.** Add a deterministic pseudo-random `[-jitter, +jitter]` to
    /// every quiet ordering score. At `jitter=1` it is INERT (it can only
    /// re-break exact ties), so whatever it moves the node count by is the
    /// metric's own chaos floor. Run it before believing any arm.
    pub jitter: i32,
    /// Seed for the jitter, so the null is a DISTRIBUTION rather than one draw.
    pub jseed: i32,

    // --- history UPDATE shape: what the ordering learns from a cutoff ---
    /// Override `QUIET_MALUS_START_NUM` (1024ths). 0 = shipped.
    pub malus_start: i32,
    /// Override `QUIET_MALUS_DECAY_NUM` (1024ths). 0 = shipped; 1024 makes the
    /// malus UNIFORM across the tried quiets instead of decaying, and above
    /// 1024 inverts the shape (late failures punished hardest).
    pub malus_decay: i32,
    /// Override `QUIET_BEST_BONUS_NUM` (1024ths). 0 = shipped.
    pub best_bonus: i32,
    /// Override `STAT_BONUS_TTMOVE`. -1 = shipped.
    pub tt_bonus: i32,
    /// Defended-destination penalty on captures (16ths of the attacker's
    /// value). Shipped capture ordering is `7*victim + capture_history` with no
    /// attacker term, and the good/bad SEE split that prices attacker risk does
    /// not run in QSEARCH.
    pub def_cap: i32,
    /// Override for the shipped `QS_CAPTURE_DEFENDED_NUM` (the quiescence
    /// recapture discount). `-1` keeps the shipped 56; `0` turns it off.
    pub qs_def: i32,
    /// Use a PAWN-ONLY recapture map instead of the full threat map
    /// (`Board::compute_threats` is lazy, so arming it in qsearch costs a fill
    /// per node).
    pub def_pawn: bool,
    /// Restore the pre-study evasion capture score: raw victim value with no
    /// capture history.
    pub evasion_mvv: bool,
    /// Override for the shipped `TT_FAILLOW_MOVE`: store the argmax move at a
    /// main-search fail low. `-1` shipped, `0` off, `1` on.
    pub fl_move: i32,
    /// Override for the shipped `TT_UPPER_PRUNE_BLIND`: the pruning/reduction
    /// gates that read `ttMove == NULL` as "this node is unresolved" read NULL
    /// on an UPPER-bound entry. `-1` shipped, `0` off, `1` on.
    ///
    /// Independent of `fl_move` on purpose: `ttup=1` alone is the pruning-only
    /// control, so the fail-low store's ORDERING effect is
    /// `(flmove=1,ttup=1) - (ttup=1)`.
    pub tt_up_blind: i32,
    /// **The TT-less deficit's own null.** Hide the TT move from the MOVE
    /// PICKER only; every pruning, reduction and extension gate still reads
    /// `tt_data.mv`, so the only thing that changes is the ordering's best
    /// hint. Separates that hint's value from the fact that a node WITH a TT
    /// move is one the search already visited, and so is intrinsically easier.
    pub hide_tt: bool,
    /// A real SEE term on quiet ordering: subtract `qsee` from a quiet's score
    /// when `see_ge(m, 0)` fails. `-1` keeps the shipped [`QUIET_SEE_PENALTY`],
    /// `0` turns the term OFF.
    ///
    /// [`QUIET_SEE_PENALTY`]: crate::search::params::QUIET_SEE_PENALTY
    pub qsee: i32,
    /// Turn OFF the shipped guard ("the destination is in the enemy attack
    /// map"), which is not EXACT: a discovered attacker through the vacated
    /// from-square can attack the destination only after the move.
    pub qsee_guard: bool,
    /// **The size-matched null for `qsee`.** Demote a deterministic
    /// pseudo-random `qsee_rate/1024` share of quiets by the same amount. Any
    /// big demotion raises `fhf` for a PRUNING reason (it can push a quiet
    /// below `GOOD_QUIET_THRESHOLD`, where late-move pruning can drop it), so
    /// matching share and magnitude is what isolates the SEE signal.
    pub qsee_rand: i32,
    pub qsee_rate: i32,
    /// Value-scaled variant: penalty = `qsee_pv * PieceValue[mover] / 16`.
    pub qsee_pv: i32,
    /// SEE threshold: the test is `see_ge(m, qsee_thr)`. 0 = "does not lose
    /// material".
    pub qsee_thr: i32,
    /// The free approximation to `qsee`: penalise a quiet whose destination is
    /// attacked by a piece of EQUAL-or-lesser value, using the threat tiers
    /// `init_quiets` already builds (the shipped term knows only STRICTLY
    /// lesser attackers). No `see_ge`, but blind to defenders.
    pub qeq: i32,
    /// Additionally require the quiet to be ABOVE `GOOD_QUIET_THRESHOLD` before
    /// paying for `see_ge` — demoting a quiet already in the bad pile only
    /// reorders the tail, and skipping those saves most of the SEE calls.
    pub qsee_gq: bool,
    /// Only score-penalise quiets whose mover is a pawn (`1`) or is NOT a pawn
    /// (`2`). The shipped threat tier is `Bitboard::EMPTY` for a pawn.
    pub qsee_pt: i32,
    /// Apply the term only at `depth >= qsee_depth`.
    pub qsee_depth: i32,
    /// **Clamp the demotion at `GOOD_QUIET_THRESHOLD + 1`.** Unclamped, the
    /// penalty is also a PRUNING change: crossing the threshold moves the quiet
    /// into the BAD pile, where late-move pruning can drop it. Clamped, it is
    /// pure reordering inside the good pile.
    pub qsee_clamp: bool,
    /// The same fail-low store in QUIESCENCE.
    pub qfl_move: i32,
    /// Give the tried-but-not-best CAPTURES a malus even when the best move was
    /// a QUIET, and vice versa. Shipped: each kind is only punished when the
    /// best move shares its kind.
    pub cross_malus: i32,
}

pub const SHIPPED: Arm = Arm {
    lva: 0,
    qsort_full: false,
    qsort_factor: 0,
    c1_w: 16,
    main_w: 16,
    evasion_caphist: false,
    evasion_full: false,
    threat_w: 16,
    offense_w: 16,
    jitter: 0,
    jseed: 0,
    malus_start: 0,
    malus_decay: 0,
    best_bonus: 0,
    tt_bonus: -1,
    def_cap: 0,
    qs_def: -1,
    def_pawn: false,
    evasion_mvv: false,
    cross_malus: 0,
    fl_move: -1,
    qfl_move: -1,
    tt_up_blind: -1,
    hide_tt: false,
    qsee: -1,
    qsee_guard: true,
    qsee_rand: 0,
    qsee_rate: 0,
    qsee_pv: 0,
    qsee_thr: 0,
    qeq: 0,
    qsee_gq: false,
    qsee_pt: 0,
    qsee_depth: 0,
    qsee_clamp: false,
};

fn parse(spec: &str) -> Arm {
    let mut a = SHIPPED;
    for kv in spec.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap_or((kv, "1"));
        let n: i32 = v.trim().parse().unwrap_or_else(|_| {
            panic!("arm70: bad value {v:?} in {kv:?}");
        });
        match k.trim() {
            "lva" => a.lva = n,
            "qsortfull" => a.qsort_full = n != 0,
            "qsortfactor" => a.qsort_factor = n,
            "c1w" => a.c1_w = n,
            "mainw" => a.main_w = n,
            "evcap" => a.evasion_caphist = n != 0,
            "evfull" => a.evasion_full = n != 0,
            "threatw" => a.threat_w = n,
            "offensew" => a.offense_w = n,
            "jitter" => a.jitter = n,
            "jseed" => a.jseed = n,
            "malusstart" => a.malus_start = n,
            "malusdecay" => a.malus_decay = n,
            "bestbonus" => a.best_bonus = n,
            "ttbonus" => a.tt_bonus = n,
            "crossmalus" => a.cross_malus = n,
            "defcap" => a.def_cap = n,
            "qsdef" => a.qs_def = n,
            "defpawn" => a.def_pawn = n != 0,
            "evmvv" => a.evasion_mvv = n != 0,
            "flmove" => a.fl_move = n,
            "qflmove" => a.qfl_move = n,
            "ttup" => a.tt_up_blind = n,
            "hidett" => a.hide_tt = n != 0,
            "qsee" => a.qsee = n,
            "qseeg" => a.qsee_guard = n != 0,
            "qseerand" => a.qsee_rand = n,
            "qseerate" => a.qsee_rate = n,
            "qseepv" => a.qsee_pv = n,
            "qseethr" => a.qsee_thr = n,
            "qeq" => a.qeq = n,
            "qseegq" => a.qsee_gq = n != 0,
            "qseept" => a.qsee_pt = n,
            "qseedepth" => a.qsee_depth = n,
            "qseeclamp" => a.qsee_clamp = n != 0,
            other => panic!("arm70: unknown knob {other:?}"),
        }
    }
    a
}

/// Read once. A `LazyLock` rather than a per-call `env::var` because the
/// scoring loops run per move and a syscall per quiet would make the screen
/// needlessly slow.
pub static ARM: LazyLock<Arm> = LazyLock::new(|| match std::env::var("ARCHON_A70") {
    Ok(spec) => {
        let a = parse(&spec);
        eprintln!("A70 arm: {a:?}");
        a
    }
    Err(_) => SHIPPED,
});

#[inline]
pub fn arm() -> &'static Arm {
    &ARM
}

/// The jitter value for one move under the current arm. A multiplicative hash
/// of `(raw16, seed)`: deterministic, so a jittered run is reproducible, and
/// uncorrelated with anything the ordering cares about.
#[inline]
pub fn jitter_for(a: &Arm, raw16: u16) -> i32 {
    if a.jitter == 0 {
        return 0;
    }
    let mut h = (raw16 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (a.jseed as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    let span = 2 * a.jitter + 1;
    (h % span as u64) as i32 - a.jitter
}
