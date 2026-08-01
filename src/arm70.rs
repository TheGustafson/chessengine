//! Perf study 70 — the move-ordering ARM BENCH (feature `arm70`, never
//! shipped).
//!
//! The study's metric is **nodes to a fixed depth**, which is a discrete,
//! deterministic integer: it does not care how fast the binary runs. That
//! makes a *runtime-configurable* screening binary strictly better than one
//! feature per arm — an `env` read costs speed the metric cannot see, and one
//! build screens a dozen ordering hypotheses instead of one.
//!
//! Every knob defaults to the shipped value, so `arm70` with no environment
//! set must reproduce the shipped node count exactly (checked before any arm
//! is read: that is the study's null).
//!
//! Set with `ARCHON_A70="lva=8,qsort=1"` etc.

use std::sync::LazyLock;

#[derive(Debug, Clone, Copy)]
pub struct Arm {
    /// Least-valuable-attacker tiebreak on captures: subtract
    /// `attacker_value / lva` from the capture ordering score. 0 = off
    /// (shipped).
    pub lva: i32,
    /// Force a FULL quiet sort instead of the depth-scaled partial one.
    /// The upper bound on "sorting completeness" as an ordering lever.
    pub qsort_full: bool,
    /// Override for `QUIET_SORT_DEPTH_FACTOR` (0 = keep shipped).
    pub qsort_factor: i32,
    /// Multiplier (in 16ths) on the first continuation-history plane in the
    /// quiet score. 16 = shipped.
    pub c1_w: i32,
    /// Multiplier (in 16ths) on the butterfly history term. 16 = shipped
    /// (which is already a weight of 2).
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
    /// every quiet ordering score. At `jitter=1` this is an INERT
    /// perturbation — it can only re-break exact ties among moves whose whole
    /// history stack agrees to the unit — so whatever it moves the node count
    /// by is the metric's own chaos floor on that position set, not an
    /// ordering effect. Rule 1: run this before believing any arm.
    pub jitter: i32,
    /// Seed for the jitter, so the null is a DISTRIBUTION rather than one
    /// draw (one draw of a chaotic statistic proves nothing).
    pub jseed: i32,

    // --- history UPDATE shape: what the ordering learns from a cutoff ---
    /// Override `QUIET_MALUS_START_NUM` (1024ths). 0 = shipped.
    pub malus_start: i32,
    /// Override `QUIET_MALUS_DECAY_NUM` (1024ths). 0 = shipped. 1024 makes
    /// the malus UNIFORM across the tried quiets instead of decaying, and
    /// values above 1024 invert the shape (late failures punished hardest).
    pub malus_decay: i32,
    /// Override `QUIET_BEST_BONUS_NUM` (1024ths). 0 = shipped.
    pub best_bonus: i32,
    /// Override `STAT_BONUS_TTMOVE`. -1 = shipped.
    pub tt_bonus: i32,
    /// **Defended-destination penalty on captures (16ths of the attacker's
    /// value).** Shipped capture ordering is `7*victim + capture_history` with
    /// NO attacker term at all — the good/bad SEE split is supposed to price
    /// attacker risk, but that split does not run in QSEARCH, where 90% of
    /// cutoffs come from the raw MVV order. One cached bitboard test
    /// (`all_threats`, already filled for every evaluated node) says whether a
    /// recapture is coming.
    pub def_cap: i32,
    /// **Override for the SHIPPED `QS_CAPTURE_DEFENDED_NUM`** (perf study 70's
    /// landed quiescence recapture discount). `-1` keeps the shipped 56; `0`
    /// turns it off, which is how the landed change is A/B'd against its own
    /// predecessor inside one binary.
    pub qs_def: i32,
    /// Use a PAWN-ONLY recapture map instead of the full threat map. The full
    /// map is `Board::compute_threats`, which is lazy precisely because
    /// quiescence never needed it — arming it there costs a fill per qsearch
    /// node. Pawn defenders are two shifts.
    pub def_pawn: bool,
    /// Restore the pre-study evasion capture score: the raw victim value with
    /// no capture history. The OFF switch for the other landed change.
    pub evasion_mvv: bool,
    /// **Override for the shipped `TT_FAILLOW_MOVE`** (perf study 71): store
    /// the argmax move at a main-search fail low. `-1` keeps the shipped
    /// value, `0` off, `1` on.
    pub fl_move: i32,
    /// **Override for the shipped `TT_UPPER_PRUNE_BLIND`** (perf study 71):
    /// the pruning/reduction gates that read `ttMove == NULL` as "this node is
    /// unresolved" read NULL on an UPPER-bound entry. `-1` shipped, `0` off,
    /// `1` on.
    ///
    /// This is a knob in its OWN right, not a modifier of `fl_move`, and that
    /// is the point: study 70's standing finding is that a node win is a
    /// pruning win until a magnitude control says otherwise, so the fail-low
    /// store must be screened as a 2x2 — `ttup=1` alone is the pruning-only
    /// control, and the ORDERING effect of the store is
    /// `(flmove=1,ttup=1) - (ttup=1)`.
    pub tt_up_blind: i32,
    /// Study 71: the same store in QUIESCENCE — a qsearch fail-low knows
    /// which capture scored highest and throws it away too (21.9 % of the
    /// NonPV moveless population, 54.6 % of the qsearch one).
    pub qfl_move: i32,
    /// Give the tried-but-not-best CAPTURES a malus even when the best move
    /// was a QUIET, and the tried quiets a malus when the best was a capture.
    /// Shipped: each kind is only punished when the best move shares its kind,
    /// so half the evidence at every node is discarded.
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
            other => panic!("arm70: unknown knob {other:?}"),
        }
    }
    a
}

/// Read once. A `LazyLock` rather than a per-call `env::var` because the
/// scoring loops run per move; the metric cannot see the difference, but a
/// syscall per quiet would make the screen needlessly slow.
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

/// The jitter value for one move under the current arm. A multiplicative
/// hash of `(raw16, seed)` — deterministic, so a jittered run is exactly
/// reproducible, and uncorrelated with anything the ordering cares about.
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
