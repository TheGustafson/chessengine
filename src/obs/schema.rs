//! The Observation-Store logical schema (design/16 §3.5.3) and its TLV
//! encoding. Tags are APPEND-ONLY per §3.5.2 — new fields get fresh numbers,
//! deprecated tags are tombstoned in a comment, never reused. Unknown tags
//! and unknown enum discriminants are skipped/preserved, never errors: a
//! reader built today must survive shards written years from now.
//!
//! Field presence is explicit: an absent optional field decodes to `None`,
//! and a reader never conjures a numeric zero for a missing value. The small
//! set of structurally mandatory fields (position identity, provenance
//! linkage) errors loudly when absent.

use super::wire::{FieldReader, FieldWriter};
use std::io;

fn missing(what: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("missing mandatory field: {what}"),
    )
}

// ─── Enums (u8 discriminants; unknown values preserved, not rejected) ──────

macro_rules! obs_enum {
    ($(#[$doc:meta])* $name:ident { $first:ident = $first_value:expr
        $(, $variant:ident = $value:expr)* $(,)? }) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub enum $name {
            /// The schema's zero value doubles as the Rust default.
            #[default]
            $first,
            $($variant,)*
            /// A discriminant this reader predates (§3.5.2: skip, don't error).
            Unknown(u8),
        }

        impl $name {
            pub fn to_u8(self) -> u8 {
                match self {
                    Self::$first => $first_value,
                    $(Self::$variant => $value,)*
                    Self::Unknown(v) => v,
                }
            }

            pub fn from_u8(value: u8) -> Self {
                match value {
                    $first_value => Self::$first,
                    $($value => Self::$variant,)*
                    other => Self::Unknown(other),
                }
            }
        }
    };
}

obs_enum!(
    /// Episode families (design/16 §3.5.3); extensible.
    EpisodeKind {
        SelfPlay = 0,
        InternalSearch = 1,
        RootMoveAnalysis = 2,
        Counterfactual = 3,
        Adversarial = 4,
        // In-search observation capture (design/16 §3.5.5, D45a): interior
        // main-search nodes sampled live during real games — the concrete
        // engine-embedded realization of the `InternalSearch` mode, kept as a
        // distinct discriminant so game-embedded capture is provenance-separable
        // from a future offline `InternalSearch` generator.
        InSearchCapture = 5,
    }
);

obs_enum!(
    /// White-relative game outcome.
    Wdl { WhiteWin = 0, Draw = 1, BlackWin = 2 }
);

obs_enum!(
    /// Node classification at capture time.
    NodeType { Root = 0, Pv = 1, Cut = 2, All = 3, Qsearch = 4 }
);

obs_enum!(
    /// Score bound classification.
    Bound { Exact = 0, Upper = 1, Lower = 2 }
);

obs_enum!(
    /// Why a move was skipped (None ⇒ searched).
    Prune {
        None = 0,
        Lmp = 1,
        Futility = 2,
        See = 3,
        History = 4,
        Razoring = 5,
        Other = 15,
    }
);

obs_enum!(
    /// Which knob differs across a counterfactual pair (D43).
    CfAxis { Reduction = 0, Extension = 1, PolicyTier = 2 }
);

obs_enum!(
    /// Observational vs forced-intervention vs reference evidence (D43).
    EvidenceKind { Observational = 0, Intervention = 1, Reference = 2 }
);

obs_enum!(
    /// Policy-inference tier in force for the move.
    PolicyTier { None = 0, Cheap = 1, Full = 2 }
);

obs_enum!(
    /// Label completion status.
    Completion { Complete = 0, Stopped = 1, TargetMiss = 2, Invalid = 3 }
);

obs_enum!(
    /// TT state regime the observation ran under (D43).
    TtRegime { Empty = 0, Restored = 1, WarmObserved = 2 }
);

obs_enum!(
    /// History-table regime the observation ran under (D43).
    HistoryRegime { Fresh = 0, Frozen = 1, Restored = 2, WarmObserved = 3 }
);

// ─── Provenance (§3.4 — mandatory; no record without it) ───────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Provenance {
    pub engine_version: String,           // 1
    pub engine_bench_sig: u64,            // 2  D24 Bench: nodes
    pub net_version: Option<String>,      // 3
    pub teacher_identity: Option<String>, // 4  engine rev + binary sha256
    pub search_config: String,            // 5  full knob set, v1: canonical text
    pub search_config_hash: u64,          // 6
    pub generator_version: String,        // 7
    pub rng_seed: u64,                    // 8
    pub dataset_source: String,           // 9
    pub opening_source: String,           // 10
    pub hardware_identity: String,        // 11 (2026-07-17, doc 19 §5.2)
}

impl Provenance {
    fn encode(&self, out: &mut FieldWriter) {
        out.string(1, &self.engine_version);
        out.fixed64(2, self.engine_bench_sig);
        if let Some(net) = &self.net_version {
            out.string(3, net);
        }
        if let Some(teacher) = &self.teacher_identity {
            out.string(4, teacher);
        }
        out.string(5, &self.search_config);
        out.fixed64(6, self.search_config_hash);
        out.string(7, &self.generator_version);
        out.fixed64(8, self.rng_seed);
        out.string(9, &self.dataset_source);
        out.string(10, &self.opening_source);
        out.string(11, &self.hardware_identity);
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut p = Provenance::default();
        let (mut saw_engine, mut saw_generator) = (false, false);
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => {
                    p.engine_version = value.as_string()?;
                    saw_engine = true;
                }
                2 => p.engine_bench_sig = value.as_fixed64()?,
                3 => p.net_version = Some(value.as_string()?),
                4 => p.teacher_identity = Some(value.as_string()?),
                5 => p.search_config = value.as_string()?,
                6 => p.search_config_hash = value.as_fixed64()?,
                7 => {
                    p.generator_version = value.as_string()?;
                    saw_generator = true;
                }
                8 => p.rng_seed = value.as_fixed64()?,
                9 => p.dataset_source = value.as_string()?,
                10 => p.opening_source = value.as_string()?,
                11 => p.hardware_identity = value.as_string()?,
                _ => {} // unknown tag: skipped by construction
            }
        }
        if !saw_engine {
            return Err(missing("Provenance.engine_version"));
        }
        if !saw_generator {
            return Err(missing("Provenance.generator_version"));
        }
        // The full §3.4 mandatory set: a record without provenance is
        // unwritable, and "provenance" means the WHOLE block (D33 round 1).
        // rng_seed/engine_bench_sig legitimately carry 0 (external-engine
        // sentinel), so only the string fields are enforced.
        if p.search_config.is_empty() {
            return Err(missing("Provenance.search_config"));
        }
        if p.dataset_source.is_empty() {
            return Err(missing("Provenance.dataset_source"));
        }
        if p.opening_source.is_empty() {
            return Err(missing("Provenance.opening_source"));
        }
        if p.hardware_identity.is_empty() {
            return Err(missing("Provenance.hardware_identity"));
        }
        Ok(p)
    }
}

// ─── Layer 1: PositionState ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PositionState {
    pub occ: u64,                         // 1
    pub pcs: Vec<u8>,                     // 2  nibble-per-occupied-square
    pub stm: u8,                          // 3
    pub castling: u8,                     // 4
    pub ep_square: Option<u8>,            // 5
    pub rule50: u8,                       // 6
    pub game_ply: u16,                    // 7
    pub plies_from_null: u16,             // 8
    pub repetition: i8,                   // 9
    pub canonical_key: u64,               // 10
    pub path_from_episode_root: Vec<u16>, // 11
}

impl PositionState {
    fn encode(&self, out: &mut FieldWriter) {
        out.fixed64(1, self.occ);
        out.bytes(2, &self.pcs);
        out.uint(3, u64::from(self.stm));
        out.uint(4, u64::from(self.castling));
        if let Some(ep) = self.ep_square {
            out.uint(5, u64::from(ep));
        }
        out.uint(6, u64::from(self.rule50));
        out.uint(7, u64::from(self.game_ply));
        out.uint(8, u64::from(self.plies_from_null));
        out.sint(9, i64::from(self.repetition));
        out.fixed64(10, self.canonical_key);
        out.packed_uints(
            11,
            self.path_from_episode_root.iter().map(|&m| u64::from(m)),
        );
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut p = PositionState::default();
        let mut saw_occ = false;
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => {
                    p.occ = value.as_fixed64()?;
                    saw_occ = true;
                }
                2 => p.pcs = value.as_bytes()?.to_vec(),
                3 => p.stm = value.as_uint()? as u8,
                4 => p.castling = value.as_uint()? as u8,
                5 => p.ep_square = Some(value.as_uint()? as u8),
                6 => p.rule50 = value.as_uint()? as u8,
                7 => p.game_ply = value.as_uint()? as u16,
                8 => p.plies_from_null = value.as_uint()? as u16,
                9 => p.repetition = value.as_sint()? as i8,
                10 => p.canonical_key = value.as_fixed64()?,
                11 => {
                    p.path_from_episode_root = value
                        .as_packed_uints()?
                        .into_iter()
                        .map(|v| v as u16)
                        .collect();
                }
                _ => {}
            }
        }
        if !saw_occ {
            return Err(missing("PositionState.occ"));
        }
        Ok(p)
    }
}

// ─── Layer 2: SearchContext ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SearchContext {
    pub depth: i32,                    // 1
    pub node_budget: Option<u64>,      // 2
    pub alpha: i32,                    // 3
    pub beta: i32,                     // 4
    pub node_type: NodeType,           // 5
    pub static_eval: i32,              // 6  raw eval at the node
    pub tt_hit: bool,                  // 7
    pub tt_depth: i32,                 // 8
    pub tt_bound: Bound,               // 9
    pub tt_value: i32,                 // 10
    pub tt_move: u16,                  // 11
    pub flags: u32,                    // 12
    pub event_ordinal: u64,            // 13
    pub event_fingerprint: u64,        // 14
    pub ply: u16,                      // 15
    pub alpha_before_move: i32,        // 16
    pub beta_before_move: i32,         // 17
    pub root_delta: i32,               // 18
    pub tt_regime: TtRegime,           // 19
    pub history_regime: HistoryRegime, // 20
    pub legal_move_count: u8,          // 21 (2026-07-17)
    pub capture_move_count: u8,        // 22 (2026-07-17)
    pub correction_value: i32,         // 23 (2026-07-17, S1)
}

impl SearchContext {
    fn encode(&self, out: &mut FieldWriter) {
        out.sint(1, i64::from(self.depth));
        if let Some(budget) = self.node_budget {
            out.uint(2, budget);
        }
        out.sint(3, i64::from(self.alpha));
        out.sint(4, i64::from(self.beta));
        out.uint(5, u64::from(self.node_type.to_u8()));
        out.sint(6, i64::from(self.static_eval));
        out.boolean(7, self.tt_hit);
        out.sint(8, i64::from(self.tt_depth));
        out.uint(9, u64::from(self.tt_bound.to_u8()));
        out.sint(10, i64::from(self.tt_value));
        out.uint(11, u64::from(self.tt_move));
        out.uint(12, u64::from(self.flags));
        out.uint(13, self.event_ordinal);
        out.fixed64(14, self.event_fingerprint);
        out.uint(15, u64::from(self.ply));
        out.sint(16, i64::from(self.alpha_before_move));
        out.sint(17, i64::from(self.beta_before_move));
        out.sint(18, i64::from(self.root_delta));
        out.uint(19, u64::from(self.tt_regime.to_u8()));
        out.uint(20, u64::from(self.history_regime.to_u8()));
        out.uint(21, u64::from(self.legal_move_count));
        out.uint(22, u64::from(self.capture_move_count));
        out.sint(23, i64::from(self.correction_value));
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut c = SearchContext::default();
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => c.depth = value.as_sint()? as i32,
                2 => c.node_budget = Some(value.as_uint()?),
                3 => c.alpha = value.as_sint()? as i32,
                4 => c.beta = value.as_sint()? as i32,
                5 => c.node_type = NodeType::from_u8(value.as_uint()? as u8),
                6 => c.static_eval = value.as_sint()? as i32,
                7 => c.tt_hit = value.as_bool()?,
                8 => c.tt_depth = value.as_sint()? as i32,
                9 => c.tt_bound = Bound::from_u8(value.as_uint()? as u8),
                10 => c.tt_value = value.as_sint()? as i32,
                11 => c.tt_move = value.as_uint()? as u16,
                12 => c.flags = value.as_uint()? as u32,
                13 => c.event_ordinal = value.as_uint()?,
                14 => c.event_fingerprint = value.as_fixed64()?,
                15 => c.ply = value.as_uint()? as u16,
                16 => c.alpha_before_move = value.as_sint()? as i32,
                17 => c.beta_before_move = value.as_sint()? as i32,
                18 => c.root_delta = value.as_sint()? as i32,
                19 => c.tt_regime = TtRegime::from_u8(value.as_uint()? as u8),
                20 => c.history_regime = HistoryRegime::from_u8(value.as_uint()? as u8),
                21 => c.legal_move_count = value.as_uint()? as u8,
                22 => c.capture_move_count = value.as_uint()? as u8,
                23 => c.correction_value = value.as_sint()? as i32,
                _ => {}
            }
        }
        Ok(c)
    }
}

// ─── Layer 3: SearchObservations ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IdSnapshot {
    pub depth: i32,   // 1
    pub score: i32,   // 2
    pub best: u16,    // 3
    pub nodes: u64,   // 4
    pub time_ms: u64, // 5
    pub pv: Vec<u16>, // 6
}

impl IdSnapshot {
    fn encode(&self, out: &mut FieldWriter) {
        out.sint(1, i64::from(self.depth));
        out.sint(2, i64::from(self.score));
        out.uint(3, u64::from(self.best));
        out.uint(4, self.nodes);
        out.uint(5, self.time_ms);
        out.packed_uints(6, self.pv.iter().map(|&m| u64::from(m)));
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut s = IdSnapshot::default();
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => s.depth = value.as_sint()? as i32,
                2 => s.score = value.as_sint()? as i32,
                3 => s.best = value.as_uint()? as u16,
                4 => s.nodes = value.as_uint()?,
                5 => s.time_ms = value.as_uint()?,
                6 => {
                    s.pv = value
                        .as_packed_uints()?
                        .into_iter()
                        .map(|v| v as u16)
                        .collect()
                }
                _ => {}
            }
        }
        Ok(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SearchObservations {
    pub best_move: u16,                // 1
    pub pv: Vec<u16>,                  // 2
    pub final_score: i32,              // 3
    pub bound: Bound,                  // 4
    pub nodes: u64,                    // 5
    pub seldepth: i32,                 // 6
    pub id_snapshots: Vec<IdSnapshot>, // 7
    /// 8 — appended 2026-07-17 (D33 round 1): a timed-out/partial REFERENCE
    /// search must never read as a complete Exact label (§5.4/§11.1 demand
    /// completion status on every label, not only per-move rows).
    pub completion: Completion, // 8
}

impl SearchObservations {
    fn encode(&self, out: &mut FieldWriter) {
        out.uint(1, u64::from(self.best_move));
        out.packed_uints(2, self.pv.iter().map(|&m| u64::from(m)));
        out.sint(3, i64::from(self.final_score));
        out.uint(4, u64::from(self.bound.to_u8()));
        out.uint(5, self.nodes);
        out.sint(6, i64::from(self.seldepth));
        for snapshot in &self.id_snapshots {
            let mut nested = FieldWriter::new();
            snapshot.encode(&mut nested);
            out.bytes(7, nested.as_bytes());
        }
        out.uint(8, u64::from(self.completion.to_u8()));
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut s = SearchObservations::default();
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => s.best_move = value.as_uint()? as u16,
                2 => {
                    s.pv = value
                        .as_packed_uints()?
                        .into_iter()
                        .map(|v| v as u16)
                        .collect()
                }
                3 => s.final_score = value.as_sint()? as i32,
                4 => s.bound = Bound::from_u8(value.as_uint()? as u8),
                5 => s.nodes = value.as_uint()?,
                6 => s.seldepth = value.as_sint()? as i32,
                7 => s.id_snapshots.push(IdSnapshot::decode(value.as_bytes()?)?),
                8 => s.completion = Completion::from_u8(value.as_uint()? as u8),
                _ => {}
            }
        }
        Ok(s)
    }
}

// ─── Layer 4: PerMoveObservation ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PerMoveObservation {
    pub mv: u16,                              // 1
    pub order_index: u16,                     // 2
    pub order_stage: u8,                      // 3
    pub order_score: i32,                     // 4
    pub searched_depth: i32,                  // 5
    pub reduction: i16,                       // 6
    pub extension: i16,                       // 7
    pub returned_score: i32,                  // 8
    pub nodes: u64,                           // 9
    pub raised_alpha: bool,                   // 10
    pub caused_cutoff: bool,                  // 11
    pub researched: bool,                     // 12
    pub pruned_by: Prune,                     // 13
    pub evidence: EvidenceKind,               // 14
    pub baseline_reduction: i16,              // 15
    pub requested_reduction: i16,             // 16
    pub actual_scout_depth: i32,              // 17
    pub policy_tier: PolicyTier,              // 18
    pub scout_score: i32,                     // 19
    pub scout_bound: Bound,                   // 20
    pub scout_nodes: u64,                     // 21
    pub research_score: Option<i32>,          // 22
    pub research_bound: Option<Bound>,        // 23
    pub research_nodes: u64,                  // 24
    pub research_changed_bound: bool,         // 25
    pub research_was_necessary: Option<bool>, // 26
    pub reference_bound: Option<Bound>,       // 27
    pub bound_error: Option<bool>,            // 28
    pub completion: Completion,               // 29
}

impl PerMoveObservation {
    fn encode(&self, out: &mut FieldWriter) {
        out.uint(1, u64::from(self.mv));
        out.uint(2, u64::from(self.order_index));
        out.uint(3, u64::from(self.order_stage));
        out.sint(4, i64::from(self.order_score));
        out.sint(5, i64::from(self.searched_depth));
        out.sint(6, i64::from(self.reduction));
        out.sint(7, i64::from(self.extension));
        out.sint(8, i64::from(self.returned_score));
        out.uint(9, self.nodes);
        out.boolean(10, self.raised_alpha);
        out.boolean(11, self.caused_cutoff);
        out.boolean(12, self.researched);
        out.uint(13, u64::from(self.pruned_by.to_u8()));
        out.uint(14, u64::from(self.evidence.to_u8()));
        out.sint(15, i64::from(self.baseline_reduction));
        out.sint(16, i64::from(self.requested_reduction));
        out.sint(17, i64::from(self.actual_scout_depth));
        out.uint(18, u64::from(self.policy_tier.to_u8()));
        out.sint(19, i64::from(self.scout_score));
        out.uint(20, u64::from(self.scout_bound.to_u8()));
        out.uint(21, self.scout_nodes);
        if let Some(score) = self.research_score {
            out.sint(22, i64::from(score));
        }
        if let Some(bound) = self.research_bound {
            out.uint(23, u64::from(bound.to_u8()));
        }
        out.uint(24, self.research_nodes);
        out.boolean(25, self.research_changed_bound);
        if let Some(necessary) = self.research_was_necessary {
            out.boolean(26, necessary);
        }
        if let Some(bound) = self.reference_bound {
            out.uint(27, u64::from(bound.to_u8()));
        }
        if let Some(error) = self.bound_error {
            out.boolean(28, error);
        }
        out.uint(29, u64::from(self.completion.to_u8()));
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut m = PerMoveObservation::default();
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => m.mv = value.as_uint()? as u16,
                2 => m.order_index = value.as_uint()? as u16,
                3 => m.order_stage = value.as_uint()? as u8,
                4 => m.order_score = value.as_sint()? as i32,
                5 => m.searched_depth = value.as_sint()? as i32,
                6 => m.reduction = value.as_sint()? as i16,
                7 => m.extension = value.as_sint()? as i16,
                8 => m.returned_score = value.as_sint()? as i32,
                9 => m.nodes = value.as_uint()?,
                10 => m.raised_alpha = value.as_bool()?,
                11 => m.caused_cutoff = value.as_bool()?,
                12 => m.researched = value.as_bool()?,
                13 => m.pruned_by = Prune::from_u8(value.as_uint()? as u8),
                14 => m.evidence = EvidenceKind::from_u8(value.as_uint()? as u8),
                15 => m.baseline_reduction = value.as_sint()? as i16,
                16 => m.requested_reduction = value.as_sint()? as i16,
                17 => m.actual_scout_depth = value.as_sint()? as i32,
                18 => m.policy_tier = PolicyTier::from_u8(value.as_uint()? as u8),
                19 => m.scout_score = value.as_sint()? as i32,
                20 => m.scout_bound = Bound::from_u8(value.as_uint()? as u8),
                21 => m.scout_nodes = value.as_uint()?,
                22 => m.research_score = Some(value.as_sint()? as i32),
                23 => m.research_bound = Some(Bound::from_u8(value.as_uint()? as u8)),
                24 => m.research_nodes = value.as_uint()?,
                25 => m.research_changed_bound = value.as_bool()?,
                26 => m.research_was_necessary = Some(value.as_bool()?),
                27 => m.reference_bound = Some(Bound::from_u8(value.as_uint()? as u8)),
                28 => m.bound_error = Some(value.as_bool()?),
                29 => m.completion = Completion::from_u8(value.as_uint()? as u8),
                _ => {}
            }
        }
        Ok(m)
    }
}

// ─── Observation (the four layers) ─────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Observation {
    pub position: PositionState,        // 1
    pub context: SearchContext,         // 2
    pub search: SearchObservations,     // 3
    pub moves: Vec<PerMoveObservation>, // 4
}

impl Observation {
    fn encode(&self, out: &mut FieldWriter) {
        let mut nested = FieldWriter::new();
        self.position.encode(&mut nested);
        out.bytes(1, nested.as_bytes());

        let mut nested = FieldWriter::new();
        self.context.encode(&mut nested);
        out.bytes(2, nested.as_bytes());

        let mut nested = FieldWriter::new();
        self.search.encode(&mut nested);
        out.bytes(3, nested.as_bytes());

        for mv in &self.moves {
            let mut nested = FieldWriter::new();
            mv.encode(&mut nested);
            out.bytes(4, nested.as_bytes());
        }
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut o = Observation::default();
        let mut saw_position = false;
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => {
                    o.position = PositionState::decode(value.as_bytes()?)?;
                    saw_position = true;
                }
                2 => o.context = SearchContext::decode(value.as_bytes()?)?,
                3 => o.search = SearchObservations::decode(value.as_bytes()?)?,
                4 => o.moves.push(PerMoveObservation::decode(value.as_bytes()?)?),
                _ => {}
            }
        }
        if !saw_position {
            return Err(missing("Observation.position"));
        }
        Ok(o)
    }
}

// ─── Episode ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Episode {
    pub kind: EpisodeKind,              // 1
    pub episode_id: u64,                // 2
    pub provenance_ref: u32,            // 3  index into ShardHeader.provenance
    pub game_result: Option<Wdl>,       // 4
    pub cf_axis: Option<CfAxis>,        // 5
    pub records: Vec<Observation>,      // 6
    pub replay_root: PositionState,     // 7
    pub root_reversible_keys: Vec<u64>, // 8
    pub event_hash_chain: Vec<u64>,     // 9
    /// 10 — appended 2026-07-17 (design/16 §3.5.5, D45a): sampled records the
    /// per-game `ObsCapturePerGameCap` bound refused (no silent loss). Encoded
    /// only when non-zero, so every pre-existing episode is byte-unchanged.
    pub dropped_records: u64, // 10
}

impl Episode {
    pub(super) fn encode(&self, out: &mut FieldWriter) {
        out.uint(1, u64::from(self.kind.to_u8()));
        out.uint(2, self.episode_id);
        out.uint(3, u64::from(self.provenance_ref));
        if let Some(result) = self.game_result {
            out.uint(4, u64::from(result.to_u8()));
        }
        if let Some(axis) = self.cf_axis {
            out.uint(5, u64::from(axis.to_u8()));
        }
        for record in &self.records {
            let mut nested = FieldWriter::new();
            record.encode(&mut nested);
            out.bytes(6, nested.as_bytes());
        }
        let mut nested = FieldWriter::new();
        self.replay_root.encode(&mut nested);
        out.bytes(7, nested.as_bytes());
        out.packed_uints(8, self.root_reversible_keys.iter().copied());
        out.packed_uints(9, self.event_hash_chain.iter().copied());
        if self.dropped_records != 0 {
            out.uint(10, self.dropped_records);
        }
    }

    pub(super) fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut e = Episode::default();
        let mut saw_provenance = false;
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                1 => e.kind = EpisodeKind::from_u8(value.as_uint()? as u8),
                2 => e.episode_id = value.as_uint()?,
                3 => {
                    e.provenance_ref = value.as_uint()? as u32;
                    saw_provenance = true;
                }
                4 => e.game_result = Some(Wdl::from_u8(value.as_uint()? as u8)),
                5 => e.cf_axis = Some(CfAxis::from_u8(value.as_uint()? as u8)),
                6 => e.records.push(Observation::decode(value.as_bytes()?)?),
                7 => e.replay_root = PositionState::decode(value.as_bytes()?)?,
                8 => e.root_reversible_keys = value.as_packed_uints()?,
                9 => e.event_hash_chain = value.as_packed_uints()?,
                10 => e.dropped_records = value.as_uint()?,
                _ => {}
            }
        }
        if !saw_provenance {
            return Err(missing("Episode.provenance_ref"));
        }
        Ok(e)
    }
}

// ─── ShardHeader ───────────────────────────────────────────────────────────

pub const OBS_MAGIC: [u8; 6] = *b"CEOBS\0";
pub const OBS_FORMAT_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShardHeader {
    pub format_version: u16,         // 2 (magic is file framing, not a field)
    pub created_unix: u64,           // 3
    pub provenance: Vec<Provenance>, // 4
}

impl ShardHeader {
    pub(super) fn encode(&self, out: &mut FieldWriter) {
        out.uint(2, u64::from(self.format_version));
        out.uint(3, self.created_unix);
        for provenance in &self.provenance {
            let mut nested = FieldWriter::new();
            provenance.encode(&mut nested);
            out.bytes(4, nested.as_bytes());
        }
    }

    pub(super) fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut h = ShardHeader::default();
        let mut saw_version = false;
        let mut reader = FieldReader::new(bytes);
        while let Some((tag, value)) = reader.next()? {
            match tag {
                2 => {
                    h.format_version = value.as_uint()? as u16;
                    saw_version = true;
                }
                3 => h.created_unix = value.as_uint()?,
                4 => h.provenance.push(Provenance::decode(value.as_bytes()?)?),
                _ => {}
            }
        }
        if !saw_version {
            return Err(missing("ShardHeader.format_version"));
        }
        if h.provenance.is_empty() {
            return Err(missing(
                "ShardHeader.provenance (a record without provenance is unwritable)",
            ));
        }
        Ok(h)
    }
}
