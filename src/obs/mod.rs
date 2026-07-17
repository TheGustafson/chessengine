//! The Observation Store (design/16 §3.5, D29/D43): the permanent,
//! forward-compatible record of what our searches saw and did — the four
//! vision layers (position / search context / outcomes / per-move), episode
//! framing, and mandatory provenance.
//!
//! Behind `feature = "datagen"`: the tournament/bench binary compiles this
//! out entirely, so the store can never perturb normal search (§6 bench-
//! neutrality). std-only by design (D15) — the TLV wire layer is hand-rolled
//! after the protobuf wire *model* (a public fact; no protobuf code or
//! dependency, D27).

pub mod schema;
pub mod shard;
mod wire;

pub use schema::{
    Bound, CfAxis, Completion, Episode, EpisodeKind, EvidenceKind, HistoryRegime, IdSnapshot,
    NodeType, OBS_FORMAT_VERSION, OBS_MAGIC, Observation, PerMoveObservation, PolicyTier,
    PositionState, Provenance, Prune, SearchContext, SearchObservations, ShardHeader, TtRegime,
    Wdl,
};
pub use shard::{ShardReader, ShardWriter};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::wire::FieldWriter;

    fn full_provenance() -> Provenance {
        Provenance {
            engine_version: "archon 0.1.0 @d895cf6".into(),
            engine_bench_sig: 701_605,
            net_version: Some("stage2-v1-s3-2048-farseer-120sb".into()),
            teacher_identity: Some("reckless v7 sha256:ab12…".into()),
            search_config: "Threads=1 Hash=16 nodes=5000".into(),
            search_config_hash: 0x1234_5678_9abc_def0,
            generator_version: "root-analysis v0".into(),
            rng_seed: 20_260_717,
            dataset_source: "root_move_analysis".into(),
            opening_source: "UHO_Lichess_4852_v1.epd".into(),
            hardware_identity: "AMD Ryzen 7 7700X, avx512vnni, 1 thread".into(),
        }
    }

    fn full_observation() -> Observation {
        Observation {
            position: PositionState {
                occ: 0xFFFF_0000_0000_FFFF,
                pcs: vec![0x12, 0x34, 0x56],
                stm: 1,
                castling: 0b1011,
                ep_square: Some(20),
                rule50: 3,
                game_ply: 24,
                plies_from_null: 7,
                repetition: -1,
                canonical_key: 0xDEAD_BEEF_0000_1111,
                path_from_episode_root: vec![0x0104, 0x0f3c],
            },
            context: SearchContext {
                depth: 9,
                node_budget: Some(5_000),
                alpha: -55,
                beta: 40,
                node_type: NodeType::Cut,
                static_eval: -12,
                tt_hit: true,
                tt_depth: 7,
                tt_bound: Bound::Lower,
                tt_value: 18,
                tt_move: 0x0104,
                flags: 0b1010,
                event_ordinal: 421,
                event_fingerprint: 0x0BAD_F00D_0000_0001,
                ply: 6,
                alpha_before_move: -55,
                beta_before_move: 40,
                root_delta: 11,
                tt_regime: TtRegime::WarmObserved,
                history_regime: HistoryRegime::Frozen,
                legal_move_count: 31,
                capture_move_count: 4,
                correction_value: -194_200,
            },
            search: SearchObservations {
                best_move: 0x0104,
                pv: vec![0x0104, 0x0f3c, 0x0777],
                final_score: 23,
                bound: Bound::Exact,
                nodes: 4_812,
                seldepth: 14,
                id_snapshots: vec![IdSnapshot {
                    depth: 8,
                    score: 19,
                    best: 0x0104,
                    nodes: 2_400,
                    time_ms: 12,
                    pv: vec![0x0104],
                }],
            },
            moves: vec![PerMoveObservation {
                mv: 0x0f3c,
                order_index: 3,
                order_stage: 2,
                order_score: -812,
                searched_depth: 7,
                reduction: 2,
                extension: 0,
                returned_score: -31,
                nodes: 388,
                raised_alpha: false,
                caused_cutoff: false,
                researched: true,
                pruned_by: Prune::None,
                evidence: EvidenceKind::Intervention,
                baseline_reduction: 1,
                requested_reduction: 2,
                actual_scout_depth: 6,
                policy_tier: PolicyTier::Cheap,
                scout_score: -40,
                scout_bound: Bound::Upper,
                scout_nodes: 121,
                research_score: Some(-31),
                research_bound: Some(Bound::Exact),
                research_nodes: 267,
                research_changed_bound: true,
                research_was_necessary: Some(true),
                reference_bound: Some(Bound::Exact),
                bound_error: Some(false),
                completion: Completion::Complete,
            }],
        }
    }

    fn full_episode() -> Episode {
        Episode {
            kind: EpisodeKind::RootMoveAnalysis,
            episode_id: 99,
            provenance_ref: 0,
            game_result: Some(Wdl::Draw),
            cf_axis: Some(CfAxis::Reduction),
            records: vec![full_observation(), full_observation()],
            replay_root: full_observation().position,
            root_reversible_keys: vec![1, 2, 3],
            event_hash_chain: vec![0xAAAA, 0xBBBB],
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("archon-obs-{}-{}.ceobs", name, std::process::id()))
    }

    /// §5 round-trip: every field of all four layers survives the store's own
    /// write→read path (not a reimplementation).
    #[test]
    fn shard_round_trips_all_four_layers() {
        let path = temp_path("roundtrip");
        let header = ShardHeader {
            format_version: OBS_FORMAT_VERSION,
            created_unix: 1_784_000_000,
            provenance: vec![full_provenance()],
        };
        let episode = full_episode();

        let mut writer = ShardWriter::create(&path, &header).unwrap();
        writer.append(&episode).unwrap();
        writer.append(&episode).unwrap();
        writer.finish().unwrap();

        let mut reader = ShardReader::open(&path).unwrap();
        assert_eq!(reader.header(), &header);
        assert_eq!(reader.next_episode().unwrap().unwrap(), episode);
        assert_eq!(reader.next_episode().unwrap().unwrap(), episode);
        assert!(reader.next_episode().unwrap().is_none());
        std::fs::remove_file(&path).ok();
    }

    /// §5 forward-compat: a shard carrying (a) a field tag this reader has
    /// never heard of and (b) an unknown EpisodeKind discriminant must decode
    /// cleanly, recovering every known field and preserving the unknown kind.
    #[test]
    fn reader_survives_future_fields_and_kinds() {
        // Hand-assemble a future episode: known fields + tag 900 (varint),
        // tag 901 (string) + an EpisodeKind from the future (200).
        let mut message = FieldWriter::new();
        message.uint(1, 200); // future kind
        message.uint(2, 7); // episode_id
        message.uint(3, 0); // provenance_ref
        message.uint(900, 42); // future varint field
        message.string(901, "from the future"); // future length-delimited field
        let mut replay_root = FieldWriter::new();
        replay_root.fixed64(1, 0xFF); // occ (mandatory)
        message.bytes(7, replay_root.as_bytes());

        let episode = Episode::decode(message.as_bytes()).unwrap();
        assert_eq!(episode.kind, EpisodeKind::Unknown(200));
        assert_eq!(episode.episode_id, 7);
        assert_eq!(episode.replay_root.occ, 0xFF);
        // Unknown kind survives a re-encode (discriminant preserved).
        assert_eq!(episode.kind.to_u8(), 200);
    }

    /// §3.4: a record without provenance is unwritable — enforced on BOTH
    /// sides: the writer refuses, and a headerless/provenance-free shard
    /// fails to decode.
    #[test]
    fn provenance_is_mandatory_on_both_sides() {
        let path = temp_path("noprov");
        let header = ShardHeader {
            format_version: OBS_FORMAT_VERSION,
            created_unix: 0,
            provenance: vec![],
        };
        assert!(ShardWriter::create(&path, &header).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// A shard claiming a NEWER format_version than this reader must be
    /// refused (additive changes never bump the version, so newer = breaking).
    #[test]
    fn newer_format_version_is_refused() {
        let path = temp_path("newver");
        let header = ShardHeader {
            format_version: OBS_FORMAT_VERSION + 1,
            created_unix: 0,
            provenance: vec![full_provenance()],
        };
        let writer = ShardWriter::create(&path, &header).unwrap();
        writer.finish().unwrap();
        assert!(ShardReader::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// Crash-truncation surfaces as an error on the partial episode, with
    /// every earlier episode still readable.
    #[test]
    fn truncated_trailing_episode_errors_after_good_ones() {
        let path = temp_path("trunc");
        let header = ShardHeader {
            format_version: OBS_FORMAT_VERSION,
            created_unix: 0,
            provenance: vec![full_provenance()],
        };
        let episode = full_episode();
        let mut writer = ShardWriter::create(&path, &header).unwrap();
        writer.append(&episode).unwrap();
        writer.append(&episode).unwrap();
        writer.finish().unwrap();

        let len = std::fs::metadata(&path).unwrap().len();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..(len as usize - 10)]).unwrap();

        let mut reader = ShardReader::open(&path).unwrap();
        assert_eq!(reader.next_episode().unwrap().unwrap(), episode);
        assert!(
            reader.next_episode().is_err(),
            "partial episode must error, not vanish"
        );
        std::fs::remove_file(&path).ok();
    }
}
