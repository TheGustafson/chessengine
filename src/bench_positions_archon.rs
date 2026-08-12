//! The ACTIVE TEMPO-original bench suite (2026-08 originality audit, item
//! 6b; `docs/audit/2026-08-originality/00-synthesis.md` §2). Re-exported as
//! [`crate::bench::BENCH_POSITIONS`], so this list IS the `Bench:` trailer's
//! input.
//!
//! It replaced a 49/49 byte-identical subsequence of Stockfish's
//! `benchmark.cpp` `Defaults` (foundation-infra.md F2), and is drawn
//! entirely from TEMPO's OWN played games:
//! 47 positions sampled from the sfdev5k-a campaign PGNs (146 files, first
//! 3000 games) by `tools/bench/derive_bench_suite.py` — deterministic
//! phase-stratified selection (10 opening / 18 middlegame / 14 endgame /
//! 5 late-endgame, side-to-move alternation, one position per source game,
//! candidates ordered by sha256(fen)) — plus the first two diagnosed-loss
//! positions of `searchopt/positions/hard.json`.
//!
//! Activating it moved the depth-6 `Bench:` pin by design; the NNUE
//! gate's eval hash and d13 total are computed on `tools/nnue/calib/wl50.fens`
//! and did NOT move.

pub const BENCH_POSITIONS_ARCHON: &[&str] = &[
    "2rqk2r/pQ2bppp/2p2n2/3p4/3P2b1/P1NnP1P1/1P2NPBP/R1B2K1R w k - 1 13", // opening
    "r2qk2r/1b1nbpp1/pp1pp1np/8/P2PP2P/3B1NP1/1P1NQP2/R1B1R1K1 b kq - 0 14", // opening
    "rn1qk2r/pb2ppbp/6p1/1Npp4/3Pn2B/3BPN2/PPP2PPP/R2QK2R w KQkq - 0 9",  // opening
    "rnbqkb1r/2p1nppp/p2pp3/4P2P/Pp1P4/5N2/1PPN1PP1/R1BQKB1R b KQkq - 2 9", // opening
    "1rb2r1k/2p3qp/1p1p1npb/p2Ppp1n/BPP1P3/P1N2P2/3N1BPP/3RRQK1 w - - 1 23", // opening
    "rnb1kb1r/pppp1ppp/4pn2/8/2NP1B2/4PPP1/qR5P/3QKBNR b Kkq - 3 10",     // opening
    "2kr1bnr/ppp3pp/5p2/2q1pb2/Q2n4/2NPBN2/PPP2PPP/R3KB1R w KQ - 0 10",   // opening
    "3r1rk1/pp2qppp/2nbbn2/3pp3/1PP4N/P2P4/1B1NBPPP/2RQ1RK1 b - - 0 14",  // opening
    "r2q1rk1/1b1n1ppp/2n1p3/pppp2PP/4P3/3P1N2/PPPBQP2/2KR1B1R w - - 0 15", // opening
    "r2qr1k1/1b1n1p1p/2pp1npQ/pp2pP2/P3P1P1/2NP2NP/1PP3B1/R4RK1 b - - 0 17", // opening
    "1n1qkb1r/1b3pp1/2pppn1p/r7/2BP1B2/N3PN2/1P3PPP/R2QK2R w KQk - 0 12", // middlegame
    "3nr2k/p2q2r1/P5p1/2p1p2p/2Q4P/2P3P1/4R3/5RK1 b - - 0 35",            // middlegame
    "1kn4r/1pp3r1/p7/4p1p1/P1PpP3/3P2P1/R1P3B1/5RK1 w - - 2 34",          // middlegame
    "rnb1k1nr/ppq3pp/4p3/3pp3/B2P4/6Q1/PP1N1PPP/R3K1NR b KQkq - 1 12",    // middlegame
    "r6r/1p1bkp2/pq2pp1p/3n3P/4N3/P7/1PP1QPP1/2KR1B1R w - - 1 19",        // middlegame
    "3r4/3q1pk1/1p3np1/1N1nR2p/P2P3P/1Q3BP1/5PK1/8 b - - 2 38",           // middlegame
    "8/r4pk1/1r4pp/1P1qp1b1/3pN1P1/3Q1P2/1R4P1/1R4K1 w - - 5 36",         // middlegame
    "2r3k1/pp3pp1/r3b2p/8/8/P1Pp1P2/PB3KPP/R3R3 b - - 5 28",              // middlegame
    "r1b2rk1/ppp2p1p/3qpnp1/4Nn2/2B2P2/2N1Q3/PP3PPP/R2R2K1 w - - 2 16",   // middlegame
    "r1bqk2r/4bpp1/p2p3p/n2P4/5Pn1/RP2PN2/4N1BP/2BQ1RK1 b kq - 0 17",     // middlegame
    "6k1/p4pp1/1p1r2n1/3n4/2PR3p/4BP2/PP4PP/6K1 w - - 0 38",              // middlegame
    "2kr2r1/pb5q/1ppPR3/4Npp1/N1Q5/P6P/1P3PP1/6K1 b - - 2 25",            // middlegame
    "2r1k1r1/p2qbp2/Pp1p1n2/1P6/R1pPP3/2P2QPp/8/3BRNK1 w - - 1 32",       // middlegame
    "r2q2k1/p5p1/1pb5/2p1pR1p/2P5/3Pb1PP/PP1NQ2n/R1B1K3 b - - 3 21",      // middlegame
    "r3r3/p5N1/1pk3pp/6B1/3P2P1/7P/PP2bP2/4R1K1 w - - 0 25",              // middlegame
    "r3r1k1/p5p1/1p1RpBqp/5p2/2P5/P4Q1P/1P3PP1/4R1K1 b - - 0 28",         // middlegame
    "r1b3r1/pp1nkp1p/2pNp3/2bpP3/5B1P/5p2/PPP3P1/2K1RB1R w - - 0 19",     // middlegame
    "1rb2rk1/1p3pb1/p7/P1ppq3/5N1Q/3P4/1PP3PP/R4RK1 b - - 4 24",          // middlegame
    "8/5pk1/4P1p1/3R4/2P1PKB1/r7/7p/2n5 w - - 0 60",                      // endgame
    "8/8/7r/pB6/4k3/2b5/2P1KB2/8 b - - 18 65",                            // endgame
    "6k1/6b1/5q2/1p6/1Q2p3/BP6/P7/1KR5 w - - 0 48",                       // endgame
    "k2r4/1p6/pNp5/P3Q3/6Pb/1PP5/4K3/8 b - - 12 68",                      // endgame
    "8/2k5/2n3p1/p1p1p2p/2P4R/1PK2P1P/8/3r4 w - - 0 38",                  // endgame
    "2R5/3r4/1r1k1R2/8/B3p2p/7P/6PK/8 b - - 1 55",                        // endgame
    "7r/8/Qr2pkn1/2p2pb1/2P5/3P4/5R1P/4NK2 w - - 0 37",                   // endgame
    "8/P1p5/3p4/1K1P4/4r3/2P5/8/7k b - - 0 61",                           // endgame
    "2k5/8/p5p1/7p/P1Ppp1rP/3N4/1P1K2b1/6R1 w - - 0 45",                  // endgame
    "8/p1k5/8/5p2/4rp1p/P1P2RpP/3KB1q1/5RN1 b - - 0 39",                  // endgame
    "8/5ppk/5r2/NP1p3p/3p4/P2P3n/5P2/1R3K2 w - - 0 36",                   // endgame
    "8/8/p2bkp2/2P4p/p2B2pP/3K2P1/1P3P2/8 b - - 0 40",                    // endgame
    "6r1/7R/1p1pkP2/2p5/p1P5/8/P1P3P1/6K1 w - - 1 42",                    // endgame
    "8/3k4/1p4PP/p2BP3/5P2/2P5/6K1/8 b - - 0 48",                         // endgame
    "8/8/2K5/1P5k/8/P5p1/8/8 w - - 0 58",                                 // late-endgame
    "8/7Q/8/8/8/8/q7/4k2K b - - 4 74",                                    // late-endgame
    "8/8/6k1/p6p/P6K/7P/8/8 w - - 10 56",                                 // late-endgame
    "8/4k3/8/8/2P5/2KR4/5r2/8 b - - 4 55",                                // late-endgame
    "8/5B2/2k5/4b3/2p5/2PR1K2/8/8 w - - 0 67",                            // late-endgame
    "8/1r4p1/1P2k1Pp/R1B1p3/4p1P1/p3N3/3K4/r7 b - - 7 60",                // known-hard
    "3r1rk1/1b1nqp1p/pb2pp2/8/NP5R/2Q2N2/2P1BPPP/5RK1 b - - 6 19",        // known-hard
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen;
    use crate::moves::MoveList;

    /// Every staged entry parses, passes the full board invariant oracle,
    /// and has at least one legal move — so activation can never trip on a
    /// malformed or terminal position.
    #[test]
    fn staged_suite_is_valid_and_playable() {
        assert_eq!(BENCH_POSITIONS_ARCHON.len(), 49);
        let mut seen = std::collections::HashSet::new();
        for fen in BENCH_POSITIONS_ARCHON {
            let board = Board::from_fen(fen)
                .unwrap_or_else(|e| panic!("staged bench FEN '{fen}' invalid: {e}"));
            board.assert_consistent();
            let mut list = MoveList::new();
            movegen::generate_legal(&board, &mut list);
            assert!(
                !list.is_empty(),
                "staged bench FEN '{fen}' has no legal moves"
            );
            assert!(seen.insert(*fen), "duplicate staged bench FEN '{fen}'");
        }
    }
}
