//! `Board` ↔ [`PositionState`] conversion — shared by generators (encode)
//! and the verifier (decode + recheck). The nibble packing is doc 15
//! §3.2.1's bulletformat convention: one 4-bit nibble per occupied square in
//! ascending square order (LSB→MSB of `occ`), value `(colour << 3) |
//! piece_type` with piece_type 0=P..5=K; even nibble index = low nibble.

use super::schema::PositionState;
use crate::board::{Board, CastlingRights};
use crate::fen::FenError;
use crate::types::{Color, PieceType, Square};

/// Encode `board` into Layer 1. `repetition` and the path are the caller's
/// knowledge (a `Board` set up from FEN has no history to derive them from).
pub fn position_from_board(
    board: &Board,
    repetition: i8,
    path_from_episode_root: Vec<u16>,
) -> PositionState {
    let mut occ = 0u64;
    let mut nibbles = Vec::with_capacity(32);
    // Bitboard iteration pops squares in ascending index order — exactly the
    // nibble order the format requires.
    for sq in board.all_occupancy() {
        occ |= 1u64 << sq.index();
        let piece = board.piece_on(sq).expect("occupied square");
        nibbles.push(((piece.color as u8) << 3) | piece.piece_type as u8);
    }
    let mut pcs = vec![0u8; nibbles.len().div_ceil(2)];
    for (i, nibble) in nibbles.iter().enumerate() {
        if i.is_multiple_of(2) {
            pcs[i / 2] |= nibble;
        } else {
            pcs[i / 2] |= nibble << 4;
        }
    }
    // Loud instead of wrapped: a permanent store must never silently
    // truncate identity-adjacent fields (D33 review round 1).
    assert!(
        board.halfmove_clock() <= u32::from(u8::MAX),
        "rule50 overflows u8"
    );
    assert!(
        board.game_ply() <= i32::from(u16::MAX),
        "game_ply overflows u16"
    );
    PositionState {
        occ,
        pcs,
        stm: board.side_to_move() as u8,
        castling: board.castling_rights().bits(),
        ep_square: board.en_passant_square().map(|sq| sq.index()),
        rule50: board.halfmove_clock() as u8,
        game_ply: board.game_ply() as u16,
        plies_from_null: board.plies_from_null() as u16,
        repetition,
        // The UNFOLDED key: placement+stm+castling+ep only. `board.key()`
        // folds rule50 buckets in (adjust_key50), which would give the same
        // position different identities at different clock values and break
        // the dedup / train-test leakage barrier this field exists for
        // (D33 review round 1, HIGH).
        canonical_key: board.raw_key(),
        path_from_episode_root,
    }
}

/// Replay a stored move path from `root`, requiring every step to be a
/// legal move (exact 16-bit identity — generators must store the canonical
/// `Move::raw16()`, never a reconstructed from/to encoding). Shared by
/// `obs-tool verify` and the replay tests.
pub fn replay_path(root: &Board, path: &[u16]) -> Result<Board, String> {
    use crate::movegen::generate_legal;
    use crate::moves::{Move, MoveList};
    let mut board = root.clone();
    for (ply, &raw) in path.iter().enumerate() {
        let mv = Move::from_raw16(raw);
        let mut legal = MoveList::new();
        generate_legal(&board, &mut legal);
        if !legal.contains(&mv) {
            return Err(format!("path ply {ply}: move {raw:#06x} is not legal here"));
        }
        let gives_check = board.gives_check(mv);
        board.make_move(mv, gives_check);
    }
    Ok(board)
}

/// The FEN a [`PositionState`] describes. Reconstructing THROUGH FEN reuses
/// `Board::from_fen`'s full validation (piece counts, ep legality, …) instead
/// of trusting shard bytes.
pub fn fen_from_position(position: &PositionState) -> Result<String, String> {
    // Unpack sq -> piece letter.
    let mut letters = [None::<char>; 64];
    let mut nibble_index = 0usize;
    for sq in 0..64u8 {
        if position.occ & (1u64 << sq) == 0 {
            continue;
        }
        let byte = position
            .pcs
            .get(nibble_index / 2)
            .ok_or_else(|| format!("pcs too short for occupancy (square {sq})"))?;
        let nibble = if nibble_index.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        nibble_index += 1;
        let piece_type = nibble & 0b0111;
        let black = nibble & 0b1000 != 0;
        let letter = *b"pnbrqk"
            .get(piece_type as usize)
            .ok_or_else(|| format!("invalid piece nibble {nibble} at square {sq}"))?
            as char;
        letters[sq as usize] = Some(if black {
            letter
        } else {
            letter.to_ascii_uppercase()
        });
    }

    let mut placement = String::new();
    for rank in (0..8).rev() {
        let mut empties = 0;
        for file in 0..8 {
            match letters[rank * 8 + file] {
                Some(letter) => {
                    if empties > 0 {
                        placement.push(char::from_digit(empties, 10).expect("<= 8"));
                        empties = 0;
                    }
                    placement.push(letter);
                }
                None => empties += 1,
            }
        }
        if empties > 0 {
            placement.push(char::from_digit(empties, 10).expect("<= 8"));
        }
        if rank > 0 {
            placement.push('/');
        }
    }

    let stm = match position.stm {
        0 => "w",
        1 => "b",
        other => return Err(format!("invalid stm {other}")),
    };

    let mut castling = String::new();
    for (bit, letter) in [(0b0001, 'K'), (0b0010, 'Q'), (0b0100, 'k'), (0b1000, 'q')] {
        if position.castling & bit != 0 {
            castling.push(letter);
        }
    }
    if castling.is_empty() {
        castling.push('-');
    }

    let ep = match position.ep_square {
        Some(sq) if sq < 64 => Square::new(sq).to_string(),
        Some(sq) => return Err(format!("invalid ep square {sq}")),
        None => "-".into(),
    };

    let fullmove = u32::from(position.game_ply) / 2 + 1;
    Ok(format!(
        "{placement} {stm} {castling} {ep} {} {fullmove}",
        position.rule50
    ))
}

/// Reconstruct and validate the board a [`PositionState`] describes.
pub fn board_from_position(position: &PositionState) -> Result<Board, String> {
    let fen = fen_from_position(position)?;
    Board::from_fen(&fen).map_err(|error: FenError| format!("invalid position ({fen}): {error}"))
}

// Compile-time guards: the nibble encoding leans on these discriminants.
const _: () = assert!(PieceType::Pawn as u8 == 0 && PieceType::King as u8 == 5);
const _: () = assert!(Color::White as u8 == 0 && Color::Black as u8 == 1);
const _: () = assert!(CastlingRights::WHITE_KINGSIDE.bits() == 0b0001);
const _: () = assert!(CastlingRights::BLACK_QUEENSIDE.bits() == 0b1000);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    #[test]
    fn board_round_trips_through_position_state() {
        for fen in [
            START_FEN,
            KIWIPETE,
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 3 40",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let state = position_from_board(&board, 0, vec![]);
            let back = board_from_position(&state).unwrap();
            assert_eq!(back.key(), board.key(), "{fen}");
            assert_eq!(back.to_fen(), board.to_fen(), "{fen}");
            assert_eq!(state.canonical_key, board.raw_key(), "{fen}");
        }
    }

    /// THE D33 round-1 HIGH: canonical_key must be pure position identity.
    /// Same placement/stm/castling/ep at different fifty-move clocks MUST
    /// share one key — the folded board.key() puts rule50 >= 14 in a
    /// different bucket, silently splitting dedup identity (and this test
    /// fails against it).
    #[test]
    fn canonical_key_ignores_the_fifty_move_clock() {
        let a = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 40").unwrap();
        let b = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 40 40").unwrap();
        assert_ne!(
            a.key(),
            b.key(),
            "the folded key differs by design at clock 40"
        );
        let state_a = position_from_board(&a, 0, vec![]);
        let state_b = position_from_board(&b, 0, vec![]);
        assert_eq!(
            state_a.canonical_key, state_b.canonical_key,
            "canonical_key must be clock-independent position identity"
        );
    }

    /// Replay must handle every special move kind — and must REJECT a path
    /// built from kind-less from/to reconstructions (the exact trap the
    /// in-repo demo fixture used to seed).
    #[test]
    fn replay_path_handles_castling_ep_and_promotion() {
        use crate::moves::MoveList;
        // A scripted line containing a castle, an en-passant capture, and a
        // promotion, all as canonical legal moves.
        let mut board =
            Board::from_fen("r3k2r/1Ppp1ppp/8/4pP2/8/8/PPPP2PP/R3K2R w KQkq e6 0 2").unwrap();
        let script = ["f5e6", "e8g8", "b7a8q", "g8h8"]; // ep, castle, promo, quiet
        let mut path = Vec::new();
        for text in script {
            let mut legal = MoveList::new();
            crate::movegen::generate_legal(&board, &mut legal);
            let mv = *legal
                .iter()
                .find(|m| m.to_string() == text)
                .unwrap_or_else(|| panic!("{text} must be legal"));
            path.push(mv.raw16());
            let gives_check = board.gives_check(mv);
            board.make_move(mv, gives_check);
        }
        let root =
            Board::from_fen("r3k2r/1Ppp1ppp/8/4pP2/8/8/PPPP2PP/R3K2R w KQkq e6 0 2").unwrap();
        let replayed = replay_path(&root, &path).expect("canonical path replays");
        assert_eq!(replayed.raw_key(), board.raw_key());

        // The same line encoded as bare from/to normals must FAIL loudly:
        // kind bits are part of move identity.
        let bad: Vec<u16> = path
            .iter()
            .map(|&raw| raw & 0x0fff) // strip kind + promo bits
            .collect();
        assert!(replay_path(&root, &bad).is_err());
    }

    #[test]
    fn corrupted_pcs_is_rejected_not_trusted() {
        let board = Board::from_fen(START_FEN).unwrap();
        let mut state = position_from_board(&board, 0, vec![]);
        state.pcs.truncate(4); // fewer nibbles than occupied squares
        assert!(board_from_position(&state).is_err());
        let mut state = position_from_board(&board, 0, vec![]);
        state.pcs[0] = 0x77; // piece_type 7 does not exist
        assert!(board_from_position(&state).is_err());
    }
}
