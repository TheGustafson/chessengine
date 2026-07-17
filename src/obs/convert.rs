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
        canonical_key: board.key(),
        path_from_episode_root,
    }
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
            assert_eq!(state.canonical_key, board.key(), "{fen}");
        }
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
