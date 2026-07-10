//! Shared helpers for the integration-test suite.
//!
//! Every file directly under `tests/` is compiled as its own crate; each one
//! pulls this module in with `mod common;` and uses only a subset of it,
//! hence the file-wide `dead_code` allowance. Deliberately dependency-light:
//! std only.
#![allow(dead_code)]

use chessengine::{Board, Piece, Square};
use std::fmt;
use std::fs;
use std::path::PathBuf;

/// Absolute path of a file under `tests/data/`.
pub fn data_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(file_name)
}

// ---------------------------------------------------------------------------
// Position-building conveniences
// ---------------------------------------------------------------------------

/// Parse a FEN, panicking with context on failure. Tests want a loud message
/// naming the offending position, not `Result` plumbing.
pub fn board(fen: &str) -> Board {
    match Board::from_fen(fen) {
        Ok(board) => board,
        Err(e) => panic!("bad test FEN {fen:?}: {e}"),
    }
}

/// Build a position from `(FEN piece letter, square)` pairs, e.g.
/// `place(&[('K', "e1"), ('k', "e8"), ('P', "e4")])`. Uppercase = white.
/// Goes through `Board::empty()` + `put_piece` per the testing conventions.
pub fn place(pieces: &[(char, &str)]) -> Board {
    let mut board = Board::empty();
    for &(letter, square) in pieces {
        let piece =
            Piece::from_char(letter).unwrap_or_else(|| panic!("bad piece letter {letter:?}"));
        let square: Square = square
            .parse()
            .unwrap_or_else(|_| panic!("bad square {square:?}"));
        board.put_piece(piece, square);
    }
    board
}

// ---------------------------------------------------------------------------
// FEN corpus loader (tests/data/fens.txt)
// ---------------------------------------------------------------------------

/// Parse the FEN-corpus format: one FEN per line; blank lines and lines
/// starting with `#` are skipped; surrounding whitespace is trimmed.
pub fn parse_fen_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Load `tests/data/fens.txt`. Panics if missing: the corpus is checked in
/// alongside the tests that consume it.
pub fn load_fens() -> Vec<String> {
    let path = data_path("fens.txt");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    parse_fen_lines(&text)
}

// ---------------------------------------------------------------------------
// Perft EPD loader (tests/data/perft.epd)
// ---------------------------------------------------------------------------

/// One `tests/data/perft.epd` record: a position plus its known perft counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerftRecord {
    pub fen: String,
    /// `(depth, node count)` pairs in file order.
    pub counts: Vec<(u32, u64)>,
}

/// Error from the perft EPD parser, carrying the 1-based line number
/// (0 for file-level errors such as a missing file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerftEpdError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for PerftEpdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "perft EPD: {}", self.message)
        } else {
            write!(f, "perft EPD line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for PerftEpdError {}

/// Parse perft EPD text (design/06 §3.4 data, one record per line):
///
/// ```text
/// <FEN> ;D1 20 ;D2 400 ;D3 8902
/// ```
///
/// Defensive by design: blank lines and `#` comments are skipped; whitespace
/// around `;` is flexible; `D`/`d` are both accepted; trailing or doubled
/// semicolons are tolerated. Structurally broken lines fail with the line
/// number rather than being silently dropped — a typo in the oracle data must
/// be a test failure, not a skipped case.
pub fn parse_perft_epd(text: &str) -> Result<Vec<PerftRecord>, PerftEpdError> {
    let err = |line: usize, message: String| PerftEpdError { line, message };
    let mut records = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // `split` always yields at least one segment.
        let mut segments = trimmed.split(';').map(str::trim);
        let fen = segments.next().unwrap();
        if fen.is_empty() {
            return Err(err(line, "missing FEN before the first ';'".to_string()));
        }
        let mut counts = Vec::new();
        for segment in segments {
            if segment.is_empty() {
                continue; // trailing or doubled semicolon
            }
            let mut tokens = segment.split_whitespace();
            let (Some(depth_token), Some(nodes_token), None) =
                (tokens.next(), tokens.next(), tokens.next())
            else {
                return Err(err(
                    line,
                    format!("expected \"D<depth> <nodes>\", found {segment:?}"),
                ));
            };
            let Some(depth_digits) = depth_token.strip_prefix(['D', 'd']) else {
                return Err(err(
                    line,
                    format!("depth token {depth_token:?} does not start with 'D'"),
                ));
            };
            let depth: u32 = depth_digits
                .parse()
                .map_err(|_| err(line, format!("bad depth in {depth_token:?}")))?;
            let nodes: u64 = nodes_token
                .parse()
                .map_err(|_| err(line, format!("bad node count {nodes_token:?}")))?;
            counts.push((depth, nodes));
        }
        if counts.is_empty() {
            return Err(err(line, "no \"D<depth> <nodes>\" entries".to_string()));
        }
        records.push(PerftRecord {
            fen: fen.to_string(),
            counts,
        });
    }
    Ok(records)
}

/// Load `tests/data/perft.epd`. Returns `Err` (rather than panicking) if the
/// file is missing so tests written before the perft data lands can still
/// compile and choose how to react.
pub fn load_perft_epd() -> Result<Vec<PerftRecord>, PerftEpdError> {
    let path = data_path("perft.epd");
    let text = fs::read_to_string(&path).map_err(|e| PerftEpdError {
        line: 0,
        message: format!("cannot read {}: {e}", path.display()),
    })?;
    parse_perft_epd(&text)
}

// ---------------------------------------------------------------------------
// Self-tests: the parsers are exercised against literal samples so they do
// not depend on the data files existing yet.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chessengine::{Color, PieceType};

    const SAMPLE_EPD: &str = "\
# sample perft oracle data
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ;D1 20 ;D2 400 ;D3 8902

8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 ; d1 14; D2 191 ;
";

    #[test]
    fn perft_epd_parses_sample() {
        let records = parse_perft_epd(SAMPLE_EPD).expect("sample must parse");
        assert_eq!(records.len(), 2);

        assert_eq!(
            records[0].fen,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
        assert_eq!(records[0].counts, vec![(1, 20), (2, 400), (3, 8902)]);

        // Flexible spacing, lowercase 'd', and a trailing ';' all tolerated.
        assert_eq!(records[1].fen, "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        assert_eq!(records[1].counts, vec![(1, 14), (2, 191)]);
    }

    #[test]
    fn perft_epd_rejects_broken_lines() {
        let broken = [
            ";D1 20",                          // no FEN
            "8/8/8/8/8/8/8/8 w - - 0 1",       // no depth entries
            "8/8/8/8/8/8/8/8 w - - ;X1 20",    // depth token without D
            "8/8/8/8/8/8/8/8 w - - ;D1",       // missing node count
            "8/8/8/8/8/8/8/8 w - - ;D1 x",     // non-numeric node count
            "8/8/8/8/8/8/8/8 w - - ;Dx 20",    // non-numeric depth
            "8/8/8/8/8/8/8/8 w - - ;D1 20 30", // trailing junk in a segment
        ];
        for text in broken {
            let result = parse_perft_epd(text);
            assert!(result.is_err(), "expected rejection of {text:?}");
        }
    }

    #[test]
    fn perft_epd_reports_line_numbers() {
        let text = "# comment\n8/8/8/8/8/8/8/8 w - - ;D1 1\nbroken ;D1\n";
        let error = parse_perft_epd(text).unwrap_err();
        assert_eq!(error.line, 3);
    }

    #[test]
    fn fen_lines_skip_comments_and_blanks() {
        let text = "# heading\n\n  8/8/8/8/8/8/8/4K2k w - - 0 1  \n# tail\n";
        assert_eq!(
            parse_fen_lines(text),
            vec!["8/8/8/8/8/8/8/4K2k w - - 0 1".to_string()]
        );
    }

    #[test]
    fn place_builds_positions_through_put_piece() {
        let board = place(&[('K', "e1"), ('k', "e8"), ('P', "e4"), ('n', "f6")]);
        assert_eq!(
            board.piece_on(Square::E1),
            Some(Piece::new(Color::White, PieceType::King))
        );
        assert_eq!(
            board.piece_on(Square::F6),
            Some(Piece::new(Color::Black, PieceType::Knight))
        );
        assert_eq!(board.all_occupancy().count(), 4);
    }
}
