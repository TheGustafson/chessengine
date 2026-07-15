// Tiny wrapper around bulletformat (zero-dep, GPLv? -> actually MIT) that:
//   convert <in.txt> <out.bin>   -- bulletformat's own text->bin (ChessBoard)
//   inspect <bin> <txt> [K]      -- decode records via bulletformat's DataLoader
//                                   and cross-check the first K against the
//                                   source text lines (piece multiset + score
//                                   sign + result + king squares, per doc 15
//                                   §3.2.1/§3.2.4). Prints a few decoded boards.
//
// Discardable scaffold (D15/D28). No engine code touched.
use bulletformat::{BulletFormat, ChessBoard, DataLoader, convert_from_text};
use std::io::{BufRead, BufReader, BufWriter};

fn piece_char(colour: u8, pt: u8) -> char {
    let base = [b'P', b'N', b'B', b'R', b'Q', b'K'][pt as usize];
    if colour == 1 {
        base.to_ascii_lowercase() as char
    } else {
        base as char
    }
}

// counts[colour][pt]
fn record_counts(b: &ChessBoard) -> [[u32; 6]; 2] {
    let mut c = [[0u32; 6]; 2];
    for (nib, _sq) in *b {
        let colour = (nib >> 3) & 1;
        let pt = nib & 7;
        c[colour as usize][pt as usize] += 1;
    }
    c
}

// parse the board field of a FEN into counts[colour][pt] (white=0, black=1)
fn fen_counts(board_field: &str) -> [[u32; 6]; 2] {
    let mut c = [[0u32; 6]; 2];
    for ch in board_field.chars() {
        let (colour, up) = if ch.is_ascii_uppercase() {
            (0usize, ch)
        } else {
            (1usize, ch.to_ascii_uppercase())
        };
        let pt = match up {
            'P' => 0,
            'N' => 1,
            'B' => 2,
            'R' => 3,
            'Q' => 4,
            'K' => 5,
            _ => continue,
        };
        c[colour][pt] += 1;
    }
    c
}

fn decode_board_ascii(b: &ChessBoard) -> String {
    // print in the stored (canonical STM) orientation: square index -> rank/file
    let mut grid = [['.'; 8]; 8];
    for (nib, sq) in *b {
        let colour = (nib >> 3) & 1;
        let pt = nib & 7;
        let r = (sq / 8) as usize;
        let f = (sq % 8) as usize;
        grid[r][f] = piece_char(colour, pt);
    }
    let mut out = String::new();
    for r in (0..8).rev() {
        for f in 0..8 {
            out.push(grid[r][f]);
        }
        out.push('\n');
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("convert") => {
            let inp = &args[2];
            let out = &args[3];
            convert_from_text::<ChessBoard>(inp, out).expect("convert failed");
        }
        Some("convert-stdin") => {
            let out = &args[2];
            let stdin = std::io::stdin();
            let mut output = BufWriter::new(std::fs::File::create(out).expect("create output"));
            let mut records = Vec::with_capacity(65_536);
            let mut count = 0usize;
            for line in stdin.lock().lines() {
                let line = line.expect("read stdin");
                if line.trim().is_empty() {
                    continue;
                }
                records.push(line.parse::<ChessBoard>().expect("parse streamed record"));
                if records.len() == records.capacity() {
                    ChessBoard::write_to_bin(&mut output, &records).expect("write output");
                    count += records.len();
                    records.clear();
                }
            }
            ChessBoard::write_to_bin(&mut output, &records).expect("write output tail");
            count += records.len();
            eprintln!("convert-stdin: wrote {count} ChessBoard records to {out}");
        }
        Some("inspect") => {
            let bin = &args[2];
            let txt = &args[3];
            let k: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

            // load all records
            let mut recs: Vec<ChessBoard> = Vec::new();
            let loader = DataLoader::<ChessBoard>::new(bin, 64).expect("open bin");
            loader.map_positions(|p| recs.push(*p));

            // load all text lines
            let mut lines: Vec<String> = Vec::new();
            let f = BufReader::new(std::fs::File::open(txt).expect("open txt"));
            for l in f.lines() {
                let l = l.unwrap();
                if !l.trim().is_empty() {
                    lines.push(l);
                }
            }

            println!("records = {}, text lines = {}", recs.len(), lines.len());
            assert_eq!(recs.len(), lines.len(), "record/line count mismatch");

            let mut checked = 0usize;
            let mut mism = 0usize;
            for (i, (b, line)) in recs.iter().zip(lines.iter()).enumerate() {
                let parts: Vec<&str> = line.split('|').collect();
                let fen = parts[0].trim();
                let s_white: i32 = parts[1].trim().parse().expect("bad score in text");
                let r_white: f32 = parts[2].trim().parse().expect("bad result in text");
                let fen_parts: Vec<&str> = fen.split_whitespace().collect();
                let board_field = fen_parts[0];
                let stm = fen_parts[1]; // "w" or "b"

                // --- structural sanity on the decoded record ---
                let rc = record_counts(b);
                assert_eq!(rc[0][5], 1, "line {i}: not exactly one mover king");
                assert_eq!(rc[1][5], 1, "line {i}: not exactly one opp king");
                assert!(rc[0][0] <= 8 && rc[1][0] <= 8, "line {i}: >8 pawns");
                let total: u32 = rc.iter().flatten().sum();
                assert_eq!(
                    total,
                    b.occ().count_ones(),
                    "line {i}: occ popcount != nibble count"
                );
                assert!(b.score().unsigned_abs() < 2500, "line {i}: |score| >= 2500");
                assert!(
                    matches!(b.result, 0 | 1 | 2),
                    "line {i}: result not in 0/1/2"
                );

                // --- king square fields (doc 15 §3.2.1) ---
                let mut mover_ksq = 64u8;
                let mut opp_ksq_raw = 64u8;
                for (nib, sq) in *b {
                    if nib == 5 {
                        mover_ksq = sq;
                    }
                    if nib == 13 {
                        opp_ksq_raw = sq;
                    }
                }
                assert_eq!(b.our_ksq(), mover_ksq, "line {i}: ksq mismatch");
                assert_eq!(b.opp_ksq(), opp_ksq_raw ^ 56, "line {i}: opp_ksq mismatch");

                // --- cross-check against source text (canonicalisation, §3.2.4) ---
                let fc = fen_counts(board_field);
                let (exp, exp_score, exp_result) = if stm == "b" {
                    // record mover(=0) == text black(=1); record opp(=1) == text white(=0)
                    let mut swapped = [[0u32; 6]; 2];
                    swapped[0] = fc[1];
                    swapped[1] = fc[0];
                    (swapped, -s_white, 2 - (2.0 * r_white) as i32)
                } else {
                    (fc, s_white, (2.0 * r_white) as i32)
                };
                if rc != exp {
                    mism += 1;
                    if mism <= 5 {
                        eprintln!(
                            "line {i}: piece-count mismatch\n text: {line}\n rec counts: {rc:?}\n exp counts: {exp:?}"
                        );
                    }
                }
                assert_eq!(
                    b.score() as i32,
                    exp_score,
                    "line {i}: score sign mismatch (text {s_white} stm {stm})"
                );
                assert_eq!(
                    b.result as i32, exp_result,
                    "line {i}: result mismatch (text {r_white} stm {stm})"
                );

                if i < k {
                    println!("\n--- record {i} ---");
                    println!("text : {line}");
                    println!(
                        "stored score={} result={} ksq={} opp_ksq={}",
                        b.score(),
                        b.result,
                        b.our_ksq(),
                        b.opp_ksq()
                    );
                    print!("{}", decode_board_ascii(b));
                }
                checked += 1;
            }
            println!("\nchecked {checked} records; piece-count mismatches = {mism}");
            assert_eq!(mism, 0, "piece-count mismatches found");
            println!("ALL CHECKS PASSED");
        }
        _ => {
            eprintln!("usage: nnue_convert convert <in.txt> <out.bin>");
            eprintln!("       nnue_convert convert-stdin <out.bin>");
            eprintln!("       nnue_convert inspect <bin> <txt> [K]");
            std::process::exit(2);
        }
    }
}
