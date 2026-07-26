//! Batch legal-move policy masks for the JAX trainer (PyO3).
//!
//! Contract (trainer parity — the shared reference is the d40_features.rs
//! mask block and the python-chess prototype in `jax/archon_jax/data.py`):
//! input records are 32-byte bulletformat `ChessBoard`s, stm-canonicalized
//! at write time (every record decodes white-to-move; castling/ep are not
//! representable and are deliberately absent from the reconstruction). For
//! each row with a packed policy target (`pol_target_id >= 0`) the mask row
//! is `-30.0` everywhere except `0.0` on the canonical ids of the legal
//! moves; rows without a target stay all-zero (their CE is identically
//! zero). The packed target must land on a legal row — frame tripwire.

use chessengine::eval::nnue::policy_vocab::{self, POLICY_VOCAB};
use chessengine::movegen::generate_legal;
use chessengine::moves::MoveList;
use chessengine::{Board, Square};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const RECORD: usize = 32;
const KSQ_OFF: usize = 27;
const ILLEGAL: f32 = -30.0;


/// Reconstruct via FEN -> Board::from_fen: the sanctioned from-scratch setup
/// path, so ALL StateInfo (pins/blockers/checkers) is computed correctly.
/// NOTE: Board::empty + put_piece leaves pin info stale and generate_legal
/// then leaks pinned-piece moves — the bullet d40_features.rs mask block has
/// exactly that bug (its tripwire only asserts the TARGET is legal).
fn stored_board(rec: &[u8]) -> Board {
    const SYM: [char; 6] = ['P', 'N', 'B', 'R', 'Q', 'K'];
    let occ = u64::from_le_bytes(rec[0..8].try_into().unwrap());
    let pcs = &rec[8..24];
    // square -> piece char (stored nibble stream is occ-bit order)
    let mut chars = [0u8; 64];
    let mut bits = occ;
    let mut idx = 0usize;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let nibble = (pcs[idx / 2] >> (4 * (idx & 1))) & 0xF;
        idx += 1;
        let c = SYM[(nibble & 7) as usize];
        chars[sq] = if nibble & 8 == 0 { c as u8 } else { c.to_ascii_lowercase() as u8 };
    }
    let mut fen = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let ch = chars[rank * 8 + file];
            if ch == 0 {
                empty += 1;
            } else {
                if empty > 0 {
                    fen.push((b'0' + empty) as char);
                    empty = 0;
                }
                fen.push(ch as char);
            }
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    fen.push_str(" w - - 0 1");
    Board::from_fen(&fen).expect("reconstructed FEN must parse")
}

/// Fills `out` (POLICY_VOCAB f32s) for one record. Returns false on a frame
/// tripwire violation (packed target not legal).
fn mask_row(rec: &[u8], target: i64, out: &mut [f32]) -> bool {
    if target < 0 {
        // no packed move: all-zero mask, zero CE either way
        return true;
    }
    let board = stored_board(rec);
    let ksq = Square::new(rec[KSQ_OFF] & 63);
    let orient = policy_vocab::canonical_orient(false, ksq);
    out.iter_mut().for_each(|x| *x = ILLEGAL);
    let mut list = MoveList::new();
    generate_legal(&board, &mut list);
    for &m in list.iter() {
        let (fc, tc) = policy_vocab::canonical_from_to(m.from(), m.to(), orient);
        if let Some(mid) = policy_vocab::policy_move_id(fc, tc, m.promotion_piece()) {
            out[mid as usize] = 0.0;
        }
    }
    out[target as usize] == 0.0
}

/// recs: flat uint8 array of n*32 bytes; targets: int64 array of n ids
/// (-1 = no packed move). Returns (n, 1858) float32 additive masks.
#[pyfunction]
fn legal_move_masks<'py>(
    py: Python<'py>,
    recs: PyReadonlyArray1<'py, u8>,
    targets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rec_bytes: Vec<u8> = recs.as_slice()?.to_vec();
    let tgt: Vec<i64> = targets.as_slice()?.to_vec();
    if rec_bytes.len() % RECORD != 0 {
        return Err(PyValueError::new_err("recs length not a multiple of 32"));
    }
    let n = rec_bytes.len() / RECORD;
    if tgt.len() != n {
        return Err(PyValueError::new_err("targets length != record count"));
    }
    let (masks, ok) = py.allow_threads(|| {
        let mut masks = vec![0f32; n * POLICY_VOCAB];
        let ok = masks
            .par_chunks_mut(POLICY_VOCAB)
            .enumerate()
            .map(|(i, out)| mask_row(&rec_bytes[i * RECORD..(i + 1) * RECORD], tgt[i], out))
            .reduce(|| true, |a, b| a && b);
        (masks, ok)
    });
    if !ok {
        return Err(PyValueError::new_err(
            "policy frame bug: a packed target id is not in its legal mask",
        ));
    }
    let flat = numpy::PyArray1::from_vec(py, masks);
    Ok(flat.reshape([n, POLICY_VOCAB])?)
}

#[pymodule]
fn archon_mask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(legal_move_masks, m)?)?;
    m.add_function(wrap_pyfunction!(legal_move_ids, m)?)?;
    m.add("POLICY_VOCAB", POLICY_VOCAB)?;
    m.add("MAX_LEGAL", MAX_LEGAL)?;
    Ok(())
}

/// Legal-move IDs instead of a dense mask — the SAME information in ~29x
/// fewer bytes. The dense (N, 1858) f32 mask costs 61 MB per 8192-batch and
/// measured ~25% of trainer throughput (145k vs 197k pos/s): memset -30,
/// scatter ~35 zeros, PCIe copy, GPU elementwise add — for ~30 bits of real
/// information per row. The consumer scatters this back to the identical
/// dense mask ON DEVICE, so the loss is bit-identical.
///
/// Returns (n, MAX_LEGAL) i16, -1 padded. Rows with no packed move return an
/// all -1 row (the caller emits an all-zero mask for those, matching
/// `legal_move_masks`). Frame tripwire unchanged: the packed target must be
/// among the legal ids.
const MAX_LEGAL: usize = 128;

fn legal_ids_row(rec: &[u8], target: i64, out: &mut [i16]) -> bool {
    out.iter_mut().for_each(|x| *x = -1);
    if target < 0 {
        return true; // no packed move: all-pad row
    }
    let board = stored_board(rec);
    let ksq = Square::new(rec[KSQ_OFF] & 63);
    let orient = policy_vocab::canonical_orient(false, ksq);
    let mut list = MoveList::new();
    generate_legal(&board, &mut list);
    let mut n = 0usize;
    let mut saw_target = false;
    for &m in list.iter() {
        let (fc, tc) = policy_vocab::canonical_from_to(m.from(), m.to(), orient);
        if let Some(mid) = policy_vocab::policy_move_id(fc, tc, m.promotion_piece()) {
            if n < MAX_LEGAL {
                out[n] = mid as i16;
                n += 1;
            }
            if mid as i64 == target {
                saw_target = true;
            }
        }
    }
    saw_target
}

/// recs: flat uint8 array of n*32 bytes; targets: int64 array of n ids.
/// Returns (n, 128) int16 of canonical legal-move ids, -1 padded.
#[pyfunction]
fn legal_move_ids<'py>(
    py: Python<'py>,
    recs: PyReadonlyArray1<'py, u8>,
    targets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i16>>> {
    let rec_bytes: Vec<u8> = recs.as_slice()?.to_vec();
    let tgt: Vec<i64> = targets.as_slice()?.to_vec();
    if rec_bytes.len() % RECORD != 0 {
        return Err(PyValueError::new_err("recs length not a multiple of 32"));
    }
    let n = rec_bytes.len() / RECORD;
    if tgt.len() != n {
        return Err(PyValueError::new_err("targets length != record count"));
    }
    let (ids, ok) = py.allow_threads(|| {
        let mut ids = vec![-1i16; n * MAX_LEGAL];
        let ok = ids
            .par_chunks_mut(MAX_LEGAL)
            .enumerate()
            .map(|(i, out)| legal_ids_row(&rec_bytes[i * RECORD..(i + 1) * RECORD], tgt[i], out))
            .reduce(|| true, |a, b| a && b);
        (ids, ok)
    });
    if !ok {
        return Err(PyValueError::new_err(
            "policy frame bug: a packed target id is not among its legal ids",
        ));
    }
    let flat = numpy::PyArray1::from_vec(py, ids);
    Ok(flat.reshape([n, MAX_LEGAL])?)
}
