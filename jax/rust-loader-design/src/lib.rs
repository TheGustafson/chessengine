//! DESIGN ONLY — loader candidate (b): PyO3 bindings for the JAX pipeline.
//! Written against the APIs read in d40_features.rs / bullet_lib; NOT compiled
//! yet (cargo builds are forbidden while the GPU trainer runs — build-window
//! item). Expect minor signature fixes at first build.
//!
//! Two exports, mirroring the split the study recommends:
//!
//! 1. `SfBinpackStream` — the existing sfbinpack streaming loader (decode +
//!    D46 quiet filter + deterministic hash skip / heldout band), yielding
//!    bulletformat records as (N, 32) u8 numpy arrays (zero-copy views of the
//!    chunk buffer). This is the ONLY component python cannot replicate
//!    cheaply (the binpack format needs the reference decoder); everything
//!    downstream already runs at 640k pos/s in numpy.
//!
//! 2. `map_features_np` — batch feature mapping through the engine's canonical
//!    `halfka_index`/`threat_index`/`policy_vocab` (single-implementation
//!    guarantee), for use if the numpy extractor ever drifts or a future
//!    feature set outgrows vectorized numpy. Returns the same -1-padded
//!    (N, 32)/(N, 160) index tensors as archon_jax.features.map_batch, plus
//!    the legal-move mask computed with the engine's real movegen (which the
//!    python-chess prototype only approximates at ~10k pos/s).

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use bullet_lib::value::loader::{self, SfBinpackLoader};
use bulletformat::ChessBoard;
use chessengine::eval::nnue::policy_vocab;
use chessengine::eval::nnue::sota_features::{for_each_halfka, for_each_threat, HALFKA_DIMS};
use chessengine::{Board, Color, Piece, PieceType, Square};

/// The D46 trainer-parity quiet filter — the same predicate as
/// d40_features.rs::sfbinpack_filter (the normative copy).
fn quiet_filter(entry: &loader::sfbinpack::TrainingDataEntry) -> bool {
    use loader::sfbinpack::{MoveType, PieceType as SfPt};
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == SfPt::None
}

/// Streaming sfbinpack reader: `next_chunk()` returns an (N, 32) u8 array of
/// bulletformat records, or None at EOF. GIL released during decode.
#[pyclass]
struct SfBinpackStream {
    // Implementation note: SfBinpackLoader's read_chunks is callback-based;
    // run it on a worker thread feeding a bounded channel, so next_chunk()
    // is a simple recv. (Same shape as bullet's own ReadMapLoader.)
    rx: std::sync::mpsc::Receiver<Vec<ChessBoard>>,
    _worker: std::thread::JoinHandle<()>,
}

#[pymethods]
impl SfBinpackStream {
    #[new]
    #[pyo3(signature = (path, buffer_mb=256, threads=4, skip_pct=0))]
    fn new(path: String, buffer_mb: usize, threads: usize, skip_pct: u64) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let worker = std::thread::spawn(move || {
            let filter: fn(&loader::sfbinpack::TrainingDataEntry) -> bool = quiet_filter;
            let loader = SfBinpackLoader::new(&path, buffer_mb, threads, filter);
            // hash-skip (position_hash % 100 >= skip_pct) applied here, same
            // FNV construction as d40_features.rs::position_hash.
            let _ = (loader, tx, skip_pct); // read_chunks -> tx.send(chunk.to_vec())
            todo!("wire read_chunks -> channel at build time");
        });
        Self { rx, _worker: worker }
    }

    fn next_chunk<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<u8>>> {
        let chunk = py.allow_threads(|| self.rx.recv().ok())?;
        let n = chunk.len();
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, n * 32).to_vec()
        };
        Some(
            numpy::ndarray::Array2::from_shape_vec((n, 32), bytes)
                .unwrap()
                .into_pyarray(py),
        )
    }
}

/// Batch feature mapping through the engine's canonical index functions.
/// records: (N, 32) u8. Returns (hk_stm, hk_ntm, th_stm, th_ntm) i32 arrays,
/// -1-padded, shapes (N,32)/(N,160) — drop-in for archon_jax.features.
#[pyfunction]
fn map_features_np<'py>(
    py: Python<'py>,
    records: numpy::PyReadonlyArray2<'py, u8>,
) -> PyResult<(
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<i32>>,
)> {
    let recs = records.as_array();
    let n = recs.shape()[0];
    let (mut hk, mut th) = (
        numpy::ndarray::Array2::from_elem((2 * n, 32), -1i32),
        numpy::ndarray::Array2::from_elem((2 * n, 160), -1i32),
    );
    py.allow_threads(|| {
        // rayon par_iter over rows at build time; scalar sketch:
        for i in 0..n {
            let rec: &ChessBoard = unsafe { &*(recs.row(i).as_ptr() as *const ChessBoard) };
            let board = board_from_record(rec);
            for (p, persp) in [Color::White, Color::Black].iter().enumerate() {
                let (mut a, mut b) = (0usize, 0usize);
                for_each_halfka(&board, *persp, |idx| {
                    hk[[p * n + i, a]] = idx as i32;
                    a += 1;
                });
                for_each_threat(&board, *persp, |idx| {
                    th[[p * n + i, b]] = idx as i32;
                    b += 1;
                });
            }
        }
    });
    let _ = (HALFKA_DIMS, policy_vocab::POLICY_VOCAB);
    // split stm/ntm halves at build time
    todo!("return the four arrays (split + into_pyarray)");
}

fn board_from_record(rec: &ChessBoard) -> Board {
    // identical to d40_features.rs::stored_board (lines 169-191)
    const PT: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];
    let mut board = Board::empty();
    let mut occ = rec.occ;
    let mut idx = 0usize;
    while occ != 0 {
        let sq = occ.trailing_zeros() as u8;
        occ &= occ - 1;
        let nibble = (rec.pcs[idx / 2] >> (4 * (idx & 1))) & 0xF;
        idx += 1;
        let color = if nibble & 8 == 0 { Color::White } else { Color::Black };
        board.put_piece(Piece::new(color, PT[(nibble & 7) as usize]), Square::new(sq));
    }
    board
}

#[pymodule]
fn archon_loader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SfBinpackStream>()?;
    m.add_function(wrap_pyfunction!(map_features_np, m)?)?;
    Ok(())
}
