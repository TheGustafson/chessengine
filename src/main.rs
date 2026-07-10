//! Thin binary entry point (design/01 §3.1): a one-line delegator.
//!
//! Everything — the argv subcommands (`bench`, `perft`, `d`) and the
//! blocking UCI loop — lives in the library (`uci::run`, design/12 §3.1),
//! so all of it is reachable from `cargo test`.

fn main() {
    chessengine::uci::run();
}
