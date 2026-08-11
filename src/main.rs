//! Thin binary entry point (design/01 §3.1). The argv subcommands (`bench`,
//! `perft`, `d`) and the blocking UCI loop all live in the library
//! (`uci::run`), so all of it is reachable from `cargo test`.

fn main() {
    chessengine::uci::run();
}
