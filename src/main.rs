use chessengine::Board;

fn main() {
    let board = Board::startpos();
    println!("{board}");
    println!();
    println!("FEN: {}", board.to_fen());
}
