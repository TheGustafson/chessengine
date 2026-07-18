# Narration — Video 02: Board Representation — Bitboards

> Generated from `scripts/02-narration.json` by `pipeline/gen_script_md.py` — edit the JSON, not this file.

- **Voice:** `en_US-ryan-high` (Piper), length_scale 1.06
- **Per-beat pad:** 0.55s   **inter-scene gap:** 0.5s   **fps:** 30
- **Total runtime:** 7:11.97 (432.0s), 1254 words

---

## Scene 1 · The question  `[S1_hook]`  —  start 0:00.00, 32.9s

**`s1a`**  ·  @0:00.00  ·  audio 11.5s / budget 12.1s

> A chessboard is eight by eight: sixty-four squares, each one empty or holding one of twelve kinds of piece. That is a tiny amount of information. You could write down any position in sixty-four bytes.

**`s1b`**  ·  @0:12.07  ·  audio 14.0s / budget 14.6s

> So the interesting question is not whether a computer can store a board. It is how to store it, so that the questions an engine asks billions of times a second are almost free. Which squares does this bishop attack? Is this square defended? Generate every pawn move.

**`s1c`**  ·  @0:26.67  ·  audio 5.2s / budget 5.8s

> The answer, used by every strong engine, is the bitboard. Let us build it from first principles.


## Scene 2 · Squares as numbers  `[S2_squares]`  —  start 0:32.93, 60.6s

**`s2a`**  ·  @0:32.93  ·  audio 15.3s / budget 15.9s

> First, every square needs a name that a computer likes. ARCHON numbers them zero to sixty-three. a1 is zero, b1 is one, along the first rank to h1, which is seven. Then a2 is eight, and so on, all the way up to h8, which is sixty-three.

**`s2b`**  ·  @0:48.80  ·  audio 7.3s / budget 7.8s

> This is the little-endian rank-file mapping, and the rule behind it is just this: index equals rank times eight, plus file.

**`s2c`**  ·  @0:56.63  ·  audio 16.9s / budget 17.5s

> That choice pays off at once. Because the file is the low three bits of the index, and the rank is the next three bits, you pull them apart with a mask and a shift. No division needed. Take e4. Its index is twenty-eight. The low three bits are four, the e-file. The next three bits are three, the fourth rank.

**`s2d`**  ·  @1:14.10  ·  audio 18.4s / budget 18.9s

> Why this order, instead of counting down from a8, which reads more like a printed diagram? Because now bit arithmetic mirrors board geometry. Shift the whole number left by eight, and every piece moves up one rank. Shift by one, and it moves east one file. Geometry, done with arithmetic. That is the payoff we are chasing.


## Scene 3 · A board is a set of squares  `[S3_bitboard]`  —  start 1:33.53, 46.1s

**`s3a`**  ·  @1:33.53  ·  audio 16.7s / budget 17.3s

> Here is the key idea. A sixty-four-bit integer has exactly sixty-four bits. A chessboard has exactly sixty-four squares. So one u64 can represent any set of squares at all: bit i is one if square i is in the set, and zero if it is not. That integer is a bitboard.

**`s3b`**  ·  @1:50.80  ·  audio 15.7s / budget 16.3s

> ARCHON wraps the raw integer in a newtype, Bitboard, a struct around a single u64, so the compiler will never let you confuse a set of squares with some other number. It exposes behavior, the set operations, and hides the representation, the raw bits.

**`s3c`**  ·  @2:07.10  ·  audio 11.5s / budget 12.0s

> And you do not keep just one. A full position is a whole collection of bitboards: the white pawns, the black knights, all white pieces, every occupied square. One board for each question you might want to ask.


## Scene 4 · Set operations, one instruction each  `[S4_setops]`  —  start 2:19.63, 46.0s

**`s4a`**  ·  @2:19.63  ·  audio 9.2s / budget 9.8s

> And this is why the representation is transformative. Set operations become single machine instructions, each one acting on all sixty-four squares at the same time.

**`s4b`**  ·  @2:29.43  ·  audio 4.5s / budget 5.0s

> The union of two sets, the squares in either one, is a bitwise OR.

**`s4c`**  ·  @2:34.47  ·  audio 3.5s / budget 4.0s

> The intersection, the squares in both, is a bitwise AND.

**`s4d`**  ·  @2:38.50  ·  audio 4.9s / budget 5.5s

> The difference, squares in the first but not the second, is an AND with the complement of the second.

**`s4e`**  ·  @2:44.00  ·  audio 4.4s / budget 4.9s

> And the complement itself, every square not in the set, is a bitwise NOT.

**`s4f`**  ·  @2:48.93  ·  audio 15.6s / budget 16.2s

> So a loaded question, like which of my pawns are attacked by an enemy piece, becomes: my pawns, AND, enemy attacks. One instruction. Geometry has turned into arithmetic, and arithmetic on sixty-four-bit words is exactly what a processor is fastest at.


## Scene 5 · A picture  `[S5_knight]`  —  start 3:05.60, 23.4s

**`s5a`**  ·  @3:05.60  ·  audio 10.6s / budget 11.2s

> Bitboards are easier to believe once you actually see one. Here is the set of squares a knight on d4 attacks: eight squares, and as a u64, one specific sixty-four-bit number.

**`s5b`**  ·  @3:16.77  ·  audio 11.1s / budget 11.7s

> ARCHON has this precomputed for all sixty-four possible knight squares. However you obtain it, the result is just a bitboard, ready to be ANDed and ORed with everything else on the board.


## Scene 6 · Three bit tricks  `[S6_tricks]`  —  start 3:28.97, 67.4s

**`s6a`**  ·  @3:28.97  ·  audio 13.2s / budget 13.8s

> Three primitive operations show up constantly, and the processor has a dedicated instruction for each. First, population count: how many squares are in the set? One instruction. How many pawns do I have is a single popcount.

**`s6b`**  ·  @3:42.77  ·  audio 8.5s / budget 9.1s

> Second, the least significant bit: which is the lowest square in the set? That is the count of trailing zeros, which is exactly the index of the lowest one-bit.

**`s6c`**  ·  @3:51.87  ·  audio 13.2s / budget 13.7s

> Third, pop the lowest bit: remove and return that square. Clearing the lowest set bit is the classic trick, x AND x minus one. Repeat until the board is empty, and you have visited every piece. This is the bitboard's for-loop.

**`s6d`**  ·  @4:05.60  ·  audio 10.4s / budget 11.0s

> Watch it run across the knights. Each step hands you one square, in ascending order, and the loop runs exactly as many times as there are pieces to visit. No wasted work, no branch per square.

**`s6e`**  ·  @4:16.60  ·  audio 18.7s / budget 19.3s

> One last family: directional shifts. Because of our mapping, shifting left by eight moves every square up a rank. So every white pawn's single push is just the pawns, shifted left by eight. The whole pawn army advances in one instruction, with a small mask to stop a piece on the h-file from wrapping around the edge of the board.


## Scene 7 · The other half of the hybrid  `[S7_mailbox]`  —  start 4:36.37, 60.7s

**`s7a`**  ·  @4:36.37  ·  audio 14.6s / budget 15.2s

> Bitboards are brilliant at which squares have property X. But they are clumsy at the opposite question: what piece, if any, is on this one square? With bitboards alone you would have to test up to twelve boards to find the answer, and that question comes up on every single capture.

**`s7b`**  ·  @4:51.57  ·  audio 16.8s / budget 17.4s

> So ARCHON keeps a second, redundant representation right alongside them: a plain array of sixty-four slots, each holding a piece, or nothing. It is called a mailbox, each square a little box you can peek into. Now, what is on d5, is a single array lookup, done in constant time.

**`s7c`**  ·  @5:08.97  ·  audio 13.7s / budget 14.3s

> Two copies of the same fact must never disagree. So every placement change funnels through two private methods, put_piece and remove_piece, which update the bitboards and the mailbox together. One code path, so the two layers can never drift out of sync.

**`s7d`**  ·  @5:23.23  ·  audio 12.8s / budget 13.4s

> That is the hybrid design: bitboards as the primary representation, for fast set algebra, and a mailbox as a constant-time index, for point lookups. A little extra memory, in exchange for both kinds of speed.


## Scene 8 · The full board  `[S8_layout]`  —  start 5:37.10, 65.7s

**`s8a`**  ·  @5:37.10  ·  audio 10.2s / budget 10.8s

> Put it all together, and here is the heart of ARCHON's board. Twelve piece bitboards, indexed by color and piece type. White pawns, white knights, on up to black king, one board for each.

**`s8b`**  ·  @5:47.87  ·  audio 15.5s / budget 16.1s

> Plus two aggregates, kept purely for speed. Color occupancy, the union of one side's six boards. And all occupancy, every occupied square, which the sliding pieces depend on. Redundant, but computing them once and keeping them saves work on every query.

**`s8c`**  ·  @6:03.93  ·  audio 15.4s / budget 16.0s

> Then the mailbox for point lookups, and a single bit, side to move, the flag that flips every ply. And the pieces themselves are small enum values, laid out from least to most valuable, pawn up to queen, then king, so that move-ordering routines can simply iterate in order.

**`s8d`**  ·  @6:19.90  ·  audio 21.9s / budget 22.4s

> Every file downstream is written in this vocabulary. Move generation is a series of ANDs and ORs against precomputed attack sets. Hashing indexes a random key by piece and square. Evaluation loops over these boards and pops squares. The board is not a detail you set up and forget. It is the substrate whose shape dictates how every later algorithm is written.


## Scene 9 · What you now know  `[S9_recap]`  —  start 6:42.83, 29.1s

**`s9a`**  ·  @6:42.83  ·  audio 17.7s / budget 18.3s

> So, to recap. Squares are numbered zero to sixty-three, so that file and rank are just a mask and a shift. A bitboard is a u64 read as a set of squares, where union, intersection, and complement are single instructions. And a redundant mailbox answers what is on this square in constant time.

**`s9b`**  ·  @7:01.10  ·  audio 10.3s / budget 10.9s

> Next, we take these attack sets and actually generate moves, including one of the most beautiful hacks in the whole field, magic bitboards. That is file three, move generation.

