"""ARCHON learning series — Pilot video for doc 02, Board Representation (bitboards).

Nine scenes, each a NarratedScene whose beats are sized by manifests/02-timing.json.
Render one scene:  venv/bin/manim render -ql --fps 15 -r 854,480 scenes/pilot02.py S2Squares
Full list + orchestration: pipeline/render.sh
"""
from __future__ import annotations

import numpy as np
from manim import (
    VGroup, Text, Square, Rectangle, RoundedRectangle, Line, Arrow, Dot,
    SurroundingRectangle, Brace, FadeIn, FadeOut, Create, Write, Transform,
    ReplacementTransform, Indicate, Circumscribe, GrowFromCenter, Flash,
    UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR,
)

from common import (
    NarratedScene, ChessBoard, piece, code_block, name_to_index, sq_index,
    BG, LIGHT_SQ, DARK_SQ, SET1, SET2, INTER, TXT, MUT, ACCENT, AMBER, GREEN,
    RED, CODE_TXT, CODE_BG, SANS, MONO, FRAME_W, FRAME_H,
)

HEADER = "02 · Board Representation"


def header_group(subtitle: str) -> VGroup:
    t = Text(HEADER, font=SANS, font_size=28, color=MUT)
    dot = Text("—", font=SANS, font_size=28, color=ACCENT)
    sub = Text(subtitle, font=SANS, font_size=28, color=ACCENT)
    row = VGroup(t, dot, sub).arrange(RIGHT, buff=0.22)
    row.to_edge(UP, buff=0.30)
    line = Line(row.get_left() + DOWN * 0.16, row.get_right() + DOWN * 0.16,
                stroke_color="#2b3340", stroke_width=2)
    return VGroup(row, line)


START_POS = {
    0: ("w", "R"), 1: ("w", "N"), 2: ("w", "B"), 3: ("w", "Q"),
    4: ("w", "K"), 5: ("w", "B"), 6: ("w", "N"), 7: ("w", "R"),
    56: ("b", "R"), 57: ("b", "N"), 58: ("b", "B"), 59: ("b", "Q"),
    60: ("b", "K"), 61: ("b", "B"), 62: ("b", "N"), 63: ("b", "R"),
}
for f in range(8):
    START_POS[8 + f] = ("w", "P")
    START_POS[48 + f] = ("b", "P")


def start_pieces(board: ChessBoard) -> VGroup:
    g = VGroup()
    for idx, (c, t) in START_POS.items():
        p = piece(c, t, size=board.size)
        p.move_to(board.center(idx))
        g.add(p)
    return g


# ============================================================================ Scene 1
class S1Hook(NarratedScene):
    SCENE_KEY = "S1_hook"

    def construct(self):
        head = header_group("the question")
        board = ChessBoard(size=0.62).shift(LEFT * 3.1 + DOWN * 0.25)
        self.assert_in_frame(board, label="board")

        with self.beat("s1a"):
            self.play_t(FadeIn(head, shift=DOWN * 0.2), run_time=0.8)
            self.play_t(Create(board), run_time=1.4)
            pieces = start_pieces(board)
            self.play_t(FadeIn(pieces, lag_ratio=0.04), run_time=2.4)
            cap = VGroup(
                Text("64 squares", font=SANS, font_size=40, color=TXT),
                Text("12 kinds of piece", font=SANS, font_size=40, color=TXT),
                Text("≈ 64 bytes to store", font=SANS, font_size=40, color=AMBER),
            ).arrange(DOWN, buff=0.42, aligned_edge=LEFT)
            cap.shift(RIGHT * 3.6 + UP * 0.1)
            self.assert_in_frame(cap, label="s1a-cap")
            self.assert_no_overlap(board, cap, label="board/cap")
            for line in cap:
                self.assert_min_height(line, 0.30, "cap-line")
            self.play_t(FadeIn(cap[0], shift=RIGHT * 0.2), run_time=0.8)
            self.play_t(FadeIn(cap[1], shift=RIGHT * 0.2), run_time=0.8)
            self.play_t(FadeIn(cap[2], shift=RIGHT * 0.2), run_time=0.9)
            self.play_t(Circumscribe(cap[2], color=AMBER), run_time=1.2)
        self._pieces = pieces
        self._cap = cap
        self._board = board

        with self.beat("s1b"):
            self.play_t(FadeOut(self._cap), run_time=0.6)
            qs = VGroup(
                Text("Which squares does this bishop attack?", font=SANS, font_size=27, color=TXT),
                Text("Is this square defended?", font=SANS, font_size=27, color=TXT),
                Text("Generate every pawn move.", font=SANS, font_size=27, color=TXT),
            ).arrange(DOWN, buff=0.46, aligned_edge=LEFT)
            self.place_right(qs, cy=0.9)
            self.assert_in_frame(qs, label="s1b-qs")
            self.assert_no_overlap(board, qs, label="board/qs")
            # illustrate the bishop question: light the c1 bishop's diagonal
            diag = [name_to_index(s) for s in ["d2", "e3", "f4", "g5", "h6"]]
            ov = board.overlay(diag, color=SET1, opacity=0.65, inset=0.06)
            for i, q in enumerate(qs):
                self.play_t(FadeIn(q, shift=RIGHT * 0.15), run_time=0.8)
                if i == 0:
                    self.play_t(FadeIn(ov, lag_ratio=0.25), run_time=1.4)
            billions = Text("billions of times a second", font=SANS, font_size=34,
                            color=AMBER).next_to(qs, DOWN, buff=0.7).align_to(qs, LEFT)
            self.assert_in_frame(billions, label="billions")
            self.play_t(Write(billions), run_time=1.2)
            self.play_t(Indicate(billions, color=AMBER, scale_factor=1.12), run_time=1.2)
            self._qs = qs
            self._billions = billions
            self._ov = ov

        with self.beat("s1c"):
            self.play_t(FadeOut(self._qs), FadeOut(self._billions),
                        FadeOut(self._ov), FadeOut(self._pieces), run_time=0.9)
            answer = Text("The answer: the bitboard.", font=SANS, font_size=44, color=ACCENT)
            self.place_right(answer, cy=0.0)
            self.assert_in_frame(answer, label="answer")
            self.play_t(Write(answer), run_time=1.4)
            self.play_t(Indicate(answer, color=ACCENT, scale_factor=1.08), run_time=1.0)
        self.finish_scene()


# ============================================================================ Scene 2
class S2Squares(NarratedScene):
    SCENE_KEY = "S2_squares"

    def construct(self):
        self.add(header_group("squares as numbers"))
        board = ChessBoard(size=0.66).shift(LEFT * 3.2 + DOWN * 0.2)
        self.add(board)
        self.assert_in_frame(board, label="board")

        # index labels on every square
        labels = VGroup()
        for idx in range(64):
            n = Text(str(idx), font=MONO, font_size=board.size * 26, color=TXT)
            n.move_to(board.center(idx))
            labels.add(n)

        with self.beat("s2a"):
            # reveal indices rank by rank, bottom to top
            self.play_t(FadeIn(labels[0:8], lag_ratio=0.15), run_time=1.6)
            key = VGroup(
                Text("a1 = 0", font=MONO, font_size=34, color=ACCENT),
                Text("h1 = 7", font=MONO, font_size=34, color=TXT),
                Text("h8 = 63", font=MONO, font_size=34, color=AMBER),
            ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).shift(RIGHT * 3.5 + UP * 0.3)
            self.assert_in_frame(key, label="s2a-key")
            self.assert_no_overlap(board, key, label="board/key")
            self.play_t(FadeIn(key[0]), run_time=0.7)
            self.play_t(FadeIn(key[1]), run_time=0.7)
            self.play_t(FadeIn(labels[8:56], lag_ratio=0.01), run_time=2.0)
            self.play_t(FadeIn(labels[56:64], lag_ratio=0.1), run_time=1.2)
            self.play_t(FadeIn(key[2]), Indicate(labels[63], color=AMBER), run_time=1.2)
            self.play_t(Indicate(labels[0], color=ACCENT), run_time=1.0)
        self._labels = labels
        self._key = key
        self._board = board

        with self.beat("s2b"):
            formula = code_block(["index = rank * 8 + file"], font_size=30,
                                 t2c={"rank": ACCENT, "file": AMBER})
            title = Text("little-endian rank-file", font=SANS, font_size=30, color=MUT)
            grp2b = VGroup(title, formula).arrange(DOWN, buff=0.4)
            self.place_right(grp2b, cy=0.1)
            self.assert_in_frame(VGroup(formula, title), label="s2b")
            self.play_t(FadeOut(self._key), run_time=0.5)
            self.play_t(FadeIn(title, shift=DOWN * 0.15), run_time=0.7)
            self.play_t(Create(formula), run_time=1.2)
            self._formula = formula
            self._title2 = title

        with self.beat("s2c"):
            self.play_t(FadeOut(self._formula), FadeOut(self._title2), run_time=0.5)
            # highlight e4, show its index 28 and the bit split
            e4 = name_to_index("e4")  # 28
            hl = board.overlay([e4], color=SET1, opacity=0.45, inset=0.04)
            self.play_t(FadeIn(hl), Indicate(self._labels[e4], color=ACCENT, scale_factor=1.4),
                        run_time=1.0)
            # bit panel: 28 = 011 100 -> rank=011=3, file=100=4
            bits = self.bit_split_panel(rank_bits="011", file_bits="100",
                                        idx=28, note1="rank = 3  →  4th rank",
                                        note2="file = 4  →  e-file")
            self.place_right(bits, cy=0.0)
            self.assert_in_frame(bits, label="s2c-bits")
            self.assert_no_overlap(board, bits, label="board/bits")
            self.play_t(FadeIn(bits, shift=RIGHT * 0.2), run_time=1.2)
            self.play_t(Indicate(bits[1], color=AMBER), run_time=1.1)  # file bits
            self.play_t(Indicate(bits[0], color=ACCENT), run_time=1.1)  # rank bits
            self._hl = hl
            self._bits = bits

        with self.beat("s2d"):
            self.play_t(FadeOut(self._bits), run_time=0.5)
            note = VGroup(
                Text("<< 8   moves every piece up one rank", font=MONO, font_size=25, color=ACCENT),
                Text("<< 1   moves east one file", font=MONO, font_size=25, color=AMBER),
                Text("geometry  =  arithmetic", font=SANS, font_size=34, color=TXT),
            ).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
            self.place_right(note, cy=0.0)
            self.assert_in_frame(note, label="s2d-note")
            self.assert_no_overlap(board, note, label="board/note")
            # animate an up-shift: move an overlay up one rank
            e4 = name_to_index("e4")
            self.play_t(FadeIn(note[0]), self._hl.animate.shift(UP * board.size), run_time=1.3)
            self.play_t(FadeIn(note[1]), self._hl.animate.shift(RIGHT * board.size), run_time=1.3)
            self.play_t(FadeIn(note[2], scale=1.05), run_time=1.0)
            self.play_t(Circumscribe(note[2], color=TXT), run_time=1.3)
        self.finish_scene()

    def bit_split_panel(self, rank_bits, file_bits, idx, note1, note2) -> VGroup:
        def bitrow(bits, color, tag):
            cells = VGroup()
            for b in bits:
                c = Square(side_length=0.5, fill_color=color if b == "1" else "#20262e",
                           fill_opacity=1.0, stroke_color="#2b3340", stroke_width=1.5)
                d = Text(b, font=MONO, font_size=26, color="#0d1117" if b == "1" else MUT)
                d.move_to(c.get_center())
                cells.add(VGroup(c, d))
            cells.arrange(RIGHT, buff=0.08)
            lab = Text(tag, font=MONO, font_size=24, color=color).next_to(cells, LEFT, buff=0.25)
            return VGroup(lab, cells)
        rank_g = bitrow(rank_bits, ACCENT, "rank")
        file_g = bitrow(file_bits, AMBER, "file")
        rows = VGroup(rank_g, file_g).arrange(RIGHT, buff=0.4)
        head = Text(f"e4  →  index {idx}", font=MONO, font_size=28, color=TXT)
        head.next_to(rows, UP, buff=0.35)
        n1 = Text(note1, font=MONO, font_size=23, color=ACCENT)
        n2 = Text(note2, font=MONO, font_size=23, color=AMBER)
        notes = VGroup(n2, n1).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        notes.next_to(rows, DOWN, buff=0.35)
        # return group whose [0]=rank bits, [1]=file bits for Indicate targets
        panel = VGroup(rank_g[1], file_g[1], head, notes, rank_g[0], file_g[0])
        return panel


# ============================================================================ Scene 3
class S3Bitboard(NarratedScene):
    SCENE_KEY = "S3_bitboard"

    def construct(self):
        self.add(header_group("a board is a set of squares"))
        board = ChessBoard(size=0.62, show_coords=True).shift(LEFT * 3.2 + DOWN * 0.15)
        self.add(board)

        with self.beat("s3a"):
            sset = [name_to_index(s) for s in ["c3", "e4", "e5", "g2", "b6", "d7"]]
            mask = 0
            for i in sset:
                mask |= (1 << i)
            eq = VGroup(
                Text("64 bits", font=SANS, font_size=38, color=ACCENT),
                Text("=", font=SANS, font_size=38, color=MUT),
                Text("64 squares", font=SANS, font_size=38, color=AMBER),
            ).arrange(RIGHT, buff=0.3)
            rule = Text("bit i = 1  ⇔  square i in the set", font=MONO, font_size=24, color=TXT)
            hexv = Text(f"0x{mask:016X}", font=MONO, font_size=26, color=AMBER)
            eqword = Text("one u64  =  this set of squares", font=SANS, font_size=26, color=MUT)
            col = VGroup(eq, rule, hexv, eqword).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
            self.place_right(col, cy=0.4)
            self.assert_in_frame(col, label="s3a-col")
            self.assert_no_overlap(board, col, label="board/col")
            self.play_t(FadeIn(eq), run_time=1.0)
            self.play_t(Write(rule), run_time=1.4)
            ov = board.overlay(sset, color=SET1, opacity=0.85, inset=0.05)
            self.play_t(FadeIn(ov, lag_ratio=0.15), run_time=1.6)
            self.play_t(FadeIn(hexv), run_time=1.0)
            self.play_t(FadeIn(eqword), run_time=1.0)
            self.play_t(Indicate(hexv, color=AMBER), run_time=1.2)
            self._ov = ov
            self._panelA = col
            self._board = board

        with self.beat("s3b"):
            self.play_t(FadeOut(self._panelA), run_time=0.6)
            code = code_block([
                "pub struct Bitboard(u64);",
            ], font_size=30, t2c={"struct": "#ff9e64", "Bitboard": ACCENT, "u64": AMBER,
                                  "pub": "#ff9e64"})
            note = VGroup(
                Text("a newtype: the compiler won't let you", font=SANS, font_size=24, color=TXT),
                Text("confuse a square-set with another u64", font=SANS, font_size=24, color=TXT),
                Text("expose behavior · hide representation", font=SANS, font_size=24, color=MUT),
            ).arrange(DOWN, buff=0.32, aligned_edge=LEFT)
            note[2].shift(DOWN * 0.15)
            grp3b = VGroup(code, note).arrange(DOWN, buff=0.6)
            self.place_right(grp3b, cy=0.4)
            self.assert_in_frame(VGroup(code, note), label="s3b")
            self.assert_no_overlap(board, code, label="board/code")
            self.play_t(Create(code), run_time=1.3)
            for line in note:
                self.play_t(FadeIn(line, shift=RIGHT * 0.1), run_time=0.9)
            self.play_t(Circumscribe(code, color=ACCENT), run_time=1.3)
            self._panelB = VGroup(code, note)

        with self.beat("s3c"):
            self.play_t(FadeOut(self._panelB), FadeOut(self._ov), run_time=0.6)
            items = VGroup(
                Text("white pawns", font=MONO, font_size=27, color=TXT),
                Text("black knights", font=MONO, font_size=27, color=TXT),
                Text("all white pieces", font=MONO, font_size=27, color=TXT),
                Text("every occupied square", font=MONO, font_size=27, color=TXT),
            ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            cap = Text("one board per question", font=SANS, font_size=30, color=AMBER)
            cap.next_to(items, DOWN, buff=0.6).align_to(items, LEFT)
            grp3c = VGroup(items, cap)
            self.place_right(grp3c, cy=0.2)
            self.assert_in_frame(VGroup(items, cap), label="s3c")
            # illustrate: white pawns on rank 2
            wp = [sq_index(f, 1) for f in range(8)]
            ov = board.overlay(wp, color=SET1, opacity=0.8, inset=0.06)
            for i, it in enumerate(items):
                self.play_t(FadeIn(it, shift=RIGHT * 0.12), run_time=0.8)
                if i == 0:
                    self.play_t(FadeIn(ov, lag_ratio=0.08), run_time=1.2)
            self.play_t(FadeIn(cap, scale=1.05), run_time=1.0)
            self.play_t(Indicate(cap, color=AMBER), run_time=1.1)
        self.finish_scene()


# ============================================================================ Scene 4
class S4SetOps(NarratedScene):
    SCENE_KEY = "S4_setops"

    def construct(self):
        self.add(header_group("set operations · one instruction each"))
        # two reference boards (A cyan, B amber) left; result board right
        A = [name_to_index(s) for s in ["c3", "c4", "c5", "d4", "e4"]]
        B = [name_to_index(s) for s in ["d4", "e4", "f4", "e5", "e3"]]

        ref = ChessBoard(size=0.40, show_coords=False).shift(LEFT * 3.5 + DOWN * 0.3)
        res = ChessBoard(size=0.40, show_coords=False).shift(RIGHT * 3.5 + DOWN * 0.3)
        self.add(ref, res)
        labL = Text("A (cyan)   B (amber)", font=MONO, font_size=24, color=MUT).next_to(ref, UP, buff=0.3)
        labR = Text("result", font=MONO, font_size=24, color=MUT).next_to(res, UP, buff=0.3)
        self.add(labL, labR)
        self.assert_no_overlap(ref, res, label="ref/res")

        ovA = ref.overlay(A, color=SET1, opacity=0.85, inset=0.05)
        ovB = ref.overlay(B, color=SET2, opacity=0.65, inset=0.14)  # inset so overlap shows both
        formula = Text("", font=MONO, font_size=30).move_to(DOWN * 3.2)

        with self.beat("s4a"):
            self.play_t(FadeIn(ovA, lag_ratio=0.1), run_time=1.3)
            self.play_t(FadeIn(ovB, lag_ratio=0.1), run_time=1.3)
            intro = Text("all 64 squares at once", font=SANS, font_size=30, color=TXT).move_to(UP * 2.2)
            self.assert_in_frame(intro, label="s4a-intro")
            self.play_t(FadeIn(intro), run_time=1.0)
            self.play_t(Indicate(intro, color=ACCENT), run_time=1.0)
            self._intro = intro

        self._res = res
        self._resov = None

        def show_result(bid, indices, text, color=GREEN):
            with self.beat(bid):
                if self._resov is not None:
                    self.play_t(FadeOut(self._resov), run_time=0.4)
                new = res.overlay(indices, color=color, opacity=0.85, inset=0.05)
                f = Text(text, font=MONO, font_size=30, color=color).move_to(DOWN * 3.15)
                self.assert_in_frame(f, label=bid + "-f")
                if getattr(self, "_formula", None) is not None:
                    self.play_t(ReplacementTransform(self._formula, f), run_time=0.7)
                else:
                    self.play_t(FadeIn(f), run_time=0.7)
                self._formula = f
                self.play_t(FadeIn(new, lag_ratio=0.08), run_time=1.1)
                self._resov = new
                return f

        self._formula = None
        union = sorted(set(A) | set(B))
        inter = sorted(set(A) & set(B))
        diff = sorted(set(A) - set(B))
        comp = [i for i in range(64) if i not in set(A)]

        show_result("s4b", union, "A ∪ B   =   a | b", color=GREEN)
        show_result("s4c", inter, "A ∩ B   =   a & b", color=GREEN)
        show_result("s4d", diff, "A \\ B   =   a & !b", color=GREEN)
        show_result("s4e", comp, "¬A   =   !a", color=GREEN)

        with self.beat("s4f"):
            self.play_t(FadeOut(self._resov), FadeOut(ovA), FadeOut(ovB),
                        FadeOut(self._formula), FadeOut(self._intro),
                        FadeOut(labL), FadeOut(labR), run_time=0.6)
            # pawns example: my pawns (cyan) & enemy attacks (amber) -> attacked pawns (green)
            newL = Text("my_pawns   enemy_attacks", font=MONO, font_size=22,
                        color=MUT).next_to(ref, UP, buff=0.3)
            newL[:8].set_color(SET1)
            newL[8:].set_color(SET2)
            newR = Text("attacked", font=MONO, font_size=22, color=GREEN).next_to(res, UP, buff=0.3)
            self.play_t(FadeIn(newL), FadeIn(newR), run_time=0.6)
            pawns = [name_to_index(s) for s in ["a2", "b2", "d4", "e5", "g3"]]
            atk = [name_to_index(s) for s in ["d4", "e5", "h6", "c6"]]
            hitp = sorted(set(pawns) & set(atk))
            pa = ref.overlay(pawns, color=SET1, opacity=0.85, inset=0.05)
            aa = ref.overlay(atk, color=SET2, opacity=0.6, inset=0.14)
            self.play_t(FadeIn(pa), FadeIn(aa), run_time=1.0)
            expr = code_block(['attacked = my_pawns & enemy_attacks'], font_size=27,
                              t2c={"my_pawns": SET1, "enemy_attacks": SET2, "attacked": GREEN})
            expr.move_to(DOWN * 3.05)
            self.assert_in_frame(expr, label="s4f-expr")
            self.play_t(FadeIn(expr), run_time=1.0)
            hit = res.overlay(hitp, color=GREEN, opacity=0.9, inset=0.05)
            arrow = Text("one instruction", font=SANS, font_size=26, color=GREEN).move_to(UP * 2.2)
            self.play_t(FadeIn(hit, lag_ratio=0.1), FadeIn(arrow), run_time=1.2)
            self.play_t(Circumscribe(expr, color=GREEN), run_time=1.3)
            self.play_t(Indicate(arrow, color=GREEN), run_time=1.1)
        self.finish_scene()


# ============================================================================ Scene 5
class S5Knight(NarratedScene):
    SCENE_KEY = "S5_knight"

    def construct(self):
        self.add(header_group("a picture"))
        board = ChessBoard(size=0.62).shift(LEFT * 3.2 + DOWN * 0.15)
        self.add(board)
        d4 = name_to_index("d4")
        atk = [name_to_index(s) for s in ["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"]]

        mask = 0
        for i in atk:
            mask |= 1 << i
        # whole right column, placed once so nothing runs off the bottom
        cnt = Text("8 squares", font=SANS, font_size=34, color=ACCENT)
        ascii_grid = self.printed_grid(atk)
        hexv = Text(f"0x{mask:016X}", font=MONO, font_size=24, color=AMBER)
        note = VGroup(
            Text("precomputed for all 64 squares", font=SANS, font_size=23, color=TXT),
            Text("just a bitboard — AND / OR it", font=SANS, font_size=23, color=MUT),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        rcol = VGroup(cnt, ascii_grid, hexv, note).arrange(DOWN, buff=0.4)
        self.place_right(rcol, cy=0.0)
        self.assert_in_frame(rcol, label="s5-rcol")
        self.assert_no_overlap(board, rcol, label="board/rcol")

        with self.beat("s5a"):
            # a subtle disc backing so the single knight glyph reads on the dark square
            disc = Dot(board.center(d4), radius=board.size * 0.42, color="#e9edf5")
            kn = piece("w", "N", size=board.size).move_to(board.center(d4))
            kn.set_fill("#101318", 1.0).set_stroke("#101318", 1.4)
            knight = VGroup(disc, kn)
            self.play_t(FadeIn(knight, scale=1.3), run_time=0.9)
            self.play_t(Circumscribe(board.squares[d4], color=TXT, buff=0), run_time=0.9)
            ov = board.overlay(atk, color=SET1, opacity=0.8, inset=0.06)
            self.play_t(FadeIn(ov, lag_ratio=0.12), run_time=1.6)
            self.play_t(FadeIn(cnt, scale=1.1), run_time=0.9)
            self.play_t(FadeIn(ascii_grid), run_time=1.2)
            self._kn, self._ov = kn, ov

        with self.beat("s5b"):
            self.play_t(FadeIn(hexv), run_time=1.0)
            self.play_t(Indicate(hexv, color=AMBER), run_time=1.0)
            for line in note:
                self.play_t(FadeIn(line, shift=RIGHT * 0.1), run_time=0.9)
            self.play_t(Circumscribe(self._ov, color=ACCENT), run_time=1.4)
        self.finish_scene()

    def printed_grid(self, indices) -> VGroup:
        s = set(indices)
        rows = []
        for rank in range(7, -1, -1):
            line = " ".join("X" if sq_index(f, rank) in s else "." for f in range(8))
            rows.append(line)
        g = Text("\n".join(rows), font=MONO, font_size=24, color=TXT, line_spacing=0.6,
                 t2c={"X": ACCENT})
        return g


# ============================================================================ Scene 6
class S6Tricks(NarratedScene):
    SCENE_KEY = "S6_tricks"

    def construct(self):
        self.add(header_group("three bit tricks"))
        board = ChessBoard(size=0.60).shift(LEFT * 3.3 + DOWN * 0.15)
        self.add(board)
        pawns = [name_to_index(s) for s in ["a2", "c2", "d2", "f2", "h2"]]

        with self.beat("s6a"):
            title = Text("1 · population count", font=SANS, font_size=30, color=ACCENT)
            code = code_block(["my_pawns.count()  →  5"], font_size=26,
                              t2c={"count": AMBER, "5": GREEN})
            sub = Text("one POPCNT instruction", font=MONO, font_size=23, color=MUT)
            col = VGroup(title, code, sub).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
            self.place_right(col, top=2.5)
            self.assert_in_frame(col, label="s6a")
            self.assert_no_overlap(board, col, label="board/col")
            self.play_t(FadeIn(title, shift=DOWN * 0.15), run_time=0.8)
            ov = board.overlay(pawns, color=SET1, opacity=0.85, inset=0.06)
            self.play_t(FadeIn(ov, lag_ratio=0.1), run_time=1.2)
            self.play_t(Create(code), run_time=1.2)
            self.play_t(FadeIn(sub), run_time=0.8)
            self.play_t(Indicate(code, color=GREEN), run_time=1.2)
            self._ov, self._rcol = ov, col

        with self.beat("s6b"):
            self.play_t(FadeOut(self._rcol), run_time=0.5)
            title = Text("2 · least significant bit", font=SANS, font_size=30, color=ACCENT)
            code = code_block(["lowest = x.trailing_zeros()"], font_size=25,
                              t2c={"trailing_zeros": AMBER})
            col = VGroup(title, code).arrange(DOWN, buff=0.8, aligned_edge=LEFT)
            self.place_right(col, top=2.5)
            self.assert_in_frame(col, label="s6b")
            self.play_t(FadeIn(title, shift=DOWN * 0.15), run_time=0.8)
            lsb = min(pawns)
            ring = SurroundingRectangle(board.squares[lsb], color=AMBER, buff=0.0, stroke_width=4)
            self.play_t(Create(ring), run_time=1.0)
            self.play_t(Create(code), run_time=1.2)
            self.play_t(Indicate(ring, color=AMBER), run_time=1.1)
            self._ring, self._rcol = ring, col

        with self.beat("s6c"):
            self.play_t(FadeOut(self._rcol), run_time=0.5)
            title = Text("3 · pop the lowest bit", font=SANS, font_size=30, color=ACCENT)
            code = code_block([
                "fn pop_lsb(&mut self) -> Square {",
                "    let sq = self.lsb();",
                "    self.0 &= self.0 - 1;  // clear it",
                "    sq",
                "}",
            ], font_size=21, t2c={"pop_lsb": ACCENT, "fn": "#ff9e64",
                                  "Square": GREEN, "let": "#ff9e64"})
            col = VGroup(title, code).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
            self.place_right(col, top=2.5)
            self.assert_in_frame(col, label="s6c")
            self.assert_no_overlap(board, code, label="board/code")
            self.play_t(FadeIn(title, shift=DOWN * 0.1), run_time=0.8)
            self.play_t(Create(code), run_time=1.8)
            self.play_t(Circumscribe(code.body, color=ACCENT), run_time=1.4)
            self._rcol = col

        with self.beat("s6d"):
            self.play_t(FadeOut(self._rcol), FadeOut(self._ring), run_time=0.5)
            title = Text("3 · pop the lowest bit", font=SANS, font_size=30, color=ACCENT)
            sub = Text("the bitboard's for-loop", font=SANS, font_size=26, color=MUT)
            counter = Text("popped: 0", font=MONO, font_size=30, color=GREEN)
            col = VGroup(title, sub, counter).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
            self.place_right(col, top=2.5)
            self.assert_in_frame(col, label="s6d")
            self.play_t(FadeIn(title), FadeIn(sub), FadeIn(counter), run_time=0.8)
            order = sorted(pawns)
            per = (self.beat_durs["s6d"] - 0.8 - 1.0) / len(order)
            per = max(0.5, min(per, 1.3))
            for n, idx in enumerate(order, start=1):
                cell = self._ov.submobjects[pawns.index(idx)]
                newc = Text(f"popped: {n}", font=MONO, font_size=30, color=GREEN)
                newc.move_to(counter)
                self.play_t(FadeOut(cell, scale=0.3), Transform(counter, newc), run_time=per)
            self._rcol = col

        with self.beat("s6e"):
            self.play_t(FadeOut(self._rcol), run_time=0.5)
            title = Text("directional shifts", font=SANS, font_size=30, color=ACCENT)
            code = code_block(["push = pawns << 8"], font_size=28,
                              t2c={"pawns": SET1, "<<": AMBER})
            note = Text("whole army advances at once", font=SANS, font_size=24, color=MUT)
            mask = Text("(a mask stops h-file wrap-around)", font=MONO, font_size=20, color=RED)
            col = VGroup(title, code, note, mask).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
            self.place_right(col, top=2.5)
            self.assert_in_frame(col, label="s6e")
            self.assert_no_overlap(board, code, label="board/code")
            self.play_t(FadeIn(title), run_time=0.7)
            row2 = [sq_index(f, 1) for f in range(8)]
            ov2 = board.overlay(row2, color=SET1, opacity=0.85, inset=0.06)
            self.play_t(FadeIn(ov2, lag_ratio=0.05), run_time=1.0)
            self.play_t(Create(code), run_time=1.1)
            self.play_t(ov2.animate.shift(UP * board.size), run_time=1.4)
            self.play_t(FadeIn(note), run_time=0.8)
            self.play_t(ov2.animate.shift(UP * board.size), run_time=1.2)
            self.play_t(FadeIn(mask), run_time=0.9)
            self.play_t(Indicate(mask, color=RED), run_time=1.1)
        self.finish_scene()


# ============================================================================ Scene 7
class S7Mailbox(NarratedScene):
    SCENE_KEY = "S7_mailbox"

    def construct(self):
        self.add(header_group("the other half of the hybrid"))
        board = ChessBoard(size=0.58).shift(LEFT * 3.4 + DOWN * 0.15)
        self.add(board)
        d5 = name_to_index("d5")

        with self.beat("s7a"):
            q = Text("what is on d5?", font=SANS, font_size=34, color=TXT).shift(RIGHT * 3.3 + UP * 2.2)
            ring = SurroundingRectangle(board.squares[d5], color=AMBER, buff=0, stroke_width=4)
            self.play_t(FadeIn(q), Create(ring), run_time=1.1)
            names = ["wP?", "wN?", "wB?", "wR?", "wQ?", "wK?",
                     "bP?", "bN?", "bB?", "bR?", "bQ?", "bK?"]
            col = VGroup(*[Text(n, font=MONO, font_size=22, color=MUT) for n in names])
            col.arrange_in_grid(rows=6, cols=2, buff=(0.55, 0.22))
            col.next_to(q, DOWN, buff=0.5).align_to(q, LEFT)
            self.assert_in_frame(col, label="s7a-col")
            self.assert_no_overlap(board, col, label="board/col")
            self.play_t(FadeIn(col, lag_ratio=0.06), run_time=1.6)
            bad = Text("up to 12 boards to test", font=SANS, font_size=27, color=RED)
            bad.next_to(col, DOWN, buff=0.45).align_to(col, LEFT)
            self.assert_in_frame(bad, label="s7a-bad")
            self.play_t(FadeIn(bad), run_time=0.9)
            self.play_t(Indicate(bad, color=RED), run_time=1.1)
            self._q, self._ring, self._col, self._bad = q, ring, col, bad

        with self.beat("s7b"):
            self.play_t(FadeOut(self._col), FadeOut(self._bad), run_time=0.5)
            # mailbox strip: 8 slots around d5 as a mini array
            slots = VGroup()
            labels = ["c5", "d5", "e5", "f5"]
            fills = [None, ("b", "N"), None, None]
            for i, name in enumerate(labels):
                cell = Square(side_length=0.62, fill_color="#20262e", fill_opacity=1,
                              stroke_color="#2b3340", stroke_width=1.5)
                lab = Text(name, font=MONO, font_size=18, color=MUT).next_to(cell, DOWN, buff=0.08)
                slots.add(VGroup(cell, lab))
            slots.arrange(RIGHT, buff=0.12)
            slots.shift(RIGHT * 3.3 + UP * 1.4)
            kn = piece("b", "N", size=0.6).move_to(slots[1][0].get_center())
            code = code_block(["mailbox: [Option<Piece>; 64]"], font_size=24,
                              t2c={"mailbox": ACCENT, "Option": "#ff9e64", "Piece": GREEN, "64": AMBER})
            code.next_to(slots, DOWN, buff=0.7)
            one = Text("one array lookup · O(1)", font=SANS, font_size=27, color=GREEN)
            one.next_to(code, DOWN, buff=0.5)
            self.assert_in_frame(VGroup(slots, code, one), label="s7b")
            self.assert_no_overlap(board, slots, label="board/slots")
            self.play_t(FadeIn(slots, lag_ratio=0.1), run_time=1.2)
            self.play_t(FadeIn(kn, scale=1.2), run_time=0.9)
            self.play_t(Create(code), run_time=1.2)
            # draw arrow from board d5 to the mailbox d5 slot
            arr = Arrow(board.squares[d5].get_right(), slots[1][0].get_left(),
                        color=AMBER, stroke_width=3, buff=0.15, max_tip_length_to_length_ratio=0.08)
            self.play_t(Create(arr), run_time=1.0)
            self.play_t(FadeIn(one, scale=1.05), run_time=0.9)
            self.play_t(Indicate(one, color=GREEN), run_time=1.0)
            self._slots, self._kn7, self._code7b, self._one, self._arr = slots, kn, code, one, arr

        with self.beat("s7c"):
            self.play_t(FadeOut(self._slots), FadeOut(self._kn7), FadeOut(self._code7b),
                        FadeOut(self._one), FadeOut(self._arr), FadeOut(self._q),
                        FadeOut(self._ring), run_time=0.6)
            func = code_block(["put_piece()", "remove_piece()"], font_size=26,
                              t2c={"put_piece": ACCENT, "remove_piece": ACCENT})
            func.shift(RIGHT * 3.4 + UP * 1.6)
            t1 = Text("bitboards", font=MONO, font_size=24, color=SET1)
            t2 = Text("mailbox", font=MONO, font_size=24, color=GREEN)
            targets = VGroup(t1, t2).arrange(RIGHT, buff=1.2).next_to(func, DOWN, buff=1.3)
            a1 = Arrow(func.get_bottom(), t1.get_top(), color=MUT, stroke_width=3, buff=0.2)
            a2 = Arrow(func.get_bottom(), t2.get_top(), color=MUT, stroke_width=3, buff=0.2)
            cap = Text("one code path · never drift", font=SANS, font_size=26, color=AMBER)
            cap.next_to(targets, DOWN, buff=0.6)
            self.assert_in_frame(VGroup(func, targets, cap), label="s7c")
            self.play_t(Create(func), run_time=1.0)
            self.play_t(Create(a1), Create(a2), run_time=1.0)
            self.play_t(FadeIn(t1, shift=DOWN * 0.1), FadeIn(t2, shift=DOWN * 0.1), run_time=0.9)
            self.play_t(FadeIn(cap), run_time=0.9)
            self.play_t(Indicate(cap, color=AMBER), run_time=1.1)
            self._c7c = VGroup(func, targets, a1, a2, cap)

        with self.beat("s7d"):
            self.play_t(FadeOut(self._c7c), run_time=0.5)
            left = Text("bitboards", font=SANS, font_size=28, color=SET1)
            lsub = Text("fast set algebra", font=SANS, font_size=22, color=MUT).next_to(left, DOWN, buff=0.2)
            right = Text("mailbox", font=SANS, font_size=28, color=GREEN)
            rsub = Text("O(1) point lookup", font=SANS, font_size=22, color=MUT).next_to(right, DOWN, buff=0.2)
            lg = VGroup(left, lsub)
            rg = VGroup(right, rsub)
            both = VGroup(lg, rg).arrange(RIGHT, buff=1.4).shift(RIGHT * 3.3 + UP * 0.9)
            plus = Text("+", font=SANS, font_size=40, color=TXT).move_to(
                (lg.get_right() + rg.get_left()) / 2)
            hyb = Text("the hybrid — you get both", font=SANS, font_size=30, color=AMBER)
            hyb.next_to(both, DOWN, buff=0.8)
            self.assert_in_frame(VGroup(both, plus, hyb), label="s7d")
            self.play_t(FadeIn(lg, shift=RIGHT * 0.1), run_time=0.9)
            self.play_t(FadeIn(plus), run_time=0.5)
            self.play_t(FadeIn(rg, shift=LEFT * 0.1), run_time=0.9)
            self.play_t(Write(hyb), run_time=1.2)
            self.play_t(Circumscribe(hyb, color=AMBER), run_time=1.3)
        self.finish_scene()


# ============================================================================ Scene 8
class S8Layout(NarratedScene):
    SCENE_KEY = "S8_layout"

    def construct(self):
        self.add(header_group("the full board"))
        # Build the struct as per-line Text mobjects so a single line can be boxed
        # reliably (multi-line Text glyph indexing skips newlines — unreliable).
        src = [
            "pub struct Board {",
            "pieces: [[Bitboard; 6]; 2],",
            "color_occupancy: [Bitboard; 2],",
            "all_occupancy: Bitboard,",
            "mailbox: [Option<Piece>; 64],",
            "side_to_move: Color,",
            "}",
        ]
        t2c = {"struct": "#ff9e64", "pub": "#ff9e64", "Board": ACCENT,
               "Bitboard": SET1, "Option": "#ff9e64", "Piece": GREEN, "Color": AMBER}
        line_mobs = VGroup(*[Text(s, font=MONO, font_size=21, color=CODE_TXT, t2c=t2c) for s in src])
        line_mobs.arrange(DOWN, buff=0.20, aligned_edge=LEFT)
        # Pango drops leading whitespace, so indent the field lines explicitly.
        for _i in range(1, 6):
            line_mobs[_i].shift(RIGHT * 0.34)
        panel = RoundedRectangle(corner_radius=0.14, fill_color=CODE_BG, fill_opacity=1.0,
                                 stroke_color="#2b3340", stroke_width=1.5)
        panel.stretch_to_fit_width(line_mobs.width + 0.55)
        panel.stretch_to_fit_height(line_mobs.height + 0.5)
        panel.move_to(line_mobs.get_center())
        code = VGroup(panel, line_mobs)
        if code.width > 6.6:
            code.scale(6.6 / code.width)
        code.move_to(np.array([-3.5, 0.2, 0.0]))
        self.add(code)
        self.assert_in_frame(code, label="struct")

        line_of = {"pieces:": 1, "color_occupancy": 2, "all_occupancy": 3,
                   "mailbox:": 4, "side_to_move:": 5}

        def box_line(substr, color):
            return SurroundingRectangle(line_mobs[line_of[substr]], color=color,
                                        buff=0.06, stroke_width=3)

        with self.beat("s8a"):
            title = Text("12 piece bitboards", font=SANS, font_size=32, color=SET1)
            title.shift(RIGHT * 3.5 + UP * 2.0)
            sub = Text("pieces[color][piece_type]", font=MONO, font_size=23, color=MUT)
            sub.next_to(title, DOWN, buff=0.4)
            self.assert_in_frame(VGroup(title, sub), label="s8a")
            self.play_t(FadeIn(title, shift=DOWN * 0.1), run_time=0.9)
            box = box_line("pieces:", SET1)
            self.play_t(Create(box), run_time=1.0)
            self.play_t(FadeIn(sub), run_time=0.9)
            self.play_t(Indicate(sub, color=SET1), run_time=1.1)
            self._box, self._t8 = box, VGroup(title, sub)

        with self.beat("s8b"):
            self.play_t(FadeOut(self._t8), run_time=0.4)
            title = Text("two aggregates, for speed", font=SANS, font_size=30, color=AMBER)
            title.shift(RIGHT * 3.5 + UP * 2.0)
            b1 = box_line("color_occupancy", AMBER)
            b2 = box_line("all_occupancy", AMBER)
            notes = VGroup(
                Text("union of one side's 6 boards", font=SANS, font_size=23, color=MUT),
                Text("every occupied square", font=SANS, font_size=23, color=MUT),
                Text("computed once, saves work per query", font=SANS, font_size=23, color=TXT),
            ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(title, DOWN, buff=0.6)
            self.assert_in_frame(VGroup(title, notes), label="s8b")
            self.play_t(FadeIn(title, shift=DOWN * 0.1), run_time=0.8)
            self.play_t(ReplacementTransform(self._box, b1), run_time=0.8)
            self.play_t(FadeIn(notes[0]), run_time=0.8)
            self.play_t(Transform(b1, b2), run_time=0.8)
            self.play_t(FadeIn(notes[1]), run_time=0.8)
            self.play_t(FadeIn(notes[2]), run_time=0.9)
            self.play_t(Indicate(notes[2], color=AMBER), run_time=1.1)
            self._box, self._t8 = b1, VGroup(title, notes)

        with self.beat("s8c"):
            self.play_t(FadeOut(self._t8), FadeOut(self._box), run_time=0.5)
            title = Text("mailbox · side · encoding", font=SANS, font_size=30, color=GREEN)
            title.shift(RIGHT * 3.5 + UP * 2.1)
            b1 = box_line("mailbox:", GREEN)
            b2 = box_line("side_to_move:", AMBER)
            order = Text("P  N  B  R  Q  K", font=MONO, font_size=30, color=TXT)
            order.next_to(title, DOWN, buff=0.7)
            osub = Text("least → most valuable · iterate in order",
                        font=SANS, font_size=22, color=MUT).next_to(order, DOWN, buff=0.35)
            self.assert_in_frame(VGroup(title, order, osub), label="s8c")
            self.play_t(FadeIn(title, shift=DOWN * 0.1), run_time=0.8)
            self.play_t(Create(b1), run_time=0.7)
            self.play_t(Transform(b1, b2), run_time=0.7)
            self.play_t(Write(order), run_time=1.2)
            self.play_t(FadeIn(osub), run_time=0.8)
            self.play_t(Indicate(order, color=GREEN), run_time=1.1)
            self._box, self._t8 = b1, VGroup(title, order, osub)

        with self.beat("s8d"):
            self.play_t(FadeOut(self._t8), FadeOut(self._box), run_time=0.5)
            title = Text("written in this vocabulary", font=SANS, font_size=28, color=TXT)
            items = VGroup(
                Text("move generation · AND / OR / shift", font=MONO, font_size=21, color=SET1),
                Text("hashing · key[piece][square]", font=MONO, font_size=21, color=AMBER),
                Text("evaluation · loop + pop squares", font=MONO, font_size=21, color=GREEN),
            ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            punch = VGroup(
                Text("the substrate, not a detail —", font=SANS, font_size=23, color=MUT),
                Text("it shapes every later algorithm", font=SANS, font_size=23, color=AMBER),
            ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            col8d = VGroup(title, items, punch).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
            self.place_right(col8d, top=2.5)
            self.assert_in_frame(col8d, label="s8d")
            self.assert_no_overlap(code, items, label="code/items")
            self.play_t(FadeIn(title, shift=DOWN * 0.1), run_time=0.8)
            for it in items:
                self.play_t(FadeIn(it, shift=RIGHT * 0.1), run_time=0.9)
            self.play_t(FadeIn(punch[0]), run_time=0.9)
            self.play_t(FadeIn(punch[1]), run_time=0.9)
            self.play_t(Circumscribe(punch, color=AMBER), run_time=1.4)
        self.finish_scene()


# ============================================================================ Scene 9
class S9Recap(NarratedScene):
    SCENE_KEY = "S9_recap"

    def construct(self):
        self.add(header_group("what you now know"))

        with self.beat("s9a"):
            rows = VGroup(
                self.recap_row("0–63", "squares numbered so file & rank are a mask and a shift", ACCENT),
                self.recap_row("u64", "a set of squares — set ops become single instructions", SET2),
                self.recap_row("O(1)", "a mailbox answers 'what's on this square' instantly", GREEN),
            ).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
            rows.move_to(ORIGIN + UP * 0.35)
            self.assert_in_frame(rows, label="s9a-rows")
            for r in rows:
                self.assert_min_height(r[1], 0.24, "recap-text")
            for i, r in enumerate(rows):
                self.play_t(FadeIn(r, shift=RIGHT * 0.15), run_time=1.1)
                self.play_t(Indicate(r[0], color=r.tag_color), run_time=0.9)
            self._rows = rows

        with self.beat("s9b"):
            self.play_t(self._rows.animate.shift(UP * 1.35).scale(0.82), run_time=0.9)
            nxt = Text("Next — file 03", font=SANS, font_size=40, color=TXT)
            sub = Text("move generation & magic bitboards", font=SANS, font_size=32, color=ACCENT)
            grp = VGroup(nxt, sub).arrange(DOWN, buff=0.35).shift(DOWN * 2.35)
            self.assert_in_frame(grp, label="s9b")
            self.play_t(Write(nxt), run_time=1.1)
            self.play_t(FadeIn(sub, scale=1.05), run_time=1.0)
            self.play_t(Circumscribe(sub, color=ACCENT), run_time=1.4)
        self.finish_scene()

    def recap_row(self, tag, text, color) -> VGroup:
        chip = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.7, width=1.5,
                             fill_color=color, fill_opacity=0.18, stroke_color=color, stroke_width=2),
            Text(tag, font=MONO, font_size=26, color=color),
        )
        chip[1].move_to(chip[0].get_center())
        body = Text(text, font=SANS, font_size=26, color=TXT).next_to(chip, RIGHT, buff=0.4)
        # keep the row within frame width
        row = VGroup(chip, body)
        if row.width > FRAME_W - 1.0:
            row.scale((FRAME_W - 1.0) / row.width)
        row.tag_color = color
        return row
