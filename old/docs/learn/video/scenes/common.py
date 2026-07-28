"""Shared visual system + timing engine for the ARCHON learning-video pilot (doc 02).

Design goals
------------
* One coherent look: dark background, muted checkered board, two bright accents
  (cyan = set A, amber = set B, green = their intersection).
* Audio-driven timing: every beat is sized from the narration manifest so the
  render is frame-exact and the narration never overruns its visuals.
* No-eyes guardrails: assert_in_frame / assert_min_height / assert_no_overlap
  fail the render loudly if a layout breaks, so bad composition can't slip past
  an agent that cannot watch the video.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Square, Text, RoundedRectangle, Scene, config,
    FadeIn, np as mnp,  # noqa: F401
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifests" / "02-timing.json"

# ----------------------------------------------------------------------------- palette
BG        = "#0d1117"   # near-black (GitHub dark) — calm, high contrast
LIGHT_SQ  = "#8b98ad"   # muted slate, light square
DARK_SQ   = "#3d4757"   # muted slate, dark square
SET1      = "#3ec6ff"   # cyan  — set A / primary highlight
SET2      = "#ffc857"   # amber — set B
INTER     = "#4ade80"   # green — intersection (A ∩ B)
DIFF      = "#3ec6ff"   # difference reuses cyan
TXT       = "#e6edf3"   # near-white body text
MUT       = "#9aa5b1"   # muted caption grey
ACCENT    = "#3ec6ff"
AMBER     = "#ffc857"
GREEN     = "#4ade80"
RED       = "#ff6b6b"
CODE_TXT  = "#c9d1d9"
CODE_BG   = "#161b22"
PANEL_STROKE = "#2b3340"
SQ_STROKE = "#12161d"

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

config.background_color = BG

FRAME_W = config.frame_width       # 14.222 for 16:9 @ height 8
FRAME_H = config.frame_height      # 8.0

# unicode chess glyphs (solid), indexed [color][piecetype], type order P N B R Q K
GLYPH = {
    "w": {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔"},
    "b": {"P": "♟", "N": "♞", "B": "♝", "R": "♜", "Q": "♛", "K": "♚"},
}

FILES = "abcdefgh"


def sq_index(file: int, rank: int) -> int:
    return rank * 8 + file


def name_to_index(name: str) -> int:
    """'e4' -> 28 (LERF)."""
    return sq_index(FILES.index(name[0]), int(name[1]) - 1)


# --------------------------------------------------------------------------- chessboard
class ChessBoard(VGroup):
    """8x8 board as a VGroup of Squares, indexed 0..63 in little-endian rank-file.

    Square 0 (a1) is bottom-left; +y is up; a1 is a dark square (real chess).
    """

    def __init__(self, size: float = 0.62, show_coords: bool = True,
                 light: str = LIGHT_SQ, dark: str = DARK_SQ):
        super().__init__()
        self.size = size
        self.squares: dict[int, Square] = {}
        cells = VGroup()
        for rank in range(8):
            for file in range(8):
                idx = sq_index(file, rank)
                color = dark if (file + rank) % 2 == 0 else light
                s = Square(side_length=size, fill_color=color, fill_opacity=1.0,
                           stroke_color=SQ_STROKE, stroke_width=1.0)
                s.move_to(np.array([(file - 3.5) * size, (rank - 3.5) * size, 0.0]))
                self.squares[idx] = s
                cells.add(s)
        self.add(cells)
        self.coords = VGroup()
        if show_coords:
            fs = size * 34
            for file in range(8):
                t = Text(FILES[file], font=MONO, font_size=fs, color=MUT)
                t.next_to(self.squares[sq_index(file, 0)], 0.6 * np.array([0, -1, 0]), buff=0.12)
                self.coords.add(t)
            for rank in range(8):
                t = Text(str(rank + 1), font=MONO, font_size=fs, color=MUT)
                t.next_to(self.squares[sq_index(0, rank)], np.array([-1, 0, 0]), buff=0.12)
                self.coords.add(t)
            self.add(self.coords)

    def center(self, idx: int) -> np.ndarray:
        return self.squares[idx].get_center()

    def overlay(self, indices, color=SET1, opacity=0.9, inset: float = 0.0) -> VGroup:
        """A group of bright squares sitting on top of the given cells (for lighting up)."""
        g = VGroup()
        for i in indices:
            s = Square(side_length=self.size - inset, fill_color=color, fill_opacity=opacity,
                       stroke_color=color, stroke_width=1.5)
            s.move_to(self.center(i))
            g.add(s)
        return g

    def dot_bits(self, indices) -> VGroup:
        """Small filled dots marking set squares — the '.'/'X' printed-grid feel."""
        from manim import Dot
        g = VGroup()
        for i in indices:
            g.add(Dot(self.center(i), radius=self.size * 0.16, color=TXT))
        return g


def piece(sym_color: str, sym_type: str, size: float = 0.62) -> Text:
    glyph = GLYPH[sym_color][sym_type]
    col = "#f4f6fb" if sym_color == "w" else "#0c0f14"
    stroke = "#0c0f14" if sym_color == "w" else "#c9d1d9"
    t = Text(glyph, font=SANS, font_size=size * 60, color=col,
             stroke_color=stroke, stroke_width=1.2)
    return t


def code_block(lines, font_size: float = 26, t2c=None, width_pad: float = 0.5,
               title: str | None = None) -> VGroup:
    """A monospace code panel with a rounded dark background."""
    body = Text("\n".join(lines), font=MONO, font_size=font_size, color=CODE_TXT,
                line_spacing=0.85, t2c=t2c or {})
    bg = RoundedRectangle(corner_radius=0.14, fill_color=CODE_BG, fill_opacity=1.0,
                          stroke_color=PANEL_STROKE, stroke_width=1.5)
    bg.stretch_to_fit_width(body.width + width_pad)
    bg.stretch_to_fit_height(body.height + 0.42)
    bg.move_to(body.get_center())
    grp = VGroup(bg, body)
    grp.body = body
    grp.bg = bg
    return grp


# --------------------------------------------------------------------------- timing base
class NarratedScene(Scene):
    """Base scene whose beats are sized from the narration timing manifest.

    Usage inside construct():
        with self.beat("s2a"):
            self.play_t(FadeIn(x), run_time=0.8)
            # helper waits out the remaining audio budget automatically
        ...
        self.finish_scene()   # add the trailing inter-scene breath
    """

    SCENE_KEY: str | None = None
    # Layout-only mode: skip animations/waits so every bounding-box assert fires
    # in ~1s (a fast, no-eyes geometry gate) instead of a full multi-minute render.
    LAYOUT_ONLY = os.environ.get("ARCHON_LAYOUT_ONLY") == "1"

    def setup(self):
        super().setup()
        man = json.loads(MANIFEST.read_text())
        spec = next(s for s in man["scenes"] if s["key"] == self.SCENE_KEY)
        self.scene_spec = spec
        self.beat_durs = {b["id"]: b["beat_dur"] for b in spec["beats"]}
        self.scene_tail = spec["tail"]
        self._anim_used = 0.0

    # -- timed play/wait: accumulate consumed time so beat() can fill the remainder
    def play_t(self, *anims, run_time: float = 1.0, **kw):
        self._anim_used += run_time
        if self.LAYOUT_ONLY:
            for a in anims:                       # place targets so later next_to() works
                m = getattr(a, "mobject", None)
                if m is not None:
                    self.add(m)
            return
        self.play(*anims, run_time=run_time, **kw)

    def wait_t(self, t: float):
        self._anim_used += t
        if not self.LAYOUT_ONLY:
            self.wait(t)

    @contextmanager
    def beat(self, bid: str):
        self._anim_used = 0.0
        budget = self.beat_durs[bid]
        yield budget
        rem = budget - self._anim_used
        if rem < -1e-6:
            raise ValueError(
                f"[{self.SCENE_KEY}/{bid}] animations overran narration budget "
                f"{budget:.3f}s by {-rem:.3f}s — shorten run_times")
        if rem > 1e-6 and not self.LAYOUT_ONLY:
            self.wait(rem)

    def finish_scene(self):
        if self.LAYOUT_ONLY:
            self.wait(0.1)                        # emit >=1 frame so manim is happy
            return
        if self.scene_tail > 1e-6:
            self.wait(self.scene_tail)

    def place_right(self, mob, cx: float = 3.35, cy: float = 0.0,
                    max_w: float = 6.7, max_h: float = 6.6, top: float | None = None):
        """Scale a right-panel mobject to fit the right region and place it there.

        top=None centers the group at (cx, cy); a numeric `top` pins the group's
        upper edge at y=top (keeps a 'title on top' column consistent across beats).
        """
        if mob.width > max_w:
            mob.scale(max_w / mob.width)
        if mob.height > max_h:
            mob.scale(max_h / mob.height)
        if top is not None:
            mob.move_to(np.array([cx, 0.0, 0.0]))
            mob.align_to(np.array([cx, top, 0.0]), np.array([0, 1, 0]))
        else:
            mob.move_to(np.array([cx, cy, 0.0]))
        return mob

    def fit_width(self, mob, max_w: float = 6.7):
        """Scale a mobject down in place if it exceeds max_w (keeps its center)."""
        if mob.width > max_w:
            mob.scale(max_w / mob.width)
        return mob

    # ------------------------------------------------------------- layout guardrails
    def assert_in_frame(self, m, margin: float = 0.18, label: str = ""):
        xr, yr = FRAME_W / 2 - margin, FRAME_H / 2 - margin
        l, r = m.get_left()[0], m.get_right()[0]
        b, t = m.get_bottom()[1], m.get_top()[1]
        if l < -xr - 1e-6 or r > xr + 1e-6 or b < -yr - 1e-6 or t > yr + 1e-6:
            raise ValueError(
                f"[{self.SCENE_KEY}] '{label}' out of frame: "
                f"x[{l:.2f},{r:.2f}] y[{b:.2f},{t:.2f}] vs x±{xr:.2f} y±{yr:.2f}")

    def assert_min_height(self, m, min_h: float = 0.26, label: str = ""):
        if m.height < min_h - 1e-6:
            raise ValueError(
                f"[{self.SCENE_KEY}] '{label}' text too small: height {m.height:.3f} < {min_h}")

    def assert_no_overlap(self, a, b, gap: float = 0.05, label: str = ""):
        ax0, ax1 = a.get_left()[0], a.get_right()[0]
        ay0, ay1 = a.get_bottom()[1], a.get_top()[1]
        bx0, bx1 = b.get_left()[0], b.get_right()[0]
        by0, by1 = b.get_bottom()[1], b.get_top()[1]
        ov_x = min(ax1, bx1) - max(ax0, bx0)
        ov_y = min(ay1, by1) - max(ay0, by0)
        if ov_x > gap and ov_y > gap:
            raise ValueError(
                f"[{self.SCENE_KEY}] overlap '{label}': x_ov={ov_x:.2f} y_ov={ov_y:.2f} (gap {gap})")

    def check_layout(self, *mobs, margin: float = 0.18):
        for m in mobs:
            self.assert_in_frame(m, margin=margin, label=getattr(m, "_dbg", type(m).__name__))
