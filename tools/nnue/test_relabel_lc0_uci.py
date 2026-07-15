#!/usr/bin/env python3
"""Fast deterministic tests for relabel_lc0_uci.py."""

from __future__ import annotations

import unittest

import chess
import numpy as np

import relabel_lc0_uci as relabel


class RelabelTests(unittest.TestCase):
    def test_affine_schedule_is_full_permutation_for_composite_size(self) -> None:
        rows = 100
        multiplier, offset = relabel.affine_parameters(rows, seed=20260714)
        self.assertEqual(np.gcd(multiplier, rows), 1)
        indices = relabel.source_indices(0, rows, rows, multiplier, offset)
        self.assertEqual(len(np.unique(indices)), rows)
        self.assertEqual(set(indices.tolist()), set(range(rows)))

    def test_shard_slices_equal_one_full_schedule(self) -> None:
        rows = 97
        multiplier, offset = relabel.affine_parameters(rows, seed=91)
        full = relabel.source_indices(0, rows, rows, multiplier, offset)
        pieces = [
            relabel.source_indices(start, min(start + 13, rows), rows, multiplier, offset)
            for start in range(0, rows, 13)
        ]
        np.testing.assert_array_equal(np.concatenate(pieces), full)

    def test_move_encoding_retains_from_to_and_promotion(self) -> None:
        move = chess.Move.from_uci("a7a8q")
        encoded = relabel.encode_move(move)
        self.assertEqual(encoded & 0x3F, chess.A7)
        self.assertEqual((encoded >> 6) & 0x3F, chess.A8)
        self.assertEqual(encoded >> 12, chess.QUEEN)
        self.assertEqual(relabel.encode_move(None), 0xFFFF)


if __name__ == "__main__":
    unittest.main()
