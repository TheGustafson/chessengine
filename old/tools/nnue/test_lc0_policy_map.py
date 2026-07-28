import unittest

import lc0_policy_map


class Lc0PolicyMapTests(unittest.TestCase):
    def test_canonical_shape_and_boundaries(self):
        moves = lc0_policy_map.canonical_moves()
        self.assertEqual(len(moves), 1858)
        self.assertEqual(len(set(moves)), 1858)
        self.assertEqual(moves[:4], ["a1b1", "a1c1", "a1d1", "a1e1"])
        self.assertEqual(moves[97], "e1a1")
        self.assertEqual(moves[99], "e1c1")
        self.assertEqual(moves[1792:1798], [
            "a7a8q", "a7a8r", "a7a8b", "a7b8q", "a7b8r", "a7b8b",
        ])
        self.assertEqual(moves[-3:], ["h7h8q", "h7h8r", "h7h8b"])

    def test_promotions_collapse_into_from_to_space(self):
        self.assertEqual(lc0_policy_map.from_to_index("a7a8q"), 48 * 64 + 56)
        self.assertEqual(
            lc0_policy_map.from_to_index("a7a8q"),
            lc0_policy_map.from_to_index("a7a8b"),
        )


if __name__ == "__main__":
    unittest.main()
