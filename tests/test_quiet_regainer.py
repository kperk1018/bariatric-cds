"""Tests for src/quiet_regainer.py — the high-alert sub-phenotype flag."""
import unittest

from src.quiet_regainer import flag_quiet_regainer, PEAK_MIN, DROP_MIN


def _traj(y1, y2, y4):
    return {"TBWL": {1: {"point": y1}, 2: {"point": y2}, 4: {"point": y4}}}


class TestQuietRegainer(unittest.TestCase):
    def test_flags_strong_early_then_regain(self):
        r = flag_quiet_regainer(_traj(39, 31, 24))   # peak 39, drop 15
        self.assertTrue(r["detected"])
        self.assertEqual(r["peak"], 39.0)
        self.assertEqual(r["drop"], 15.0)
        self.assertIn("quiet regainer", r["reason"].lower())

    def test_strong_but_sustained_not_flagged(self):
        r = flag_quiet_regainer(_traj(40, 40, 37))   # peak 40, drop only 3
        self.assertFalse(r["detected"])
        self.assertIn("holds it", r["reason"])

    def test_weak_responder_not_flagged(self):
        r = flag_quiet_regainer(_traj(25, 24, 14))   # peak 25 < 34, big drop but not strong early
        self.assertFalse(r["detected"])

    def test_boundary_is_inclusive(self):
        r = flag_quiet_regainer(_traj(PEAK_MIN, 0, PEAK_MIN - DROP_MIN))
        self.assertTrue(r["detected"])

    def test_missing_year_not_assessable(self):
        r = flag_quiet_regainer({"TBWL": {1: {"point": 39}, 2: {"point": 31}, 4: {"point": None}}})
        self.assertFalse(r["detected"])
        self.assertIsNone(r["drop"])
        self.assertIn("not assessable", r["reason"].lower())


if __name__ == "__main__":
    unittest.main()
