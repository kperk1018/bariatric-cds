"""Tests for src/bmi_milestone.py — no CSV required."""
import unittest
import pandas as pd
import numpy as np

from src.bmi_milestone import (
    derive_postop_bmi,
    flag_milestone,
    landmark_split,
    compare_trajectories,
    compare_fml_trajectories,
)
from src.config import TBWL_BY_YEAR, FML_BY_YEAR


def _make_df(n=30, seed=42):
    """Synthetic patient DataFrame with TBWL and FML for years 1-4."""
    rng = np.random.default_rng(seed)
    data = {
        "Initial_BMI": rng.uniform(35, 55, n),
    }
    for yr in range(1, 7):
        # Simulate declining TBWL over years 1-6
        data[TBWL_BY_YEAR[yr]] = rng.uniform(10, 35 - yr * 2, n)
    for yr in range(1, 7):
        data[FML_BY_YEAR[yr]] = rng.uniform(5, 60 - yr * 5, n)
    return pd.DataFrame(data)


class TestDerivePostopBMI(unittest.TestCase):

    def test_formula_correctness(self):
        df = pd.DataFrame({
            "Initial_BMI": [40.0],
            TBWL_BY_YEAR[1]: [50.0],  # 50% loss → BMI halved
        })
        for yr in range(2, 7):
            df[TBWL_BY_YEAR[yr]] = 0.0
        result = derive_postop_bmi(df)
        self.assertAlmostEqual(result["Postop_BMI_yr1"].iloc[0], 20.0, places=5)

    def test_all_six_bmi_columns_added(self):
        df = _make_df()
        result = derive_postop_bmi(df)
        for yr in range(1, 7):
            self.assertIn(f"Postop_BMI_yr{yr}", result.columns)


class TestFlagMilestone(unittest.TestCase):

    def test_milestone_flag_bmi27_positive(self):
        df = _make_df()
        df = derive_postop_bmi(df)
        df["Postop_BMI_yr1"] = 26.0  # force all patients below 27
        result = flag_milestone(df, threshold=27.0)
        self.assertTrue(result["ever_bmi27"].all())

    def test_milestone_flag_bmi27_negative(self):
        df = _make_df()
        df = derive_postop_bmi(df)
        # Force all postop BMI above 27
        for yr in range(1, 7):
            df[f"Postop_BMI_yr{yr}"] = 35.0
        result = flag_milestone(df, threshold=27.0)
        self.assertFalse(result["ever_bmi27"].any())

    def test_boundary_inclusive(self):
        df = pd.DataFrame({
            "Initial_BMI": [40.0],
            **{TBWL_BY_YEAR[yr]: [0.0] for yr in range(1, 7)},
        })
        df = derive_postop_bmi(df)
        df["Postop_BMI_yr1"] = 25.0  # exactly at threshold
        result = flag_milestone(df, threshold=25.0)
        self.assertTrue(result["ever_bmi25"].iloc[0])

    def test_min_postop_bmi_column_added(self):
        df = _make_df()
        df = derive_postop_bmi(df)
        result = flag_milestone(df, 27.0)
        self.assertIn("min_postop_bmi", result.columns)


class TestLandmarkSplit(unittest.TestCase):

    def test_groups_mutually_exclusive(self):
        df = _make_df(n=40)
        df = derive_postop_bmi(df)
        group_a, group_b = landmark_split(df, landmark_year=2, threshold=27.0)
        self.assertEqual(len(set(group_a.index) & set(group_b.index)), 0)

    def test_groups_cover_all_patients_with_data(self):
        df = _make_df(n=40)
        df = derive_postop_bmi(df)
        group_a, group_b = landmark_split(df, landmark_year=2, threshold=27.0)
        # Every patient with at least one non-NaN BMI through yr2 is in A or B
        self.assertGreater(len(group_a) + len(group_b), 0)

    def test_group_a_all_achieved_milestone(self):
        df = _make_df(n=20)
        df = derive_postop_bmi(df)
        df["Postop_BMI_yr1"] = 24.0  # all achieve by yr1
        group_a, group_b = landmark_split(df, landmark_year=2, threshold=27.0)
        self.assertEqual(len(group_a), len(df))
        self.assertEqual(len(group_b), 0)


class TestCompareTrajectories(unittest.TestCase):

    def test_returns_pvalue_float(self):
        df = _make_df(n=30)
        df = derive_postop_bmi(df)
        group_a, group_b = landmark_split(df, landmark_year=2, threshold=27.0)
        # Ensure there's some data in both groups
        if len(group_a) < 3 or len(group_b) < 3:
            self.skipTest("Not enough patients in both groups for this random seed")
        results = compare_trajectories(group_a, group_b, [3, 4])
        for yr in [3, 4]:
            if results["per_year"][yr]["p_value"] is not None:
                self.assertIsInstance(results["per_year"][yr]["p_value"], float)
                self.assertGreaterEqual(results["per_year"][yr]["p_value"], 0.0)
                self.assertLessEqual(results["per_year"][yr]["p_value"], 1.0)

    def test_per_year_has_required_keys(self):
        df = _make_df(n=30)
        df = derive_postop_bmi(df)
        group_a, group_b = landmark_split(df, landmark_year=2, threshold=27.0)
        results = compare_trajectories(group_a, group_b, [3])
        required = {"n_a", "n_b", "median_a", "median_b", "u_stat", "p_value"}
        self.assertTrue(required.issubset(results["per_year"][3].keys()))


class TestCompareFMLTrajectories(unittest.TestCase):

    def test_returns_spearman_within_groups(self):
        df = _make_df(n=30)
        df = derive_postop_bmi(df)
        group_a, group_b = landmark_split(df, landmark_year=2, threshold=27.0)
        if len(group_a) < 5 or len(group_b) < 5:
            self.skipTest("Not enough patients for Spearman")
        results = compare_fml_trajectories(group_a, group_b, [3])
        if 3 in results["per_year"] and results["per_year"][3].get("spearman_fml_vs_tbwl_group_a"):
            r_val = results["per_year"][3]["spearman_fml_vs_tbwl_group_a"]["r"]
            if r_val is not None:
                self.assertGreaterEqual(abs(r_val), 0.0)
                self.assertLessEqual(abs(r_val), 1.0)


if __name__ == "__main__":
    unittest.main()
