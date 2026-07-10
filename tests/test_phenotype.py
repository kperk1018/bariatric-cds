"""Tests for src/phenotype.py — 1A-aligned clustering (UMAP + silhouette-derived k).

Uses a tiny synthetic cohort and a stub predict_trajectory so no real CSV or
trained models are needed. k is derived by silhouette (with a documented tie-break
to the 1A manuscript k=5 when it's within noise of the peak), never blindly forced.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.phenotype as phe


def _synthetic_cohort(n=90, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    # two separated preop groups (low vs high Preop_TBWL / BMI)
    for i in range(n):
        hi = i % 2 == 0
        rows.append({
            "Age": rng.normal(45, 5), "Sex": "Female" if i % 3 else "Male",
            "Race": "White", "Height": rng.normal(64, 3),
            "Initial_BMI": rng.normal(48 if hi else 40, 1.5),
            "Initial_Weight": rng.normal(260 if hi else 210, 8),
            "Initial_BMR": rng.normal(1900, 80), "Initial_VF": rng.normal(14, 2),
            "Initial_FATpct": rng.normal(45, 3), "Initial_FATMASS": rng.normal(115, 10),
            "Initial_FFM": rng.normal(137, 8), "Time_to_Surgery": rng.normal(135, 20),
            "Surgery_Type": "Sleeve" if i % 2 else "Bypass",
            "Preop_BMI": rng.normal(38, 2),
            "Preop_TBWL": rng.normal(14 if hi else 8, 1.0),
        })
    return pd.DataFrame(rows)


def _stub_predict(patient):
    """Trajectory correlated with Preop_TBWL so clusters are separable."""
    base = float(patient["Preop_TBWL"]) * 2.0
    return {"TBWL": {y: {"point": base - y} for y in range(1, 7)}}


class TestPhenotypeClustering(unittest.TestCase):
    def test_fit_derives_k_and_persists_bundle(self):
        df = _synthetic_cohort()
        with patch("src.predict.predict_trajectory", side_effect=_stub_predict):
            bundle = phe.fit_phenotypes(df, save=False)
        self.assertIn(bundle["k"], set(phe.K_RANGE))          # k came from the sweep
        # k follows choose_k: silhouette argmax, tie-broken toward the manuscript k
        self.assertEqual(bundle["k"], phe.choose_k(bundle["silhouette_by_k"]))
        self.assertEqual(bundle["n_train"], len(df))
        # clusters ordered by ascending mean Preop_TBWL (label 0 = lowest)
        self.assertEqual(set(bundle["remap"].values()), set(range(bundle["k"])))
        # UMAP must NOT be persisted (retains _raw_data); online assignment uses a
        # coefficient-only classifier on the scaled features instead.
        self.assertNotIn("reducer", bundle)
        self.assertIn("assigner", bundle)
        self.assertGreaterEqual(bundle["assigner_agreement"], 0.90)
        self.assertIn("cluster_actual_traj", bundle)

    def test_assign_is_deterministic_and_returns_k(self):
        df = _synthetic_cohort()
        with patch("src.predict.predict_trajectory", side_effect=_stub_predict):
            bundle = phe.fit_phenotypes(df, save=False)
            patient = df.iloc[0].to_dict()
            traj = _stub_predict(patient)
            a = phe.assign_phenotype(patient, traj=traj, bundle=bundle)
            b = phe.assign_phenotype(patient, traj=traj, bundle=bundle)
        self.assertEqual(a["phenotype"], b["phenotype"])
        self.assertEqual(a["k"], bundle["k"])
        self.assertIn(a["phenotype"], range(bundle["k"]))

    def test_select_k_argmax_on_embedding(self):
        rng = np.random.default_rng(0)
        emb = np.vstack([rng.normal(c, 0.15, size=(40, 2)) for c in (0, 5, 10)])
        k, sil = phe.select_k(emb)
        self.assertEqual(k, max(sil, key=sil.get))
        self.assertEqual(k, 3)


if __name__ == "__main__":
    unittest.main()
