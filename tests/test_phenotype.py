"""Tests for src/phenotype.py — k is derived by silhouette, never hard-coded."""
import unittest
import numpy as np
import pandas as pd

from src.phenotype import select_k, fit_phenotypes, assign_phenotype, TRAJ_COLS


class TestSelectK(unittest.TestCase):
    def test_picks_two_for_two_clear_blobs(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 0.2, size=(60, 3))
        b = rng.normal(6, 0.2, size=(60, 3))
        X = np.vstack([a, b])
        k, sil = select_k(X)
        self.assertEqual(k, 2)
        self.assertEqual(k, max(sil, key=sil.get))  # k is the argmax, not assumed
        self.assertTrue(set(sil).issubset(set(range(2, 11))))

    def test_argmax_is_reported_k(self):
        rng = np.random.default_rng(0)
        # three separated blobs -> silhouette should favor 3
        X = np.vstack([rng.normal(c, 0.1, size=(40, 3)) for c in (0, 5, 10)])
        k, sil = select_k(X)
        self.assertEqual(k, max(sil, key=sil.get))
        self.assertEqual(k, 3)


class TestFitAssign(unittest.TestCase):
    def _blob_df(self):
        rng = np.random.default_rng(42)
        lo = rng.normal(10, 1.0, size=(50, 3))
        hi = rng.normal(40, 1.0, size=(50, 3))
        X = np.vstack([lo, hi])
        return pd.DataFrame(X, columns=TRAJ_COLS)

    def test_fit_derives_k_and_stores_metadata(self):
        bundle = fit_phenotypes(self._blob_df(), save=False)
        self.assertEqual(bundle["k"], 2)
        self.assertIn("silhouette_by_k", bundle)
        self.assertEqual(bundle["n_train"], 100)
        # clusters relabeled 0..k-1 by ascending yr-3 TBWL
        self.assertEqual(set(bundle["remap"].values()), {0, 1})

    def test_assign_returns_k_and_places_patient(self):
        bundle = fit_phenotypes(self._blob_df(), save=False)
        low = assign_phenotype(dict(zip(TRAJ_COLS, [10, 10, 10])), bundle=bundle)
        high = assign_phenotype(dict(zip(TRAJ_COLS, [40, 40, 40])), bundle=bundle)
        self.assertEqual(low["k"], 2)
        self.assertEqual(low["phenotype"], 0)   # lowest-loss cluster
        self.assertEqual(high["phenotype"], 1)  # highest-loss cluster


if __name__ == "__main__":
    unittest.main()
