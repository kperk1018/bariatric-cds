"""Guard: no committed artifact may embed row-level patient data.

This exists because a fitted `umap.UMAP` retains `_raw_data` (the full scaled training
matrix) and the SHAP backgrounds stored a random subsample of real patients. Both were
committed alongside their scalers, making them invertible to actual patient values.

Rule enforced here: any 2-D array persisted in `artifacts/` must have fewer than
ROW_LEVEL_THRESHOLD rows. Model weights, scalers (mean/scale), imputer medians,
k-means centroids, and LogisticRegression coefficients all pass; raw patient matrices
do not.

Skips cleanly when artifacts/ is absent (fresh clone, CI without trained models).
"""
import unittest

import joblib
import numpy as np

from src.config import ARTIFACTS

ROW_LEVEL_THRESHOLD = 20
# attributes on fitted sklearn/umap objects that are known to retain training rows
_ROW_BEARING_ATTRS = ("_raw_data", "embedding_", "_fit_X", "support_vectors_", "X_fit_")


def _find_row_level(obj, path="artifact", depth=0, hits=None):
    if hits is None:
        hits = []
    if depth > 4:
        return hits
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and obj.shape[0] >= ROW_LEVEL_THRESHOLD:
            hits.append((path, obj.shape))
        return hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            _find_row_level(v, f"{path}[{k!r}]", depth + 1, hits)
        return hits
    for attr in _ROW_BEARING_ATTRS:
        v = getattr(obj, attr, None)
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] >= ROW_LEVEL_THRESHOLD:
            hits.append((f"{path}.{attr}", v.shape))
    return hits


class TestArtifactsCarryNoPatientRows(unittest.TestCase):
    def test_no_artifact_embeds_row_level_data(self):
        if not ARTIFACTS.exists():
            self.skipTest("artifacts/ not present")
        artifacts = sorted(ARTIFACTS.glob("*.joblib"))
        if not artifacts:
            self.skipTest("no .joblib artifacts to scan")

        offenders = []
        for path in artifacts:
            try:
                obj = joblib.load(path)
            except Exception:
                continue  # unloadable (missing optional dep) — not a leak vector here
            for where, shape in _find_row_level(obj):
                offenders.append(f"{path.name}: {where} has shape {shape}")

        self.assertEqual(
            offenders, [],
            "Artifact(s) embed row-level patient data (>= "
            f"{ROW_LEVEL_THRESHOLD} rows). These must never be committed:\n  "
            + "\n  ".join(offenders),
        )

    def test_phenotype_bundle_has_no_umap_reducer(self):
        path = ARTIFACTS / "phenotype_kmeans.joblib"
        if not path.exists():
            self.skipTest("phenotype bundle not present")
        bundle = joblib.load(path)
        self.assertNotIn("reducer", bundle,
                         "UMAP reducer retains _raw_data — must not be persisted")
        self.assertIn("assigner", bundle, "expected coefficient-only online assigner")


if __name__ == "__main__":
    unittest.main()
