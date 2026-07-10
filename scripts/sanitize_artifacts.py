"""One-shot: strip row-level patient data out of existing artifacts, without refitting.

Why: `*_background.joblib` shipped a random subsample of REAL scaled training rows for
shap.KernelExplainer. Combined with the committed `*_scaler.joblib`, those invert back
to actual patient values — row-level data in git. This replaces each background with
k-means centroids over the same rows (aggregates, ~10 patients per centroid).

The estimators themselves are untouched, so predictions are unchanged; only the SHAP
*reference distribution* becomes an aggregate (base_value shifts negligibly).

The phenotype bundle is fixed separately by re-running `scripts/run_clustering.py`,
which no longer persists the UMAP reducer.

Idempotent: already-aggregated backgrounds are left alone.

Run:
    PYTHONPATH=. python scripts/sanitize_artifacts.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ARTIFACTS, SEED

PER_CENTROID = 10          # each centroid averages ~this many patients
ROW_LEVEL_THRESHOLD = 20   # >= this many rows is considered row-level data


def main() -> None:
    files = sorted(ARTIFACTS.glob("*_background.joblib"))
    if not files:
        print("No *_background.joblib found — nothing to sanitize.")
        return
    for path in files:
        bg = joblib.load(path)
        if not isinstance(bg, np.ndarray) or bg.ndim != 2:
            print(f"  {path.name}: not a 2-D array, skipped")
            continue
        n = bg.shape[0]
        if n < ROW_LEVEL_THRESHOLD:
            print(f"  {path.name}: {n} rows already aggregate, skipped")
            continue
        k = max(2, min(n - 1, n // PER_CENTROID))
        centers = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(bg).cluster_centers_
        joblib.dump(centers, path)
        print(f"  {path.name}: {n} raw rows -> {k} centroids "
              f"(~{n // k} patients averaged per centroid)")
    print("\nDone. Re-run scripts/run_clustering.py to rebuild the phenotype bundle.")


if __name__ == "__main__":
    main()
