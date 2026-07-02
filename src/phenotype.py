"""Trajectory phenotypes via k-means (ported from the blind chat exploration).

RECIPE NOT FINAL — see CLAUDE.md. Current default: standardized TBWL% at years 1-3,
complete-case (N=149), k=5, relabeled 0..4 by ascending year-3 TBWL. Silhouette
prefers k=2; k=5 matches the manuscript pending confirmation with Dr. Raftopoulos.

Swap the recipe in ONE place: TRAJ_COLS and K below. `assign_phenotype` loads the
frozen fitted model so new patients are placed consistently.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.config import ARTIFACTS

TRAJ_COLS = ["1yr_Postop_TBWL", "2yr_Postop_TBWL", "3yr_Postop_TBWL"]
K = 5
SEED = 42
_MODEL_PATH = ARTIFACTS / "phenotype_kmeans.joblib"


def fit_phenotypes(df: pd.DataFrame, save: bool = True) -> dict:
    """Fit scaler + k-means on complete-case trajectories; persist the frozen model."""
    sub = df.dropna(subset=TRAJ_COLS).copy()
    scaler = StandardScaler().fit(sub[TRAJ_COLS].values)
    X = scaler.transform(sub[TRAJ_COLS].values)
    km = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(X)

    # stable relabel: 0..K-1 by ascending final-year mean TBWL
    sub["_c"] = km.labels_
    order = sub.groupby("_c")[TRAJ_COLS[-1]].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}

    bundle = {"scaler": scaler, "kmeans": km, "remap": remap,
              "traj_cols": TRAJ_COLS, "k": K, "n_train": len(sub)}
    if save:
        ARTIFACTS.mkdir(exist_ok=True)
        joblib.dump(bundle, _MODEL_PATH)
    return bundle


def assign_phenotype(patient: dict, bundle: dict | None = None) -> dict:
    """Assign one patient (dict with TRAJ_COLS) to a frozen phenotype."""
    if bundle is None:
        bundle = joblib.load(_MODEL_PATH)
    x = np.array([[patient[c] for c in bundle["traj_cols"]]], dtype=float)
    raw = int(bundle["kmeans"].predict(bundle["scaler"].transform(x))[0])
    return {"phenotype": bundle["remap"][raw],
            "note": "Provisional — clustering recipe pending mentor confirmation."}
