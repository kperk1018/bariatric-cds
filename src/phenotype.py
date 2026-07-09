"""Trajectory phenotypes via k-means (ported from the blind chat exploration).

Recipe: standardized TBWL% at years 1-3, complete-case, relabeled by ascending
year-3 TBWL. The number of clusters **k is derived**, not assumed: we sweep
k=2..10 and pick the k that maximizes silhouette (matches Ioanna's 1A convention;
she was explicit that k must not be hard-coded). `assign_phenotype` loads the one
frozen fitted model so new patients are placed consistently.

Note: the clustering *feature set / population* (actual TBWL yrs 1-3, complete-case)
is still 1B's own and is NOT yet converged onto her predicted-trajectory/UMAP
approach — that reconciliation is parked pending her intermediate CSVs.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import ARTIFACTS

TRAJ_COLS = ["1yr_Postop_TBWL", "2yr_Postop_TBWL", "3yr_Postop_TBWL"]
K_RANGE = range(2, 11)  # candidate cluster counts; k chosen by max silhouette
SEED = 42
_MODEL_PATH = ARTIFACTS / "phenotype_kmeans.joblib"


def select_k(X: np.ndarray) -> tuple[int, dict[int, float]]:
    """Return (argmax-silhouette k, {k: silhouette}) over K_RANGE for scaled X."""
    sil: dict[int, float] = {}
    for k in K_RANGE:
        if k >= len(X):
            break
        labels = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(X)
        sil[k] = float(silhouette_score(X, labels))
    chosen = max(sil, key=sil.get)
    return chosen, sil


def fit_phenotypes(df: pd.DataFrame, save: bool = True) -> dict:
    """Fit scaler + k-means on complete-case trajectories; persist the frozen model.

    k is selected automatically by silhouette over K_RANGE — never hard-coded.
    """
    sub = df.dropna(subset=TRAJ_COLS).copy()
    scaler = StandardScaler().fit(sub[TRAJ_COLS].values)
    X = scaler.transform(sub[TRAJ_COLS].values)

    chosen_k, silhouette_by_k = select_k(X)
    km = KMeans(n_clusters=chosen_k, n_init=50, random_state=SEED).fit(X)

    # stable relabel: 0..k-1 by ascending final-year mean TBWL
    sub["_c"] = km.labels_
    order = sub.groupby("_c")[TRAJ_COLS[-1]].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}

    bundle = {"scaler": scaler, "kmeans": km, "remap": remap,
              "traj_cols": TRAJ_COLS, "k": chosen_k,
              "silhouette_by_k": silhouette_by_k, "n_train": len(sub)}
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
            "k": bundle["k"],
            "n_train": bundle["n_train"],
            "note": "Provisional — clustering recipe pending mentor confirmation."}
