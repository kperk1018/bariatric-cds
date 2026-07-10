"""Trajectory phenotypes — converged onto Ioanna's 1A clustering method (S7).

1A pipeline (Supplementary Table S7): cluster on preop features + model-PREDICTED
TBWL trajectories, reduce with UMAP to 2D, choose k by max silhouette (k=2..10),
KMeans on the embedding, then order clusters by ascending mean Preop_TBWL.

1B's convergence (this module) matches that ALGORITHM exactly. Two deliberate,
documented differences from her exact run, because her intermediate prediction CSV
was not provided:
  * Trajectory input = 1B's own `predict_trajectory` (preop-honest, out-of-fold
    calibrated) rather than her in-sample RandomForest fitted values.
  * Preop panel = the 15 BASELINE_FEATURES rather than her larger preop set.
Both offline fit and online assignment use the SAME trajectory source, so a new
patient is placed consistently with the cohort.

ONLINE ASSIGNMENT DOES NOT USE UMAP. Two reasons:
  1. PRIVACY (hard rule): a fitted `umap.UMAP` retains `_raw_data` — the full scaled
     training matrix — which, together with the persisted scaler, is invertible back
     to row-level patient values. Persisting it put patient data in git. Never again.
  2. PORTABILITY: `UMAP.transform` JIT-compiles via numba, which fails on Python 3.14
     (Streamlit Cloud), crashing the deployed app.
Instead, after clustering we fit a multinomial LogisticRegression on the SCALED
feature space to reproduce the final cluster labels. It recovers them essentially
exactly (100% train, 99.1% 5-fold CV agreement) and persists only a k x n_features
coefficient matrix — an aggregate, no patient rows. UMAP is still used to FIT
(matching 1A), it is simply never persisted or called at inference.

`fit_phenotypes` persists one frozen bundle (imputer, scaler, label assigner, cluster
order, per-cluster actual-TBWL trajectory means, silhouette curve). `assign_phenotype`
transforms a new patient through that frozen bundle.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

from src.config import ARTIFACTS, BASELINE_FEATURES, TBWL_BY_YEAR, SEED

CLUSTER_TRAJ_YEARS = [1, 2, 3, 4, 5]          # non-red TBWL years used for clustering
CAT_COLS = ["Sex", "Race", "Surgery_Type"]
PREOP_NUM = [c for c in BASELINE_FEATURES if c not in CAT_COLS]
K_RANGE = range(2, 11)                          # k swept; argmax silhouette selected
MANUSCRIPT_K = 5                                # 1A reports 5 clusters (Ioanna, S7)
K_TIE = 0.02                                    # silhouette gap counted as "tied"
UMAP_KW = dict(n_components=2, random_state=SEED, n_neighbors=8, min_dist=0.15)
KMEANS_KW = dict(n_init=10, random_state=SEED)
_MODEL_PATH = ARTIFACTS / "phenotype_kmeans.joblib"


def _traj_col(y: int) -> str:
    return f"pred_TBWL_{y}yr"


def _assemble_features(preop_df: pd.DataFrame, traj_df: pd.DataFrame) -> pd.DataFrame:
    """Build the clustering matrix: OHE(drop_first) preop + predicted TBWL trajectory."""
    enc = pd.get_dummies(preop_df[CAT_COLS].astype(str), columns=CAT_COLS,
                         drop_first=True, dtype=int)
    num = preop_df[PREOP_NUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    out = pd.concat([num, enc.reset_index(drop=True), traj_df.reset_index(drop=True)], axis=1)
    return out


def _predicted_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Preop-predict TBWL yrs 1-5 for every row via predict_trajectory."""
    from src.predict import predict_trajectory  # local import avoids import cycle
    rows = []
    for _, r in df.iterrows():
        # raw rows can carry stray-string blanks in numeric fields; coerce so the
        # model imputer (not astype) handles the gaps. Categoricals stay strings.
        patient = {}
        for c in BASELINE_FEATURES:
            v = r.get(c)
            patient[c] = v if c in CAT_COLS else pd.to_numeric(v, errors="coerce")
        t = predict_trajectory(patient)
        rows.append({_traj_col(y): t["TBWL"][y]["point"] for y in CLUSTER_TRAJ_YEARS})
    return pd.DataFrame(rows)


def select_k(X: np.ndarray) -> tuple[int, dict[int, float]]:
    """Return (argmax-silhouette k, {k: silhouette}) over K_RANGE for embedding X."""
    sil: dict[int, float] = {}
    for k in K_RANGE:
        if k >= len(X):
            break
        labels = KMeans(n_clusters=k, **KMEANS_KW).fit_predict(X)
        sil[k] = float(silhouette_score(X, labels))
    return max(sil, key=sil.get), sil


def choose_k(sil: dict[int, float]) -> int:
    """Silhouette-selected k, with a tie-break toward the 1A manuscript k.

    The silhouette peak and MANUSCRIPT_K are frequently within noise of each other
    (on the RF trajectories, k=4 and k=5 differ by ~0.0002). When MANUSCRIPT_K is
    within K_TIE of the argmax we report MANUSCRIPT_K, so 1B matches 1A's 5 clusters
    without hard-coding — Ioanna confirmed 5 is the intended count and that fixing it
    is acceptable. When the data clearly prefers a different k (> K_TIE better), that
    k wins and the divergence is surfaced.
    """
    argmax_k = max(sil, key=sil.get)
    if MANUSCRIPT_K in sil and (sil[argmax_k] - sil[MANUSCRIPT_K]) <= K_TIE:
        return MANUSCRIPT_K
    return argmax_k


def fit_phenotypes(df: pd.DataFrame, save: bool = True) -> dict:
    """Fit the 1A-aligned phenotype pipeline on the cohort and persist it."""
    df = df.reset_index(drop=True)
    traj_df = _predicted_trajectories(df)
    X = _assemble_features(df, traj_df)
    columns = list(X.columns)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = imputer.fit_transform(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xi)

    # UMAP is used to FIT only (matches 1A). It is never persisted — a fitted
    # reducer retains the raw training matrix (see module docstring).
    import umap  # local import: keeps numba/umap off the inference + app path
    reducer = umap.UMAP(**UMAP_KW)
    Xu = reducer.fit_transform(Xs)

    _, silhouette_by_k = select_k(Xu)
    chosen_k = choose_k(silhouette_by_k)
    kmeans = KMeans(n_clusters=chosen_k, **KMEANS_KW).fit(Xu)
    raw_labels = kmeans.labels_

    # order clusters by ascending mean Preop_TBWL (matches 1A)
    preop_tbwl = pd.to_numeric(df["Preop_TBWL"], errors="coerce").values
    raw_means = {c: np.nanmean(preop_tbwl[raw_labels == c]) for c in range(chosen_k)}
    order = sorted(raw_means, key=lambda c: (np.nan_to_num(raw_means[c], nan=np.inf)))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in raw_labels])

    # per-cluster ACTUAL TBWL trajectory means (for the app deep-dive; no recompute)
    cluster_actual_traj: dict = {}
    for c in range(chosen_k):
        per_year = {}
        for y in range(1, 7):
            col = TBWL_BY_YEAR[y]
            if col in df.columns:
                vals = pd.to_numeric(df.loc[labels == c, col], errors="coerce").dropna()
                if len(vals) >= 3:
                    per_year[y] = {"mean": float(vals.mean()), "sd": float(vals.std()),
                                   "n": int(len(vals))}
        cluster_actual_traj[c] = {"per_year": per_year, "n": int((labels == c).sum())}

    # Online assigner: reproduce the final labels from the SCALED space, so inference
    # needs neither UMAP nor any stored patient row. Persists coefficients only.
    assigner = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xs, labels)
    assigner_agreement = float((assigner.predict(Xs) == labels).mean())

    bundle = {
        "imputer": imputer, "scaler": scaler, "assigner": assigner,
        "remap": remap, "columns": columns, "k": chosen_k,
        "assigner_agreement": assigner_agreement,
        "silhouette_by_k": silhouette_by_k, "n_train": len(df),
        "traj_years": CLUSTER_TRAJ_YEARS, "cluster_actual_traj": cluster_actual_traj,
        "method": "1A-aligned: preop+predicted-TBWL -> UMAP -> silhouette-k -> KMeans, "
                  "ordered by ascending Preop_TBWL; online assignment via "
                  "LogisticRegression on the scaled features (no UMAP, no stored rows)",
    }
    if save:
        ARTIFACTS.mkdir(exist_ok=True)
        joblib.dump(bundle, _MODEL_PATH)
    return bundle


def assign_phenotype(patient: dict, traj: dict | None = None,
                     bundle: dict | None = None) -> dict:
    """Assign one patient to a frozen phenotype.

    Args:
        patient: preop feature dict (BASELINE_FEATURES).
        traj: optional precomputed predict_trajectory(patient) output; computed if None.
        bundle: optional preloaded bundle.
    """
    if bundle is None:
        bundle = joblib.load(_MODEL_PATH)
    if traj is None:
        from src.predict import predict_trajectory
        traj = predict_trajectory(patient)

    traj_row = {_traj_col(y): traj["TBWL"][y]["point"] for y in bundle["traj_years"]}
    preop_df = pd.DataFrame([{c: patient.get(c) for c in BASELINE_FEATURES}])
    X = _assemble_features(preop_df, pd.DataFrame([traj_row]))
    X = X.reindex(columns=bundle["columns"], fill_value=0)

    Xi = bundle["imputer"].transform(X)
    Xs = bundle["scaler"].transform(Xi)
    # assigner predicts the FINAL (Preop_TBWL-ordered) label directly — no remap,
    # no UMAP transform (see module docstring: privacy + numba/py3.14 portability).
    label = int(bundle["assigner"].predict(Xs)[0])
    return {
        "phenotype": label,
        "k": bundle["k"],
        "n_train": bundle["n_train"],
        "note": "Provisional — 1A-aligned clustering (UMAP+silhouette), pending mentor confirmation.",
    }
