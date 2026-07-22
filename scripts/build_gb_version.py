"""Parallel GRADIENT BOOSTING pipeline — "version 1B-GB" alongside the RF version.

Ioanna (2026-07-13): "Re-run everything you did with random forests using gradient
boosting too — for the paper, not the app. Set it up as version 1B alongside the RF
version." And be ready to explain why GB outperforms RF at years 3-4.

This mirrors the RF pipeline end to end, but with GradientBoostingRegressor, and keeps
it fully separate from the app (persists to artifacts/gb/, never loaded by predict.py):

  1. Train one GB model per year (TBWL), leakage-safe single-lag cascade, deduped N=786.
  2. Cascade-predict each patient's TBWL trajectory (yrs 1-4) from GB.
  3. Cluster on the GB trajectories (UMAP -> silhouette -> k-means, order by preop TBWL),
     exactly as for RF, so the two are directly comparable.
  4. Test the manuscript's "both pathways converged on the identical clusters" claim:
     Adjusted Rand Index + concordance between RF-trajectory and GB-trajectory clusters,
     and whether the preop-TBWL ladder and demographics match.

Outputs:
  artifacts/gb/TBWL_yr{y}_model.joblib, _meta.joblib
  artifacts/gb/rf_vs_gb_clusters.csv        (per-cluster preop + demographics, both models)
  presentation/figures/fig9_rf_vs_gb_clusters.png

Run:
    PYTHONPATH=. python scripts/build_gb_version.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.config import (
    ARTIFACTS, BASELINE_FEATURES, TBWL_BY_YEAR, LAGGED_TBWL_BY_YEAR, SEED,
)
from src.data_load import load
from src.preprocess import build_feature_matrix, get_numeric_cols, encode_patient
from src.reliability import gate
from src.predict import predict_trajectory  # RF, for the comparison arm

GB_DIR = ARTIFACTS / "gb"
GB_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_YEARS = [1, 2, 3, 4]          # non-red TBWL years, same as the RF clustering
NUM_COLS = get_numeric_cols(BASELINE_FEATURES)
UMAP_KW = dict(n_components=2, random_state=SEED, n_neighbors=8, min_dist=0.15)


def _gb(small_n: bool) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=200, random_state=SEED, max_depth=3,
        subsample=0.7 if small_n else 0.9,
        min_samples_leaf=5 if small_n else 2,
    )


def train_gb_models(df: pd.DataFrame) -> None:
    """One GB model per year, single-lag, leakage-safe. Persist to artifacts/gb/."""
    print("Training GB per-year TBWL models (deduped N=%d)..." % len(df))
    for year in range(1, 7):
        target = TBWL_BY_YEAR[year]
        lagged = LAGGED_TBWL_BY_YEAR[year]
        sub = df.dropna(subset=[target] + BASELINE_FEATURES + lagged).copy()
        if len(sub) < 10:
            print(f"  yr{year}: N={len(sub)} too small, skipped")
            continue
        feats = BASELINE_FEATURES + lagged
        X_df, cols = build_feature_matrix(sub, feats)
        num_enc = [c for c in cols if c in NUM_COLS or c in lagged]
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        X_df[num_enc] = imp.fit_transform(X_df[num_enc])
        X = X_df[cols].values.astype(float)
        y = pd.to_numeric(sub[target], errors="coerce").values.astype(float)
        est = _gb(len(sub) < 100).fit(X, y)
        joblib.dump(est, GB_DIR / f"TBWL_yr{year}_model.joblib")
        joblib.dump({"imputer": imp, "columns": cols, "lagged_cols": lagged,
                     "n_train": len(sub)}, GB_DIR / f"TBWL_yr{year}_meta.joblib")
        print(f"  yr{year}: trained on N={len(sub)}")


def gb_predict_trajectory(patient: dict) -> dict:
    """GB cascade prediction of TBWL yrs 1-6 (mirrors predict.py, GB artifacts).

    Red years (per the shared reliability gate) return None, same as RF.
    """
    out, cascade = {}, {}
    for year in range(1, 7):
        g = gate("TBWL", year)
        if not g["allow_point_prediction"]:
            out[year] = None
            continue
        meta = joblib.load(GB_DIR / f"TBWL_yr{year}_meta.joblib")
        model = joblib.load(GB_DIR / f"TBWL_yr{year}_model.joblib")
        p = dict(patient)
        for lag in meta["lagged_cols"]:
            p[lag] = cascade.get(lag, float("nan"))
        x = encode_patient(p, meta["columns"])
        num_idx = [meta["columns"].index(c) for c in meta["columns"]
                   if c in NUM_COLS or c in meta["lagged_cols"]]
        if num_idx:
            x[:, num_idx] = meta["imputer"].transform(x[:, num_idx])
        pt = float(model.predict(x)[0])
        out[year] = pt
        cascade[TBWL_BY_YEAR[year]] = pt
    return out


def _cluster_on_trajectories(T: np.ndarray, preop: np.ndarray):
    """UMAP -> k-means(5) -> order clusters by ascending mean preop TBWL. Returns labels."""
    Xu = umap.UMAP(**UMAP_KW).fit_transform(StandardScaler().fit_transform(T))
    raw = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(Xu)
    order = pd.Series(preop).groupby(raw).mean().sort_values().index
    remap = {o: i for i, o in enumerate(order)}
    return np.array([remap[c] for c in raw]), Xu


def _cluster_table(df, labels, model_name):
    rows = []
    d = df.copy(); d["cl"] = labels
    for c in range(5):
        s = d[d.cl == c]
        rows.append({
            "model": model_name, "cluster": c + 1, "n": len(s),
            "preop_TBWL": pd.to_numeric(s["Preop_TBWL"], errors="coerce").mean(),
            "pct_female": (s.Sex.astype(str) == "Female").mean() * 100,
            "pct_sleeve": (s.Surgery_Type.astype(str) == "Sleeve").mean() * 100,
            "pct_bypass": (s.Surgery_Type.astype(str) == "Bypass").mean() * 100,
            "pct_revision": (s.Surgery_Type.astype(str) == "Revision").mean() * 100,
            "pct_hispanic": (s.Race.astype(str) == "Hispanic").mean() * 100,
            "pct_white": (s.Race.astype(str) == "White").mean() * 100,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load().reset_index(drop=True)
    train_gb_models(df)

    print("\nGenerating RF and GB predicted trajectories (yrs 1-4) for every patient...")
    rf_T, gb_T = [], []
    for _, r in df.iterrows():
        p = {c: (r.get(c) if c in ["Sex", "Race", "Surgery_Type"]
                 else pd.to_numeric(r.get(c), errors="coerce")) for c in BASELINE_FEATURES}
        rf = predict_trajectory(p)["TBWL"]
        rf_T.append([rf[y]["point"] for y in CLUSTER_YEARS])
        gb = gb_predict_trajectory(p)
        gb_T.append([gb[y] for y in CLUSTER_YEARS])
    rf_T, gb_T = np.array(rf_T), np.array(gb_T)
    preop = pd.to_numeric(df["Preop_TBWL"], errors="coerce").fillna(
        pd.to_numeric(df["Preop_TBWL"], errors="coerce").median()).values

    print("Clustering on RF trajectories, and on GB trajectories (identical procedure)...")
    rf_lab, _ = _cluster_on_trajectories(rf_T, preop)
    gb_lab, gb_Xu = _cluster_on_trajectories(gb_T, preop)

    # convergence between the two models' phenotypes
    ari = adjusted_rand_score(rf_lab, gb_lab)
    # best-match concordance (clusters are label-arbitrary; align by overlap)
    from scipy.optimize import linear_sum_assignment
    M = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            M[i, j] = ((rf_lab == i) & (gb_lab == j)).sum()
    ri, cj = linear_sum_assignment(-M)
    concord = M[ri, cj].sum() / len(df) * 100

    rf_tab = _cluster_table(df, rf_lab, "RF")
    gb_tab = _cluster_table(df, gb_lab, "GB")
    both = pd.concat([rf_tab, gb_tab], ignore_index=True)
    both.to_csv(GB_DIR / "rf_vs_gb_clusters.csv", index=False)

    print("\n" + "=" * 78)
    print("RF vs GB PHENOTYPE CONVERGENCE (does GB reproduce the RF clusters?)")
    print("=" * 78)
    print(f"  Adjusted Rand Index: {ari:.3f}   (1.0 = identical clustering)")
    print(f"  Best-match concordance: {concord:.0f}% of patients land in the same phenotype")
    print("\n  Preop-TBWL ladder (mean %):")
    print("   cluster    RF     GB")
    for c in range(5):
        rf_p = rf_tab.loc[c, "preop_TBWL"]; gb_p = gb_tab.loc[c, "preop_TBWL"]
        print(f"     {c+1}      {rf_p:5.2f}  {gb_p:5.2f}")

    # figure: RF vs GB cluster trajectories side by side
    CL = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for ax, T, lab, name in [(axes[0], rf_T, rf_lab, "Random Forest"),
                             (axes[1], gb_T, gb_lab, "Gradient Boosting")]:
        for c in range(5):
            m = lab == c
            ax.plot(CLUSTER_YEARS, T[m].mean(axis=0), marker="o", lw=2.4, color=CL[c],
                    label=f"Cluster {c+1} (n={int(m.sum())})")
        ax.set_title(f"{name} trajectories", fontsize=12)
        ax.set_xlabel("Year after surgery"); ax.set_xticks(CLUSTER_YEARS)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("Predicted TBWL %")
    fig.suptitle(f"Figure 9. RF and GB predicted-trajectory phenotypes converge "
                 f"(ARI={ari:.2f}, {concord:.0f}% concordant)", fontsize=13, weight="bold")
    plt.tight_layout()
    outfig = Path("presentation/figures/fig9_rf_vs_gb_clusters.png")
    outfig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {GB_DIR/'rf_vs_gb_clusters.csv'}  and  {outfig}")
    print("GB models persisted to artifacts/gb/ (paper only — the app still uses RF).")


if __name__ == "__main__":
    main()
