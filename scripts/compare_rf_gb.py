"""RandomForest vs GradientBoosting — head-to-head on OUR deduped, leakage-safe data.

Why this exists (Ioanna, 2026-07-13 meeting): the S5 bake-off shows GradientBoosting
edging out RandomForest at years 3-4. A reviewer will ask "then why did you pick RF?".
This script answers that with our own numbers rather than hers, using the identical
feature set, dedup, and CV scheme for both models — so the comparison is apples to
apples. RF remains the deployed/app model; GB is reported for the paper.

Metrics per (outcome, year, model), 5-fold CV, seed 42:
  R2    — variance explained (the stringent metric)
  RMSE  — average error in percentage points
  AUC   — discrimination. Computed the way 1A does it: dichotomise the outcome at the
          TRAINING median, then ask how well the regressor's continuous prediction ranks
          test patients above/below that line. This is a ranking metric, not a classifier
          AUC, and is labelled as such.

Outputs:
  artifacts/rf_vs_gb.csv        — tidy results table
  presentation/figures/fig2b_rf_vs_gb.png (via make_presentation_figures.py)

Run:
    PYTHONPATH=. python scripts/compare_rf_gb.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.config import (
    ARTIFACTS, BASELINE_FEATURES, TBWL_BY_YEAR, FML_BY_YEAR,
    LAGGED_TBWL_BY_YEAR, LAGGED_FML_BY_YEAR, SEED,
)
from src.data_load import load
from src.preprocess import build_feature_matrix, get_numeric_cols

CV_FOLDS = 5
MIN_N = 25          # below this, CV is noise — report but flag


def _models(small_n: bool) -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1,
            min_samples_leaf=5 if small_n else 2,
            max_features=0.5 if small_n else "sqrt",
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, random_state=SEED, max_depth=3,
            subsample=0.7 if small_n else 0.9,
            min_samples_leaf=5 if small_n else 2,
        ),
    }


def _auc_rank(y_true_test, y_pred_test, train_median) -> float:
    """AUC of the regressor's ranking vs a median split (1A's convention)."""
    y_bin = (y_true_test > train_median).astype(int)
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return roc_auc_score(y_bin, y_pred_test)


def evaluate(X, y, model_name, small_n) -> dict:
    n = len(y)
    n_splits = min(CV_FOLDS, max(2, n // 5))
    if n < 10:
        return {"r2": np.nan, "rmse": np.nan, "auc": np.nan, "n": n}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    r2s, rmses, aucs = [], [], []
    for tr, te in kf.split(X):
        est = _models(small_n)[model_name]
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        rmses.append(float(np.sqrt(np.mean((y[te] - pred) ** 2))))
        aucs.append(_auc_rank(y[te], pred, np.median(y[tr])))
    return {"r2": float(np.nanmean(r2s)), "rmse": float(np.nanmean(rmses)),
            "auc": float(np.nanmean(aucs)), "n": n}


def main() -> None:
    df = load()
    num_cols = get_numeric_cols(BASELINE_FEATURES)
    rows = []

    print(f"RF vs GB — 5-fold CV on N={len(df)} (deduped), leakage-safe features\n")
    hdr = (f"{'outcome':<6}{'yr':>3}{'n':>6}  "
           f"{'RF R2':>7}{'GB R2':>7}{'  ':>2}{'RF RMSE':>8}{'GB RMSE':>8}{'  ':>2}"
           f"{'RF AUC':>8}{'GB AUC':>8}   winner")
    print(hdr); print("-" * len(hdr))

    for outcome, col_map, lag_map in [("TBWL", TBWL_BY_YEAR, LAGGED_TBWL_BY_YEAR),
                                      ("FML", FML_BY_YEAR, LAGGED_FML_BY_YEAR)]:
        for year in range(1, 7):
            target = col_map[year]
            lagged = lag_map[year]
            sub = df.dropna(subset=[target] + BASELINE_FEATURES + lagged).copy()
            n = len(sub)
            if n < 10:
                continue
            feats = BASELINE_FEATURES + lagged
            X_df, cols = build_feature_matrix(sub, feats)
            num_enc = [c for c in cols if c in num_cols or c in lagged]
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            X_df[num_enc] = imp.fit_transform(X_df[num_enc])
            X = X_df[cols].values.astype(float)
            y = pd.to_numeric(sub[target], errors="coerce").values.astype(float)
            small_n = n < 100

            res = {m: evaluate(X, y, m, small_n) for m in _models(small_n)}
            for m, r in res.items():
                rows.append({"outcome": outcome, "year": year, "model": m, **r})

            rf, gb = res["RandomForest"], res["GradientBoosting"]
            win = "RF" if (rf["r2"] or -9) > (gb["r2"] or -9) else "GB"
            flag = "  (N small)" if n < MIN_N else ""
            print(f"{outcome:<6}{year:>3}{n:>6}  {rf['r2']:>7.3f}{gb['r2']:>7.3f}  "
                  f"{rf['rmse']:>8.2f}{gb['rmse']:>8.2f}  "
                  f"{rf['auc']:>8.3f}{gb['auc']:>8.3f}   {win}{flag}")

    out = pd.DataFrame(rows)
    ARTIFACTS.mkdir(exist_ok=True)
    out.to_csv(ARTIFACTS / "rf_vs_gb.csv", index=False)
    print(f"\nSaved: {ARTIFACTS/'rf_vs_gb.csv'}")

    # headline summary for the slide / manuscript
    t = out[out.outcome == "TBWL"].pivot(index="year", columns="model", values="r2")
    t = t.dropna()
    rf_wins = int((t["RandomForest"] > t["GradientBoosting"]).sum())
    print(f"\nTBWL: RandomForest wins on R² in {rf_wins}/{len(t)} years "
          f"(mean R²  RF={t['RandomForest'].mean():.3f}  GB={t['GradientBoosting'].mean():.3f})")
    a = out[out.outcome == "TBWL"].pivot(index="year", columns="model", values="auc").dropna()
    print(f"TBWL AUC: RF mean={a['RandomForest'].mean():.3f}  GB mean={a['GradientBoosting'].mean():.3f}")


if __name__ == "__main__":
    main()
