"""Direct preop models — fix the cascade collapse found by validate_models.py.

Problem: preop-mode predictions cascade predicted lags through lag-trained
models; OOF R² collapses (TBWL yr2 0.18, yr4 ≈ 0). The standard remedy for
multi-step forecasting is the DIRECT strategy: train year N straight from
preop features, so training conditions match deployment conditions.

This script:
1. OOF-compares (same KFold 5, seed 42 as validate_models.py) for each
   (outcome, year): the recursive cascade reference vs direct models over a
   small menu — {15 baseline features, expanded preop features} ×
   {RandomForest, HistGradientBoosting, Ridge}.
2. Selects the winner per (outcome, year) by OOF R² (winner-take-all across
   the 6 direct configs; mild selection optimism accepted and logged).
3. Refits the winner on all available patients and persists:
     artifacts/{outcome}_yr{year}_direct_model.joblib
     artifacts/{outcome}_yr{year}_direct_meta.joblib
   and appends direct-mode entries (OOF residual quantiles + R²) to
   artifacts/calibration.joblib under key "direct_per_year".

Expanded features are preop-only. Excluded to avoid leakage: LOS,
Operation_Time, Follow_up_months, Revision_Conversion, Hepatomegaly
(intraop finding). Excluded for sparsity (<40% coverage): CHQ_score,
PHQ-9_score, Exercise, Calories_burned_per_week.

Run:
    PYTHONPATH=. python scripts/train_direct_models.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA, ARTIFACTS, BASELINE_FEATURES, TBWL_BY_YEAR, FML_BY_YEAR, SEED,
)
from src.data_load import load
from src.preprocess import build_feature_matrix, get_numeric_cols

N_FOLDS = 5
MIN_CALIB = 30

# Preop-only expansion candidates (verified against the CSV; coverage >= 40%).
# Names use the cleaned 1A/1B convention (data_load.load standardizes on load).
EXPANDED_EXTRA = [
    "Preop_Glucose", "Preop_Albumin", "Preop_Total_Protein", "Preop_Insulin",
    "Preop_HbA1c", "Preop_chol", "Preop_HDL", "Preop_LDL", "Preop_TG",
    "Preop_TSH", "Preop_AST", "Preop_ALT", "Preop_ALP", "Preop_CRP",
    "OSA", "DM", "HTN", "Hyperlipidemia", "Depression", "GERD", "Smoker",
    "CPAP", "Home_Anticoagulation", "Prior_MBS_Procedure",
    "Prior_Abdominal_Procedure", "Mobility_Affecting_Surgery", "Habitus",
    "BES_score", "ACE_score", "Epworth_score", "IWQoL_score",
    "Preop_Visits", "GERD_on_UGI", "HH_on_UGI", "Dysmotility",
]

FEATURE_SETS = {
    "F15": BASELINE_FEATURES,
    "Fexp": BASELINE_FEATURES + EXPANDED_EXTRA,
}

# Years worth modelling directly: all non-red-by-N years where preop point
# predictions are shown (red-tier years remain refused regardless).
TARGET_YEARS = {
    "TBWL": [1, 2, 3, 4, 5],
    "FML":  [2, 3, 4],
}


def _estimators() -> dict:
    return {
        "RF": RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=-1,
            min_samples_leaf=5, max_features=0.5,
        ),
        "HGB": HistGradientBoostingRegressor(
            random_state=SEED, learning_rate=0.06, max_iter=300,
            min_samples_leaf=20, l2_regularization=1.0,
        ),
        "Ridge": Ridge(alpha=10.0, random_state=SEED),
    }


def _prepare_xy(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """Complete-case on target only; coerce numerics; return raw feature df + y."""
    y_all = pd.to_numeric(df[target_col], errors="coerce")
    mask = y_all.notna()
    sub = df.loc[mask, feature_cols].copy()
    num_cols = get_numeric_cols(feature_cols)
    for c in num_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub, y_all[mask].values.astype(float), mask


def _oof_direct(df, target_col, feature_cols, est_name):
    """OOF predictions for one direct config. Imputation/scaling inside folds."""
    sub, y, mask = _prepare_xy(df, target_col, feature_cols)
    idx = sub.index.to_numpy()
    num_cols = get_numeric_cols(feature_cols)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(sub):
        X_tr_df, cols = build_feature_matrix(sub.iloc[tr], feature_cols)
        X_te_df, _ = build_feature_matrix(sub.iloc[te], feature_cols)
        X_te_df = X_te_df.reindex(columns=cols, fill_value=0)

        num_enc = [c for c in cols if c in num_cols]
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr_df[num_enc] = imp.fit_transform(X_tr_df[num_enc])
        X_te_df[num_enc] = imp.transform(X_te_df[num_enc])

        X_tr, X_te = X_tr_df.values.astype(float), X_te_df.values.astype(float)
        est = _estimators()[est_name]
        if est_name == "Ridge":
            sc = StandardScaler()
            X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_tr, y[tr])
            oof[te] = est.predict(X_te)
    return y, oof, idx


def _r2_rmse(y, p):
    r = y - p
    ss_tot = np.sum((y - y.mean()) ** 2)
    return (1 - np.sum(r ** 2) / ss_tot if ss_tot > 0 else np.nan,
            float(np.sqrt(np.mean(r ** 2))))


def main() -> None:
    print(f"Loading {DATA} ...")
    df = load(str(DATA))
    print(f"  {len(df)} patients.\n")

    # cascade reference from the existing calibration artifact
    calib_path = ARTIFACTS / "calibration.joblib"
    calib = joblib.load(calib_path) if calib_path.exists() else {"per_year": {}}
    cascade_ref = calib.get("per_year", {})

    print("OOF comparison — direct configs vs deployed cascade (R²):")
    hdr = (f"{'outcome':<7}{'yr':>3}{'n':>6} {'cascade':>8} "
           + " ".join(f"{fs}+{en:<5}" for fs in FEATURE_SETS for en in _estimators()))
    print(hdr)
    print("-" * len(hdr))

    winners: dict = {}
    for outcome, col_map in [("TBWL", TBWL_BY_YEAR), ("FML", FML_BY_YEAR)]:
        for yr in TARGET_YEARS[outcome]:
            target_col = col_map[yr]
            cas_r2 = cascade_ref.get((outcome, yr), {}).get("cascade_r2")
            row_scores = {}
            best = None
            for fs_name, fs_cols in FEATURE_SETS.items():
                for est_name in _estimators():
                    y, oof, idx = _oof_direct(df, target_col, fs_cols, est_name)
                    r2, rmse = _r2_rmse(y, oof)
                    row_scores[(fs_name, est_name)] = r2
                    if best is None or r2 > best["r2"]:
                        best = {"fs": fs_name, "est": est_name, "r2": r2,
                                "rmse": rmse, "y": y, "oof": oof, "n": len(y)}
            winners[(outcome, yr)] = best
            cas_str = f"{cas_r2:>8.3f}" if cas_r2 is not None else f"{'--':>8}"
            score_str = " ".join(
                f"{row_scores[(fs, en)]:>8.3f}"
                for fs in FEATURE_SETS for en in _estimators()
            )
            print(f"{outcome:<7}{yr:>3}{best['n']:>6} {cas_str} {score_str}"
                  f"   -> {best['fs']}+{best['est']}")

    # Refit winners on all data; persist artifacts + direct calibration entries
    print("\nRefitting winners on full data and persisting artifacts...")
    direct_calib = {}
    for (outcome, yr), best in winners.items():
        target_col = (TBWL_BY_YEAR if outcome == "TBWL" else FML_BY_YEAR)[yr]
        fs_cols = FEATURE_SETS[best["fs"]]
        sub, y, _ = _prepare_xy(df, target_col, fs_cols)
        X_df, cols = build_feature_matrix(sub, fs_cols)
        num_cols = get_numeric_cols(fs_cols)
        num_enc = [c for c in cols if c in num_cols]
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        X_df[num_enc] = imp.fit_transform(X_df[num_enc])
        X = X_df.values.astype(float)

        est = _estimators()[best["est"]]
        scaler = None
        if best["est"] == "Ridge":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X, y)

        joblib.dump(est, ARTIFACTS / f"{outcome}_yr{yr}_direct_model.joblib")
        joblib.dump(
            {"columns": cols, "imputer": imp, "num_encoded": num_enc,
             "scaler": scaler, "feature_set": best["fs"],
             "feature_cols": fs_cols, "estimator": best["est"],
             "n_train": len(y)},
            ARTIFACTS / f"{outcome}_yr{yr}_direct_meta.joblib",
        )

        resid = best["y"] - best["oof"]
        direct_calib[(outcome, yr)] = {
            "n_oof": int(best["n"]),
            "direct_r2": round(float(best["r2"]), 3),
            "direct_rmse": round(float(best["rmse"]), 2),
            "resid_q025": float(np.quantile(resid, 0.025)),
            "resid_q975": float(np.quantile(resid, 0.975)),
            "config": f"{best['fs']}+{best['est']}",
        }
        print(f"  {outcome} yr{yr}: {best['fs']}+{best['est']} "
              f"R2={best['r2']:.3f} RMSE={best['rmse']:.2f} n={best['n']}")

    calib["direct_per_year"] = direct_calib
    calib["min_calib"] = calib.get("min_calib", MIN_CALIB)
    joblib.dump(calib, calib_path)
    print(f"\nCalibration updated: {calib_path} (added 'direct_per_year').")


if __name__ == "__main__":
    main()
