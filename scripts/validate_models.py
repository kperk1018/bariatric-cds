"""Honest model validation — checks the deployed system, not just the training CV.

Five checks, all out-of-fold (KFold 5, seed 42, split once over all patients):

1. CASCADE vs ORACLE-LAG accuracy.
   Models are trained with the *actual* prior-year outcome as a lag feature, but
   at inference predict.py cascades *predicted* lags forward. S5 R² was computed
   under oracle-lag conditions. This measures what the app actually delivers.

2. UNCERTAINTY BAND COVERAGE.
   The app shows point ± 1.96×RMSE(S5). If errors were Gaussian and RMSE honest,
   ~95% of actuals would fall inside. Empirical coverage below ~90% means the
   bands are lying to the surgeon.

3. NAIVE BASELINES.
   Cohort-mean predictor and carry-forward (predict yr N = actual yr N-1).
   A model that can't beat carry-forward has no clinical information content
   beyond the previous measurement.

4. ATTRITION BIAS.
   Standardized mean difference (SMD) of baseline features between patients
   with vs without follow-up at each year. |SMD| > 0.2 = the training subsample
   for that year is systematically different from the full cohort.

5. SUBGROUP RESIDUAL BIAS.
   Mean OOF residual by Sex and Surgery_Type at green-tier years. Systematic
   over/under-prediction for a subgroup is a fairness and validity problem.

Outputs:
  - report to stdout
  - artifacts/calibration.joblib — per-(outcome, year) empirical OOF cascade
    residual quantiles [2.5%, 97.5%] + coverage stats, consumed by predict.py
    for calibrated uncertainty bands.

Run:
    PYTHONPATH=. python scripts/validate_models.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATA, ARTIFACTS, BASELINE_FEATURES, TBWL_BY_YEAR, FML_BY_YEAR,
    LAGGED_TBWL_BY_YEAR, LAGGED_FML_BY_YEAR, MODEL_PERFORMANCE,
)
from src.data_load import load
from src.preprocess import build_feature_matrix, get_numeric_cols, CATEGORICAL_COLS
from src.reliability import tier
from reproduce_models import _make_estimator, _needs_scaling  # scripts/ on path

SEED = 42
N_FOLDS = 5
MIN_TRAIN = 20     # skip a fold-year if the training subset is smaller
MIN_CALIB = 30     # minimum OOF residuals to trust empirical quantiles

OUTCOME_MAPS = {
    "TBWL": (TBWL_BY_YEAR, LAGGED_TBWL_BY_YEAR),
    "FML":  (FML_BY_YEAR,  LAGGED_FML_BY_YEAR),
}
NUM_COLS = get_numeric_cols(BASELINE_FEATURES)


def _fit_year_model(train_df: pd.DataFrame, outcome: str, year: int):
    """Fit one year's model on a training fold, mirroring reproduce_models.py exactly.

    Returns None if the training subset is too small.
    """
    col_map, lagged_map = OUTCOME_MAPS[outcome]
    target_col = col_map[year]
    lagged_cols = lagged_map[year]
    required = [target_col] + BASELINE_FEATURES + lagged_cols
    sub = train_df.dropna(subset=required).copy()
    if len(sub) < MIN_TRAIN:
        return None

    all_features = BASELINE_FEATURES + lagged_cols
    X_df, encoded_cols = build_feature_matrix(sub, all_features)
    imputer = SimpleImputer(strategy="median")
    num_encoded = [c for c in encoded_cols if c in NUM_COLS or c in lagged_cols]
    for c in num_encoded:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df[num_encoded] = imputer.fit_transform(X_df[num_encoded])

    X = X_df[encoded_cols].values.astype(float)
    y = pd.to_numeric(sub[target_col], errors="coerce").values.astype(float)

    est = _make_estimator(outcome, year, len(sub))
    scaler = None
    if _needs_scaling(outcome, year):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y)
    return {
        "est": est, "scaler": scaler, "imputer": imputer,
        "columns": encoded_cols, "num_encoded": num_encoded,
        "lagged_cols": lagged_cols, "n_train": len(sub),
    }


def _predict_batch(model_bundle: dict, feat_df: pd.DataFrame) -> np.ndarray:
    """Predict for a batch of patients given raw feature rows (baseline + lag cols)."""
    X_df, _ = build_feature_matrix(feat_df, list(feat_df.columns))
    X_df = X_df.reindex(columns=model_bundle["columns"], fill_value=0)
    num_encoded = model_bundle["num_encoded"]
    # raw CSV has stray strings (e.g. ' ') in numeric columns; coerce before imputing
    for c in num_encoded:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df[num_encoded] = model_bundle["imputer"].transform(X_df[num_encoded])
    X = X_df.values.astype(float)
    if model_bundle["scaler"] is not None:
        X = model_bundle["scaler"].transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model_bundle["est"].predict(X)


def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return r2, rmse


def run_oof(df: pd.DataFrame) -> dict:
    """Out-of-fold cascade and oracle-lag predictions for every (outcome, year).

    Returns {(outcome, year): {"idx", "y", "cascade_pred", "oracle_pred"}} where
    oracle_pred is NaN for patients missing the actual lag.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    store: dict = {
        (o, y): {"idx": [], "y": [], "cascade": [], "oracle": []}
        for o in OUTCOME_MAPS for y in range(1, 7)
    }

    for fold_i, (tr_pos, te_pos) in enumerate(kf.split(df), 1):
        train_df = df.iloc[tr_pos]
        test_df = df.iloc[te_pos]
        print(f"  fold {fold_i}/{N_FOLDS}: train={len(train_df)} test={len(test_df)}")

        for outcome, (col_map, lagged_map) in OUTCOME_MAPS.items():
            models = {yr: _fit_year_model(train_df, outcome, yr) for yr in range(1, 7)}

            # cascade lag per test patient: predicted value of the prior year
            cascade_lag = pd.Series(np.nan, index=test_df.index)

            for yr in range(1, 7):
                mb = models[yr]
                lag_col = lagged_map[yr][0] if lagged_map[yr] else None
                y_actual = pd.to_numeric(test_df[col_map[yr]], errors="coerce")

                if mb is None:
                    # can't predict this year; cascade uses actuals where present
                    if lag_col is not None:
                        cascade_lag = pd.to_numeric(test_df[lag_col], errors="coerce")
                    continue

                feat = test_df[BASELINE_FEATURES].copy()
                if lag_col is not None:
                    feat[lag_col] = cascade_lag.values

                cas_pred = _predict_batch(mb, feat)

                # oracle: same model, actual lag (only defined where lag observed)
                if lag_col is not None:
                    actual_lag = pd.to_numeric(test_df[lag_col], errors="coerce")
                    feat_o = test_df[BASELINE_FEATURES].copy()
                    feat_o[lag_col] = actual_lag.values
                    ora_pred_full = _predict_batch(mb, feat_o)
                    ora_pred = np.where(actual_lag.notna().values, ora_pred_full, np.nan)
                else:
                    ora_pred = cas_pred.copy()

                mask = y_actual.notna().values
                s = store[(outcome, yr)]
                s["idx"].extend(test_df.index[mask].tolist())
                s["y"].extend(y_actual.values[mask].tolist())
                s["cascade"].extend(cas_pred[mask].tolist())
                s["oracle"].extend(ora_pred[mask].tolist())

                # advance the cascade with this year's predictions
                cascade_lag = pd.Series(cas_pred, index=test_df.index)

    return store


def report_accuracy(df: pd.DataFrame, store: dict) -> dict:
    """Check 1-3: cascade vs oracle vs S5 vs naive baselines; band coverage."""
    print("\n" + "=" * 100)
    print("CHECK 1-3 — OOF accuracy: cascade (deployed) vs oracle-lag (S5 conditions) vs baselines")
    print("=" * 100)
    hdr = (f"{'outcome':<7}{'yr':>3} {'tier':<6}{'n':>5} {'S5_R2':>7} {'oracle_R2':>10} "
           f"{'cascade_R2':>11} {'cascade_RMSE':>13} {'carryfwd_RMSE':>14} {'cover95%':>9}")
    print(hdr)
    print("-" * len(hdr))

    calibration: dict = {}
    for outcome, (col_map, lagged_map) in OUTCOME_MAPS.items():
        for yr in range(1, 7):
            s = store[(outcome, yr)]
            if len(s["y"]) < 10:
                continue
            y = np.array(s["y"])
            cas = np.array(s["cascade"])
            ora = np.array(s["oracle"])
            t = tier(outcome, yr)
            s5_r2 = MODEL_PERFORMANCE[outcome][yr]["r2"]
            s5_rmse = MODEL_PERFORMANCE[outcome][yr]["rmse"]

            cas_r2, cas_rmse = _r2_rmse(y, cas)
            ora_mask = ~np.isnan(ora)
            ora_r2, _ = _r2_rmse(y[ora_mask], ora[ora_mask]) if ora_mask.sum() >= 10 else (float("nan"), float("nan"))

            # carry-forward baseline: predict yr N = actual yr N-1 (paired subset)
            cf_rmse = float("nan")
            if lagged_map[yr]:
                lag_col = lagged_map[yr][0]
                sub = df.loc[s["idx"]]
                lag_vals = pd.to_numeric(sub[lag_col], errors="coerce").values
                paired = ~np.isnan(lag_vals)
                if paired.sum() >= 10:
                    _, cf_rmse = _r2_rmse(y[paired], lag_vals[paired])

            # coverage of the deployed ±1.96×RMSE(S5) band around cascade predictions
            resid = y - cas
            covered = np.abs(resid) <= 1.96 * s5_rmse
            coverage = float(covered.mean())

            calibration[(outcome, yr)] = {
                "n_oof": int(len(y)),
                "cascade_r2": round(cas_r2, 3),
                "cascade_rmse": round(cas_rmse, 2),
                "oracle_r2": round(ora_r2, 3) if ora_r2 == ora_r2 else None,
                "coverage_1p96_s5": round(coverage, 3),
                "resid_q025": float(np.quantile(resid, 0.025)),
                "resid_q975": float(np.quantile(resid, 0.975)),
                "resid_median": float(np.median(resid)),
            }

            print(f"{outcome:<7}{yr:>3} {t:<6}{len(y):>5} {s5_r2:>7.3f} "
                  f"{ora_r2:>10.3f} {cas_r2:>11.3f} {cas_rmse:>13.2f} "
                  f"{cf_rmse:>14.2f} {coverage:>8.0%}")

    print("\n  oracle_R2  = accuracy when the actual prior-year value is known (S5 conditions;")
    print("               this is what the app gets in follow-up mode)")
    print("  cascade_R2 = accuracy when prior years are themselves predicted (preop mode —")
    print("               what the app delivers from baseline-only inputs)")
    print("  cover95%   = fraction of actuals inside the displayed ±1.96×RMSE(S5) band")
    return calibration


def report_attrition(df: pd.DataFrame) -> None:
    """Check 4: SMD of baseline features, completers vs dropouts per year."""
    print("\n" + "=" * 100)
    print("CHECK 4 — Attrition bias: baseline SMD, patients WITH vs WITHOUT follow-up (|SMD|>0.2 flagged)")
    print("=" * 100)
    num_baseline = [c for c in BASELINE_FEATURES if c not in CATEGORICAL_COLS]
    for yr in [2, 4, 5, 6]:
        col = TBWL_BY_YEAR[yr]
        has = pd.to_numeric(df[col], errors="coerce").notna()
        flags = []
        for feat in num_baseline:
            v = pd.to_numeric(df[feat], errors="coerce")
            a, b = v[has].dropna(), v[~has].dropna()
            if len(a) < 10 or len(b) < 10:
                continue
            pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
            if pooled == 0:
                continue
            smd = (a.mean() - b.mean()) / pooled
            if abs(smd) > 0.2:
                flags.append(f"{feat} (SMD={smd:+.2f})")
        n_with = int(has.sum())
        flag_str = "; ".join(flags) if flags else "none — subsample looks representative"
        print(f"  yr{yr} (n={n_with} with follow-up): {flag_str}")


def report_subgroups(df: pd.DataFrame, store: dict) -> None:
    """Check 5: mean OOF cascade residual by Sex / Surgery_Type at green TBWL years."""
    print("\n" + "=" * 100)
    print("CHECK 5 — Subgroup residual bias (OOF cascade, green TBWL years; + = model under-predicts)")
    print("=" * 100)
    for yr in [2, 3, 4]:
        s = store[("TBWL", yr)]
        if len(s["y"]) < 30:
            continue
        resid = np.array(s["y"]) - np.array(s["cascade"])
        sub = df.loc[s["idx"]]
        print(f"  TBWL yr{yr}:")
        for cat in ["Sex", "Surgery_Type"]:
            for level, grp_mask in sub.groupby(cat).groups.items():
                m = sub.index.isin(grp_mask)
                # sub may have duplicate index labels only if df did; use positional mask
                m = np.array([i in set(grp_mask) for i in s["idx"]])
                if m.sum() < 15:
                    continue
                mr = float(resid[m].mean())
                flag = "  <-- check" if abs(mr) > 2.0 else ""
                print(f"    {cat}={level:<20} n={int(m.sum()):>4}  mean_resid={mr:+6.2f} pp{flag}")


def main() -> None:
    print(f"Loading {DATA} ...")
    df = load(str(DATA))
    print(f"  {len(df)} patients.\n\nRunning {N_FOLDS}-fold OOF evaluation (cascade + oracle)...")

    store = run_oof(df)
    calibration = report_accuracy(df, store)
    report_attrition(df)
    report_subgroups(df, store)

    # persist calibration for predict.py (empirical bands where n_oof >= MIN_CALIB)
    out = ARTIFACTS / "calibration.joblib"
    joblib.dump({"min_calib": MIN_CALIB, "per_year": calibration, "seed": SEED,
                 "n_folds": N_FOLDS}, out)
    print(f"\nCalibration saved: {out}")
    print("predict.py will use empirical OOF residual quantiles for uncertainty bands "
          f"where n_oof >= {MIN_CALIB}.")


if __name__ == "__main__":
    main()
