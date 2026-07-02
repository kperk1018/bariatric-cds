"""BMI milestone analysis — Research Questions 1a and 1b.

Q1a: Do patients who reach BMI ≤ 27 (or ≤ 25) at any postop year gain less weight
     in subsequent years vs. those who did not?
Q1b: How does FML% correlate between milestone and non-milestone groups?

All functions accept DataFrames and return DataFrames/dicts — testable without
the real CSV. Plotting lives in scripts/analyze_bmi_milestone.py.

BMI derivation formula:
    Postop_BMI_yr{n} = Initial_BMI × (1 - TBWL_yr{n} / 100)
This is exact because BMI = Weight / Height² and height is constant, so
BMI scales linearly with weight.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from src.config import TBWL_BY_YEAR, FML_BY_YEAR

_POSTOP_BMI_COLS = {n: f"Postop_BMI_yr{n}" for n in range(1, 7)}


def derive_postop_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Add Postop_BMI_yr{1..6} columns derived from Initial_BMI and TBWL%.

    If columns already exist (real data), sanity-checks vs. formula and warns
    if mean relative error > 5%.
    """
    df = df.copy()
    for yr, col in _POSTOP_BMI_COLS.items():
        tbwl_col = TBWL_BY_YEAR[yr]
        tbwl_num = pd.to_numeric(df[tbwl_col], errors="coerce")
        derived = pd.to_numeric(df["Initial_BMI"], errors="coerce") * (1 - tbwl_num / 100)
        if col in df.columns:
            rel_err = ((df[col] - derived).abs() / df[col].abs()).mean()
            if rel_err > 0.05:
                warnings.warn(
                    f"{col} already exists but diverges from formula by {rel_err:.1%}. "
                    "Using formula-derived values."
                )
        df[col] = derived
    return df


def flag_milestone(df: pd.DataFrame, threshold: float = 27.0) -> pd.DataFrame:
    """Add ever_bmi{int(threshold)} bool and min_postop_bmi columns."""
    df = df.copy()
    bmi_cols = [_POSTOP_BMI_COLS[n] for n in range(1, 7) if _POSTOP_BMI_COLS[n] in df.columns]
    df["min_postop_bmi"] = df[bmi_cols].min(axis=1)
    flag_col = f"ever_bmi{int(threshold)}"
    df[flag_col] = df["min_postop_bmi"] <= threshold
    return df


def landmark_split(
    df: pd.DataFrame,
    landmark_year: int,
    threshold: float = 27.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into milestone (A) and non-milestone (B) groups by landmark_year.

    Group A: achieved BMI ≤ threshold at ANY year ≤ landmark_year.
    Group B: did NOT achieve the milestone by landmark_year.
    Only includes patients with at least one non-NaN BMI value through landmark_year.
    Groups are mutually exclusive by design.
    """
    df = df.copy()
    bmi_cols_up_to = [_POSTOP_BMI_COLS[n] for n in range(1, landmark_year + 1)
                      if _POSTOP_BMI_COLS[n] in df.columns]

    # Require at least one observed BMI value through landmark year
    has_data = df[bmi_cols_up_to].notna().any(axis=1)
    df = df[has_data].copy()

    achieved = (df[bmi_cols_up_to] <= threshold).any(axis=1)
    group_a = df[achieved].copy()
    group_b = df[~achieved].copy()
    return group_a, group_b


def compare_trajectories(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    years: list[int],
) -> dict:
    """Mann-Whitney U + LME for TBWL% trajectories in post-landmark years.

    Returns per-year stats and an LME summary (year × milestone_group interaction).
    """
    per_year: dict = {}
    for yr in years:
        col = TBWL_BY_YEAR[yr]
        a_vals = pd.to_numeric(group_a[col], errors="coerce").dropna().values
        b_vals = pd.to_numeric(group_b[col], errors="coerce").dropna().values
        if len(a_vals) < 3 or len(b_vals) < 3:
            per_year[yr] = {
                "n_a": len(a_vals), "n_b": len(b_vals),
                "median_a": float(np.median(a_vals)) if len(a_vals) else None,
                "median_b": float(np.median(b_vals)) if len(b_vals) else None,
                "u_stat": None, "p_value": None, "note": "insufficient data",
            }
            continue
        u_stat, p_val = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        per_year[yr] = {
            "n_a": len(a_vals), "n_b": len(b_vals),
            "median_a": round(float(np.median(a_vals)), 2),
            "median_b": round(float(np.median(b_vals)), 2),
            "u_stat": round(float(u_stat), 2),
            "p_value": round(float(p_val), 4),
        }

    lme_summary = _run_lme(group_a, group_b, years, outcome_col_map=TBWL_BY_YEAR)
    return {"per_year": per_year, "lme_summary": lme_summary}


def compare_fml_trajectories(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    years: list[int],
) -> dict:
    """Mann-Whitney U + Spearman correlation for FML% in post-landmark years."""
    per_year: dict = {}
    for yr in years:
        col = FML_BY_YEAR[yr]
        if col not in group_a.columns:
            continue
        a_vals = pd.to_numeric(group_a[col], errors="coerce").dropna().values
        b_vals = pd.to_numeric(group_b[col], errors="coerce").dropna().values
        if len(a_vals) < 3 or len(b_vals) < 3:
            per_year[yr] = {
                "n_a": len(a_vals), "n_b": len(b_vals),
                "median_a": None, "median_b": None,
                "u_stat": None, "p_value": None, "note": "insufficient data",
            }
            continue
        u_stat, p_val = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")

        # Within-group Spearman: FML% vs TBWL% at the same year
        tbwl_col = TBWL_BY_YEAR[yr]
        def _spearman(grp: pd.DataFrame) -> tuple[float | None, float | None]:
            paired = grp[[tbwl_col, col]].copy()
            paired[tbwl_col] = pd.to_numeric(paired[tbwl_col], errors="coerce")
            paired[col] = pd.to_numeric(paired[col], errors="coerce")
            paired = paired.dropna()
            if len(paired) < 5:
                return None, None
            r, p = stats.spearmanr(paired[tbwl_col].values, paired[col].values)
            return round(float(r), 3), round(float(p), 4)

        r_a, p_a = _spearman(group_a)
        r_b, p_b = _spearman(group_b)

        per_year[yr] = {
            "n_a": len(a_vals), "n_b": len(b_vals),
            "median_a": round(float(np.median(a_vals)), 2),
            "median_b": round(float(np.median(b_vals)), 2),
            "u_stat": round(float(u_stat), 2),
            "p_value": round(float(p_val), 4),
            "spearman_fml_vs_tbwl_group_a": {"r": r_a, "p": p_a},
            "spearman_fml_vs_tbwl_group_b": {"r": r_b, "p": p_b},
        }

    lme_summary = _run_lme(group_a, group_b, years, outcome_col_map=FML_BY_YEAR)
    return {"per_year": per_year, "lme_summary": lme_summary}


def _run_lme(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    years: list[int],
    outcome_col_map: dict,
) -> str:
    """Linear mixed-effects model: outcome ~ year * milestone_group (random intercept)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return "statsmodels not available — skip LME."

    rows = []
    for grp_label, grp_df in [("milestone", group_a), ("control", group_b)]:
        for yr in years:
            col = outcome_col_map[yr]
            if col not in grp_df.columns:
                continue
            sub = grp_df[[col]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna()
            sub["year"] = yr
            sub["milestone_group"] = grp_label
            sub["patient_idx"] = sub.index  # preserve original row index → correct random intercepts
            rows.append(sub.rename(columns={col: "outcome"}))

    if not rows:
        return "No data available for LME."

    long_df = pd.concat(rows, ignore_index=True)
    long_df["milestone_group"] = (long_df["milestone_group"] == "milestone").astype(int)

    try:
        model = smf.mixedlm(
            "outcome ~ year * milestone_group",
            long_df,
            groups=long_df["patient_idx"],
        )
        result = model.fit(disp=False)
        return result.summary().as_text()
    except Exception as exc:
        return f"LME failed: {exc}"
