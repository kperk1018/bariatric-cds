"""Q1a + Q1b — BMI Milestone Analysis.

Q1a: If a patient reaches BMI ≤ 27 (or ≤ 25) at any postop year, do they gain
     less weight in the years after that milestone vs. patients who didn't?

Q1b: If a patient reaches BMI ≤ 27 (or ≤ 25), how does their FML% correlate
     to patients who didn't reach the milestone?

Run:
    python scripts/analyze_bmi_milestone.py

Outputs:
  - Summary tables printed to stdout
  - artifacts/bmi_milestone_analysis.png (trajectory comparison plots)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA, ARTIFACTS, TBWL_BY_YEAR, FML_BY_YEAR
from src.data_load import load
from src.bmi_milestone import (
    derive_postop_bmi,
    flag_milestone,
    landmark_split,
    compare_trajectories,
    compare_fml_trajectories,
)

THRESHOLDS = [27.0, 25.0]
LANDMARK_YEARS = [2, 3]
TIER_COLORS = {"milestone": "#2196F3", "control": "#F44336"}


def _print_table(title: str, results: dict, years: list[int]) -> None:
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    per_year = results.get("per_year", {})
    print(f"{'Yr':>3}  {'n_A':>5}  {'n_B':>5}  {'Median_A':>9}  {'Median_B':>9}  {'p-value':>9}")
    print("-" * 50)
    for yr in years:
        if yr not in per_year:
            continue
        r = per_year[yr]
        p_str = f"{r['p_value']:.4f}" if r["p_value"] is not None else "  N/A  "
        m_a = f"{r['median_a']:.1f}" if r["median_a"] is not None else "  N/A"
        m_b = f"{r['median_b']:.1f}" if r["median_b"] is not None else "  N/A"
        print(f"{yr:>3}  {r['n_a']:>5}  {r['n_b']:>5}  {m_a:>9}  {m_b:>9}  {p_str:>9}")


def _plot_trajectory_comparison(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    years: list[int],
    threshold: float,
    landmark_year: int,
    ax: plt.Axes,
    outcome_col_map: dict,
    ylabel: str,
) -> None:
    for grp_df, label, color in [
        (group_a, f"BMI ≤ {int(threshold)} (n={len(group_a)})", TIER_COLORS["milestone"]),
        (group_b, f"BMI > {int(threshold)} (n={len(group_b)})", TIER_COLORS["control"]),
    ]:
        means, sems, valid_yrs = [], [], []
        for yr in years:
            col = outcome_col_map[yr]
            if col not in grp_df.columns:
                continue
            vals = pd.to_numeric(grp_df[col], errors="coerce").dropna()
            if len(vals) < 3:
                continue
            means.append(float(vals.mean()))
            sems.append(float(vals.sem()))
            valid_yrs.append(yr)
        if valid_yrs:
            ax.errorbar(
                valid_yrs, means, yerr=sems,
                label=label, color=color,
                marker="o", linewidth=2, capsize=4,
            )

    ax.axvline(landmark_year, color="gray", linestyle="--", alpha=0.5,
               label=f"Landmark yr{landmark_year}")
    ax.set_xlabel("Year Postoperative")
    ax.set_ylabel(ylabel)
    ax.set_title(f"BMI ≤ {int(threshold)} milestone (landmark yr{landmark_year})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    print(f"Loading data from {DATA} ...")
    df = load(str(DATA))
    print(f"  {len(df)} patients loaded.\n")

    df = derive_postop_bmi(df)
    ARTIFACTS.mkdir(exist_ok=True)

    # Figure layout: rows = thresholds × landmark_years, cols = TBWL + FML
    n_combos = len(THRESHOLDS) * len(LANDMARK_YEARS)
    fig, axes = plt.subplots(n_combos, 2, figsize=(14, 5 * n_combos))
    if n_combos == 1:
        axes = [axes]
    axes = list(axes)

    plot_idx = 0

    for threshold in THRESHOLDS:
        df = flag_milestone(df, threshold)
        flag_col = f"ever_bmi{int(threshold)}"
        total_milestone = df[flag_col].sum()
        print(f"\nBMI ≤ {int(threshold)} milestone: {total_milestone}/{len(df)} patients "
              f"({100*total_milestone/len(df):.1f}%)")

        for landmark_year in LANDMARK_YEARS:
            group_a, group_b = landmark_split(df, landmark_year, threshold)
            follow_up_years = list(range(landmark_year + 1, 7))

            print(f"\n  Landmark yr{landmark_year}: Group A (milestone) n={len(group_a)}, "
                  f"Group B (control) n={len(group_b)}")

            # Q1a: TBWL trajectories
            tbwl_results = compare_trajectories(group_a, group_b, follow_up_years)
            _print_table(
                f"Q1a — TBWL% post-landmark | BMI≤{int(threshold)} | landmark yr{landmark_year}",
                tbwl_results, follow_up_years,
            )
            if tbwl_results.get("lme_summary"):
                print("\nLME (tbwl ~ year * milestone_group, random intercept per patient):")
                print(tbwl_results["lme_summary"][:800])  # truncate long output

            # Q1b: FML% trajectories (years where FML is amber or green)
            fml_years = [yr for yr in follow_up_years if yr in [2, 3, 4]]
            fml_results = compare_fml_trajectories(group_a, group_b, fml_years)
            _print_table(
                f"Q1b — FML%  post-landmark | BMI≤{int(threshold)} | landmark yr{landmark_year}",
                fml_results, fml_years,
            )

            # Plots
            ax_row = axes[plot_idx]
            _plot_trajectory_comparison(
                group_a, group_b, follow_up_years, threshold, landmark_year,
                ax_row[0], TBWL_BY_YEAR, "TBWL%",
            )
            _plot_trajectory_comparison(
                group_a, group_b, fml_years, threshold, landmark_year,
                ax_row[1], FML_BY_YEAR, "FML%",
            )
            plot_idx += 1

    fig.suptitle(
        "Post-landmark Trajectories by BMI Milestone Achievement\n"
        "(Mean ± SEM; Group A = reached milestone, Group B = did not)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out_path = ARTIFACTS / "bmi_milestone_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
