"""Central config: paths, feature groups, S5 performance table, constants.

Keep data-dependent facts here so modules and Claude Code sessions share one source
of truth. Values in MODEL_PERFORMANCE are transcribed from Supplementary Table S5
(best model per outcome/year) and drive the reliability gating.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "patients-data5b-may11data-merge-edited.csv"
ARTIFACTS = ROOT / "artifacts"

# --- Reproducibility: single source of truth for the seed ---
# Every stochastic estimator, split, and dimensionality reducer (incl. any future
# UMAP/t-SNE added during clustering convergence with 1A) MUST use this. Import it
# rather than re-declaring a local 42, so "random_state=42 everywhere" is guaranteed.
SEED = 42
RANDOM_STATE = SEED

# --- Clinical constant ---
PREOP_TBWL_THRESHOLD = 10.5  # % ; below this = actionable risk flag

# --- Feature groups (extend as needed; verify against the CSV, don't guess) ---
ID_COL = "ID"  # study id, NOT an MRN
BASELINE_FEATURES = [
    "Age", "Sex", "Race", "Height", "Initial_BMI", "Initial_Weight",
    "Initial_BMR", "Initial_VF", "Initial_FATpct", "Initial_FATMASS",
    "Initial_FFM", "Time_to_Surgery", "Surgery_Type", "Preop_BMI", "Preop_TBWL",
]
# Column names use the shared 1A/1B cleaned convention (src.data_load.load applies
# it): "%"→"pct", "-"/" "→"_", collapsed underscores. So FML% → FMLpct.
TBWL_BY_YEAR = {y: f"{y}yr_Postop_TBWL" for y in range(1, 7)}
FML_BY_YEAR = {y: f"{y}yr_Postop_FMLpct" for y in range(1, 7)}

# Lagged features used by reproduce_models.py for years 2-6.
# Only the single most-recent prior year is used (one-lag model). This keeps
# complete-case N viable for later years (vs. all-lags which collapses N to <20
# for yr5-6). The immediately preceding TBWL/FML% is by far the strongest
# predictor anyway; adding older lags adds overfitting risk at small N.
# At inference, predict_trajectory cascades: yr2 uses predicted yr1, etc.
# When actual postop data is available, pass it via the `postop_tbwl` argument.
LAGGED_TBWL_BY_YEAR: dict[int, list[str]] = {
    1: [],
    2: ["1yr_Postop_TBWL"],
    3: ["2yr_Postop_TBWL"],
    4: ["3yr_Postop_TBWL"],
    5: ["4yr_Postop_TBWL"],
    6: ["5yr_Postop_TBWL"],
}
LAGGED_FML_BY_YEAR: dict[int, list[str]] = {
    1: [],
    2: ["1yr_Postop_FMLpct"],
    3: ["2yr_Postop_FMLpct"],
    4: ["3yr_Postop_FMLpct"],
    5: ["4yr_Postop_FMLpct"],
    6: ["5yr_Postop_FMLpct"],
}

# --- S5: best-model R² (and RMSE) per outcome/year. Source of the gating. ---
# tier thresholds: green R² >= 0.40 ; amber 0.20-0.40 ; red < 0.20
MODEL_PERFORMANCE = {
    "TBWL": {
        1: {"best_model": "RandomForest",     "r2": 0.262, "rmse": 8.29},
        2: {"best_model": "RandomForest",     "r2": 0.524, "rmse": 7.88},
        3: {"best_model": "GradientBoosting", "r2": 0.522, "rmse": 8.27},
        4: {"best_model": "GradientBoosting", "r2": 0.459, "rmse": 8.73},
        5: {"best_model": "SVR",              "r2": 0.259, "rmse": 11.70},
        6: {"best_model": "GradientBoosting", "r2": 0.102, "rmse": 9.83},
    },
    "FML": {
        1: {"best_model": "RandomForest",     "r2": 0.175, "rmse": 13.82},
        2: {"best_model": "RandomForest",     "r2": 0.479, "rmse": 13.64},
        3: {"best_model": "SVR",              "r2": 0.300, "rmse": 19.44},
        4: {"best_model": "XGBoost",          "r2": 0.379, "rmse": 15.01},
        5: {"best_model": "RandomForest",     "r2": 0.000, "rmse": 30.01},
        6: {"best_model": "GradientBoosting", "r2": 0.000, "rmse": 12.21},
    },
}

GREEN_MIN, AMBER_MIN = 0.40, 0.20
