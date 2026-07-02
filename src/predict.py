"""Per-year trajectory prediction using frozen artifacts from reproduce_models.py.

Contract (stable — do not change without updating callers and CLAUDE.md):

    predict_trajectory(patient: dict, postop_tbwl: dict | None = None,
                       postop_fml: dict | None = None) -> dict
        {
          "TBWL": {year: {"point": float|None, "lo": float|None, "hi": float|None, **gate}},
          "FML":  {year: {...}},
        }

    - point=None, lo=None, hi=None for RED years — never fabricate a number.
    - Uncertainty band: ±1.96 × S5 RMSE (approximate 95% interval).
    - Models were trained with lagged prior-year TBWL/FML as additional features.
      At inference time the tool cascades: yr1 is predicted from baseline only;
      yr2 uses the predicted yr1; yr3 uses predicted yr1+yr2; etc.
      If actual postop measurements are available (postop_tbwl / postop_fml dicts
      mapping year → measured value), those override the cascade for that year.
    - Loads per-year models from artifacts/ on first call (cached in _MODEL_CACHE).
    - Raises FileNotFoundError with a clear message if artifacts are missing.

Cascade example (preop, no postop data):
    traj = predict_trajectory(patient)

Updated prediction when yr1 actual is available:
    traj = predict_trajectory(patient, postop_tbwl={1: 22.5})
"""
import joblib

from src.config import (
    ARTIFACTS, BASELINE_FEATURES, MODEL_PERFORMANCE,
    LAGGED_TBWL_BY_YEAR, LAGGED_FML_BY_YEAR,
    TBWL_BY_YEAR, FML_BY_YEAR,
)
from src.reliability import gate
from src.preprocess import encode_patient, get_numeric_cols

_MODEL_CACHE: dict = {}

_NUMERIC_COLS = get_numeric_cols(BASELINE_FEATURES)

_SVR_OUTCOMES = {
    (outcome, year)
    for outcome, years in MODEL_PERFORMANCE.items()
    for year, m in years.items()
    if m["best_model"] == "SVR"
}


def _load_artifact(filename: str):
    if filename not in _MODEL_CACHE:
        path = ARTIFACTS / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run `python scripts/reproduce_models.py` first to train and persist models."
            )
        _MODEL_CACHE[filename] = joblib.load(path)
    return _MODEL_CACHE[filename]


def predict_trajectory(
    patient: dict,
    postop_tbwl: dict | None = None,
    postop_fml: dict | None = None,
) -> dict:
    """Predict TBWL% and FML% trajectories for years 1-6 with reliability gating.

    Args:
        patient: 15-feature baseline dict (BASELINE_FEATURES).
        postop_tbwl: Optional {year: measured_tbwl_pct} for years where actual
                     postop data exists. Overrides cascaded predictions for those years.
        postop_fml:  Optional {year: measured_fml_pct} — same for FML.
    """
    postop_tbwl = postop_tbwl or {}
    postop_fml = postop_fml or {}

    result: dict = {}

    # Must predict TBWL first (FML cascade depends on FML, TBWL cascade on TBWL)
    for outcome, col_map, lagged_map, actual_map in [
        ("TBWL", TBWL_BY_YEAR, LAGGED_TBWL_BY_YEAR, postop_tbwl),
        ("FML",  FML_BY_YEAR,  LAGGED_FML_BY_YEAR,  postop_fml),
    ]:
        result[outcome] = {}
        outcome_cascade: dict[str, float] = {}

        for year in range(1, 7):
            gate_info = gate(outcome, year)

            if not gate_info["allow_point_prediction"]:
                result[outcome][year] = {"point": None, "lo": None, "hi": None, **gate_info}
                # Still populate cascade from actual data if available
                if year in actual_map:
                    outcome_cascade[col_map[year]] = actual_map[year]
                continue

            meta = _load_artifact(f"{outcome}_yr{year}_meta.joblib")
            model = _load_artifact(f"{outcome}_yr{year}_model.joblib")

            # Build patient dict with lagged features injected
            lagged_cols = meta.get("lagged_cols", [])
            patient_with_lags = dict(patient)
            for lag_col in lagged_cols:
                if lag_col in outcome_cascade:
                    patient_with_lags[lag_col] = outcome_cascade[lag_col]
                else:
                    # Missing lag — imputer will handle (median substitution)
                    patient_with_lags[lag_col] = float("nan")

            # Encode and align to training column order
            x = encode_patient(patient_with_lags, meta["columns"])

            # Apply saved imputer (transform only — never fit at inference)
            all_numeric = _NUMERIC_COLS + lagged_cols
            num_col_indices = [
                meta["columns"].index(c)
                for c in meta["columns"]
                if c in all_numeric
            ]
            if num_col_indices:
                x[:, num_col_indices] = meta["imputer"].transform(
                    x[:, num_col_indices]
                )

            # Scale for SVR models
            if (outcome, year) in _SVR_OUTCOMES:
                scaler = _load_artifact(f"{outcome}_yr{year}_scaler.joblib")
                x = scaler.transform(x)

            point = float(model.predict(x)[0])
            rmse = MODEL_PERFORMANCE[outcome][year]["rmse"]
            half = 1.96 * rmse

            result[outcome][year] = {
                "point": round(point, 2),
                "lo": round(point - half, 2),
                "hi": round(point + half, 2),
                **gate_info,
            }

            # Update cascade: prefer actual postop data over prediction
            cascade_val = actual_map.get(year, point)
            outcome_cascade[col_map[year]] = cascade_val

    return result
