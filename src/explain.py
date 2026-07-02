"""SHAP-based feature importance for trajectory predictions.

Uses TreeExplainer for tree-based models (RF, GB, XGBoost) and
KernelExplainer for SVR (uses a 50-row background subsample saved during
reproduce_models.py to keep inference feasible).

KernelExplainer is slow (~minutes). Gate it behind a UI button, not auto-compute.
The explainer objects are cached at module level after first construction.
"""
import joblib
import numpy as np
import shap

from src.config import ARTIFACTS, BASELINE_FEATURES, MODEL_PERFORMANCE
from src.reliability import gate
from src.preprocess import encode_patient, get_numeric_cols

_CACHE: dict = {}
_NUMERIC_COLS = get_numeric_cols(BASELINE_FEATURES)

_SVR_OUTCOMES = {
    (outcome, year)
    for outcome, years in MODEL_PERFORMANCE.items()
    for year, m in years.items()
    if m["best_model"] == "SVR"
}


def _load(filename: str):
    if filename not in _CACHE:
        path = ARTIFACTS / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run `python scripts/reproduce_models.py` first."
            )
        _CACHE[filename] = joblib.load(path)
    return _CACHE[filename]


def _humanize(col_name: str) -> str:
    """Convert OHE column names to readable labels: 'Sex_Female' → 'Sex (Female)'."""
    for cat in ("Sex", "Race", "Surgery_Type"):
        prefix = cat + "_"
        if col_name.startswith(prefix):
            level = col_name[len(prefix):].replace("_", " ")
            return f"{cat} ({level})"
    return col_name


def explain_with_shap(
    patient: dict,
    outcome: str,
    year: int,
    top_n: int = 5,
) -> dict:
    """Return top SHAP drivers for a single patient prediction.

    Raises ValueError if the requested (outcome, year) is a red tier —
    point predictions are refused there and explanations are meaningless.
    """
    gate_info = gate(outcome, year)
    if not gate_info["allow_point_prediction"]:
        raise ValueError(
            f"Cannot explain {outcome} yr{year}: reliability tier is RED. "
            "Use phenotype and preop-risk reasoning instead."
        )

    meta = _load(f"{outcome}_yr{year}_meta.joblib")
    model = _load(f"{outcome}_yr{year}_model.joblib")

    # Encode and impute (mirrors predict_trajectory exactly)
    x = encode_patient(patient, meta["columns"])
    num_idx = [meta["columns"].index(c) for c in meta["columns"] if c in _NUMERIC_COLS]
    if num_idx:
        x[:, num_idx] = meta["imputer"].transform(x[:, num_idx])

    is_svr = (outcome, year) in _SVR_OUTCOMES
    if is_svr:
        scaler = _load(f"{outcome}_yr{year}_scaler.joblib")
        x_for_shap = scaler.transform(x)
    else:
        x_for_shap = x

    # Build or retrieve cached explainer
    explainer_key = f"explainer_{outcome}_yr{year}"
    if explainer_key not in _CACHE:
        if is_svr:
            background = _load(f"{outcome}_yr{year}_background.joblib")
            _CACHE[explainer_key] = shap.KernelExplainer(model.predict, background)
        else:
            _CACHE[explainer_key] = shap.TreeExplainer(model)
    explainer = _CACHE[explainer_key]

    shap_vals = explainer.shap_values(x_for_shap)
    shap_vec = np.asarray(shap_vals).flatten()
    feature_names = meta["columns"]

    # Sort by magnitude and build driver list
    order = np.argsort(np.abs(shap_vec))[::-1]
    all_drivers = [
        {
            "feature": _humanize(feature_names[i]),
            "shap_value": round(float(shap_vec[i]), 4),
            "magnitude": round(float(abs(shap_vec[i])), 4),
            "direction": "positive" if shap_vec[i] >= 0 else "negative",
        }
        for i in order
    ]

    base_value = float(
        explainer.expected_value
        if not isinstance(explainer.expected_value, np.ndarray)
        else explainer.expected_value[0]
    )

    return {
        "outcome": outcome,
        "year": year,
        "tier": gate_info["tier"],
        "base_value": round(base_value, 2),
        "top_positive": [d for d in all_drivers if d["direction"] == "positive"][:top_n],
        "top_negative": [d for d in all_drivers if d["direction"] == "negative"][:top_n],
    }
