"""What-if / counterfactual analysis.

Re-runs predict_trajectory with one or more baseline features overridden and
returns the delta vs. the original prediction.

Example usage:
    what_if_analysis(patient, {'Preop_TBWL': 11.5})
    # Returns delta: if she had reached 11.5% instead of 7.8%...
"""
from src.predict import predict_trajectory
from src.config import MODEL_PERFORMANCE


def what_if_analysis(patient: dict, modified_features: dict) -> dict:
    """Return original, modified trajectory and per-year deltas.

    Args:
        patient: Original 15-feature baseline dict.
        modified_features: Feature overrides (keys must be valid BASELINE_FEATURES).

    Returns:
        {
          "original": {...},        # predict_trajectory(patient)
          "modified": {...},        # predict_trajectory(modified_patient)
          "deltas": {
            "TBWL": {year: {"delta_point": float|None, "description": str}, ...},
            "FML":  {...},
          },
          "modified_features": modified_features,
        }

    Delta is None for red years (no point prediction in original or modified).
    Original patient dict is never mutated.
    """
    modified_patient = {**patient, **modified_features}

    orig = predict_trajectory(patient)
    modified = predict_trajectory(modified_patient)

    deltas: dict = {}
    for outcome in ["TBWL", "FML"]:
        deltas[outcome] = {}
        for year in range(1, 7):
            o_pt = orig[outcome][year]["point"]
            m_pt = modified[outcome][year]["point"]
            if o_pt is None or m_pt is None:
                delta_val = None
                desc = f"Delta unavailable — {outcome} yr{year} is RED tier (unreliable)."
            else:
                delta_val = round(m_pt - o_pt, 2)
                direction = "higher" if delta_val > 0 else ("lower" if delta_val < 0 else "unchanged")
                desc = (
                    f"Modified {outcome} yr{year}: {m_pt:.1f}% vs original {o_pt:.1f}% "
                    f"({abs(delta_val):.1f} pp {direction})."
                )
            deltas[outcome][year] = {"delta_point": delta_val, "description": desc}

    return {
        "original": orig,
        "modified": modified,
        "deltas": deltas,
        "modified_features": modified_features,
    }
