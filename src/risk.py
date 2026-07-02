"""Preop weight-loss risk flag. This is a documented rule, not a model."""
from src.config import PREOP_TBWL_THRESHOLD


def assess_preop_risk(preop_tbwl_pct: float) -> dict:
    """Flag whether a patient is below the actionable 10.5% PreopTBWL% threshold."""
    if preop_tbwl_pct is None:
        return {"flag": None, "message": "PreopTBWL% not provided."}
    gap = preop_tbwl_pct - PREOP_TBWL_THRESHOLD
    below = gap < 0
    return {
        "preop_tbwl_pct": preop_tbwl_pct,
        "threshold": PREOP_TBWL_THRESHOLD,
        "below_threshold": below,
        "gap": round(gap, 1),
        "flag": "at_risk" if below else "on_track",
        "message": (
            f"{abs(gap):.1f} pts below the {PREOP_TBWL_THRESHOLD}% threshold — "
            "consider extra preoperative support."
            if below else
            f"{gap:.1f} pts above the {PREOP_TBWL_THRESHOLD}% threshold."
        ),
    }
