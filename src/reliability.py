"""Reliability gating derived from Supplementary Table S5.

The tool must never present a year's prediction without its tier, and must refuse
point predictions in RED years (route to phenotype/threshold reasoning instead).
"""
from src.config import MODEL_PERFORMANCE, GREEN_MIN, AMBER_MIN


def tier(outcome: str, year: int) -> str:
    """Return 'green' | 'amber' | 'red' for an outcome ('TBWL'|'FML') and year 1-6."""
    r2 = MODEL_PERFORMANCE[outcome][year]["r2"]
    if r2 >= GREEN_MIN:
        return "green"
    if r2 >= AMBER_MIN:
        return "amber"
    return "red"


def gate(outcome: str, year: int) -> dict:
    """Full gating record for surfacing in the UI / agent output."""
    perf = MODEL_PERFORMANCE[outcome][year]
    t = tier(outcome, year)
    messages = {
        "green": "Usable — moderate predictive performance.",
        "amber": "Weak — present with wide uncertainty; do not over-anchor.",
        "red": "Unreliable — do NOT give a point prediction; reason via phenotype "
               "and preop-risk instead.",
    }
    return {
        "outcome": outcome, "year": year, "tier": t,
        "r2": perf["r2"], "rmse": perf["rmse"],
        "allow_point_prediction": t != "red",
        "message": messages[t],
    }
