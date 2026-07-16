"""Quiet Regainer flag — a high-alert sub-phenotype for the surgeon-facing tool.

Motivation (Ioanna, 2026-07-16): the trajectory-shape-only sensitivity clustering
surfaced one group that loses strongly early and then quietly regains by year 4 —
"the patient you'd never flag from their chart." She asked us to position it as a
high-alert sub-phenotype in the app.

Rather than ship an opaque cluster assignment, we express it as a transparent,
clinician-legible rule on the predicted trajectory, calibrated to that cluster:

    QUIET REGAINER  ⇔  peak(yr1, yr2) TBWL% ≥ 34%   AND   peak − predicted yr4 TBWL% ≥ 12 pp

i.e. a strong early responder (top of the distribution) who then gives back a large
share of that loss by year 4. Calibrated against the trajectory-only cluster: this
rule recovers ~56% of it at ~70% precision, and flags ~21% of the cohort. The flagged
group's mean predicted trajectory is 39 → 31 → 28 → 24 (clear early-loss-then-regain)
vs 32 → 29 → 27 → 25 for everyone else.

The flag needs green/amber-tier predictions at years 1, 2 and 4 (all non-red under RF).
If any is missing it returns detected=False with a reason.
"""
from __future__ import annotations

PEAK_MIN = 34.0     # % TBWL — "strong early responder" floor (max of yr1, yr2)
DROP_MIN = 12.0     # pp — regain from early peak to year 4 to count as "quiet regain"


def flag_quiet_regainer(traj: dict) -> dict:
    """Return the Quiet-Regainer assessment for one predicted trajectory.

    Args:
        traj: output of predict_trajectory(patient) — traj["TBWL"][year]["point"].

    Returns:
        {
          "detected": bool,
          "peak": float|None,        # max(yr1, yr2) predicted TBWL%
          "yr4": float|None,         # predicted yr4 TBWL%
          "drop": float|None,        # peak - yr4 (pp)
          "reason": str,             # plain-English explanation
        }
    """
    tb = traj.get("TBWL", {})
    y1 = tb.get(1, {}).get("point")
    y2 = tb.get(2, {}).get("point")
    y4 = tb.get(4, {}).get("point")

    if y1 is None or y2 is None or y4 is None:
        return {"detected": False, "peak": None, "yr4": None, "drop": None,
                "reason": "Not assessable — needs reliable year 1, 2 and 4 predictions."}

    peak = max(y1, y2)
    drop = peak - y4
    detected = (peak >= PEAK_MIN) and (drop >= DROP_MIN)

    if detected:
        reason = (
            f"Strong early loss (peaks at {peak:.0f}% by year 1–2) followed by a "
            f"{drop:.0f} percentage-point regain by year 4 (down to {y4:.0f}%). This is "
            f"the 'quiet regainer' pattern — a patient who looks like a success early and "
            f"gives much of it back. Consider proactive year 2–3 follow-up."
        )
    elif peak >= PEAK_MIN:
        reason = (f"Strong early responder ({peak:.0f}% by year 1–2), and holds it "
                  f"(only {drop:.0f} pp lost by year 4). Not a quiet regainer.")
    else:
        reason = (f"Early loss peaks at {peak:.0f}% — below the {PEAK_MIN:.0f}% "
                  f"strong-responder threshold, so the quiet-regainer pattern does not apply.")

    return {"detected": detected, "peak": round(peak, 1), "yr4": round(y4, 1),
            "drop": round(drop, 1), "reason": reason}
