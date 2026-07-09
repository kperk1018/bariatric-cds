"""Bariatric CDS — Streamlit MVP (Stage 3).

Research / quality-improvement prototype. Not a medical device. Not patient-facing.
Every output states uncertainty and defers to the care team.

Run:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from datetime import datetime

from src.predict import predict_trajectory
from src.phenotype import assign_phenotype
from src.risk import assess_preop_risk
from src.config import MODEL_PERFORMANCE, ARTIFACTS, TBWL_BY_YEAR
from src.what_if import what_if_analysis

st.set_page_config(page_title="Bariatric CDS", layout="wide")

# ── colour map for reliability tiers ──────────────────────────────────────────
TIER_COLOR = {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c"}

# ── provisional phenotype descriptions (k is data-derived, so labels are dynamic) ─
def phenotype_label(rank: int, k: int) -> tuple[str, str]:
    """(short, description) for a cluster. Clusters are ordered 0..k-1 by ascending
    year-3 TBWL%, so 0 = least weight lost, k-1 = most."""
    if k <= 1:
        return "single group", ""
    if rank == 0:
        short = "lowest-loss responders"
    elif rank == k - 1:
        short = "highest-loss responders"
    else:
        short = f"mid-loss responders (tier {rank + 1} of {k})"
    desc = (f"Cluster {rank} of {k}, ordered by ascending year-3 TBWL% "
            f"(0 = least weight lost, {k - 1} = most).")
    return short, desc

# ── comorbidity remission reference table (population-level literature) ────────
COMORBIDITY_REMISSION = {
    "Type 2 Diabetes (DM)": {
        ">25% TBWL": "60-80%", "20-25% TBWL": "45-65%",
        "15-20% TBWL": "30-50%", "<15% TBWL": "15-30%",
        "source": "Schauer 2017 STAMPEDE",
    },
    "Hypertension (HTN)": {
        ">25% TBWL": "40-60%", "20-25% TBWL": "30-50%",
        "15-20% TBWL": "20-40%", "<15% TBWL": "10-25%",
        "source": "Sjoestroem 2004 SOS",
    },
    "Hyperlipidemia (HLD)": {
        ">25% TBWL": "50-70%", "20-25% TBWL": "40-60%",
        "15-20% TBWL": "25-45%", "<15% TBWL": "10-20%",
        "source": "Sjoestroem 2004 SOS",
    },
    "Sleep Apnea (OSA)": {
        ">25% TBWL": "55-75%", "20-25% TBWL": "40-60%",
        "15-20% TBWL": "25-45%", "<15% TBWL": "10-25%",
        "source": "Buchwald 2004 meta-analysis",
    },
    "GERD": {
        ">25% TBWL": "50-70% (Bypass); variable (Sleeve)",
        "20-25% TBWL": "35-55%", "15-20% TBWL": "20-40%", "<15% TBWL": "10-20%",
        "source": "Oor 2016 systematic review",
    },
    "Depression": {
        ">25% TBWL": "40-55%", "20-25% TBWL": "30-45%",
        "15-20% TBWL": "20-35%", "<15% TBWL": "10-20%",
        "source": "Dawes 2016 systematic review",
    },
}

# ── cached cohort data (features 3 and 6) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_cohort():
    try:
        from src.data_load import load
        return load()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _tbwl_distributions():
    df = _load_cohort()
    if df is None:
        return None
    dists = {}
    for yr, col in TBWL_BY_YEAR.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(vals) >= 10:
                dists[yr] = vals
    return dists


@st.cache_data(show_spinner=False)
def _cluster_trajectories():
    """Per-cluster actual-TBWL trajectory means, read from the fitted phenotype
    bundle (computed at fit time in src.phenotype.fit_phenotypes)."""
    import joblib
    pheno_path = ARTIFACTS / "phenotype_kmeans.joblib"
    if not pheno_path.exists():
        return None
    try:
        bundle = joblib.load(pheno_path)
        return bundle.get("cluster_actual_traj")
    except Exception:
        return None


# ── helper: generate HTML summary card (feature 8) ────────────────────────────
def _generate_summary_html(patient, traj, pheno_id, risk, shap_result=None,
                           pheno_short="", pheno_desc=""):
    today = datetime.now().strftime("%Y-%m-%d")
    tbwl_rows = ""
    for yr in [1, 2, 4]:
        d = traj["TBWL"][yr]
        if d["point"] is not None:
            tbwl_rows += f"<tr><td>Year {yr}</td><td>{d['point']:.1f}%</td><td>{d['tier'].upper()}</td></tr>"
        else:
            tbwl_rows += f"<tr><td>Year {yr}</td><td>&mdash;</td><td>UNRELIABLE</td></tr>"

    risk_icon = "At risk (below 10.5% threshold)" if risk["flag"] == "at_risk" else "On track (above 10.5% threshold)"
    pheno_text = ""
    if pheno_id is not None:
        pheno_text = (
            f"<h2>Trajectory Pattern</h2>"
            f"<p>This patient's predicted trajectory most closely resembles the "
            f"<b>{pheno_short or f'cluster {pheno_id}'}</b> group. {pheno_desc}</p>"
        )
    shap_text = ""
    if shap_result:
        top3 = shap_result.get("top_positive", [])[:3]
        if top3:
            shap_text = (
                f"<h2>Top Predictive Drivers</h2>"
                f"<p>{', '.join(d['feature'] for d in top3)}</p>"
            )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Bariatric Surgery Trajectory Summary</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:700px;margin:40px auto;color:#222;line-height:1.5}}
  h1{{color:#2c3e50;border-bottom:2px solid #eee;padding-bottom:8px}}
  h2{{color:#34495e;margin-top:24px}}
  table{{border-collapse:collapse;width:100%;margin:12px 0}}
  th,td{{border:1px solid #ddd;padding:10px;text-align:left}}
  th{{background:#f4f4f4}}
  .disclaimer{{font-size:11px;color:#888;border-top:1px solid #eee;margin-top:30px;padding-top:10px}}
</style>
</head>
<body>
<h1>Bariatric Surgery Trajectory Summary</h1>
<p><b>Generated:</b> {today} &nbsp;|&nbsp; <b>Surgery type:</b> {patient['Surgery_Type']}
&nbsp;|&nbsp; <b>Initial BMI:</b> {patient['Initial_BMI']:.1f}
&nbsp;|&nbsp; <b>Age:</b> {patient['Age']}</p>

<h2>Predicted Weight Loss</h2>
<table>
<tr><th>Year</th><th>TBWL%</th><th>Reliability</th></tr>
{tbwl_rows}
</table>

<h2>Preoperative Risk Flag</h2>
<p>{risk_icon} &mdash; Preop TBWL: {risk['preop_tbwl_pct']:.1f}% (threshold: 10.5%)</p>

{pheno_text}
{shap_text}

<div class="disclaimer">
  <b>Research / quality-improvement use only.</b> Not a medical device. Not patient-facing.
  All predictions carry uncertainty. Defer to the care team for clinical decisions.
  No patient identifiers are included in this document.
</div>
</body>
</html>"""


# ── sidebar — patient inputs ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Patient Features")
    st.caption("Enter baseline values. All fields required.")

    age             = st.number_input("Age (years)",            min_value=18,    max_value=85,    value=43)
    height          = st.number_input("Height (inches)",        min_value=48,    max_value=84,    value=64)
    initial_bmi     = st.number_input("Initial BMI",            min_value=18.0,  max_value=95.0,  value=43.0,  step=0.1)
    initial_wt      = st.number_input("Initial Weight (lbs)",   min_value=100.0, max_value=550.0, value=255.0, step=0.1)
    initial_bmr     = st.number_input("Initial BMR (kcal/day)", min_value=500,   max_value=4000,  value=1957)
    initial_vf      = st.number_input("Initial Visceral Fat",   min_value=0.0,   max_value=50.0,  value=14.0,  step=0.1)
    initial_fatpct  = st.number_input("Initial FAT%",           min_value=15.0,  max_value=72.0,  value=45.0,  step=0.1)
    initial_fatmass = st.number_input("Initial Fat Mass (lbs)", min_value=20.0,  max_value=320.0, value=115.0, step=0.1)
    initial_ffm     = st.number_input("Initial FFM (lbs)",      min_value=40.0,  max_value=240.0, value=137.0, step=0.1)
    time_to_surg    = st.number_input("Time to Surgery (days)", min_value=1,     max_value=1500,  value=135)
    preop_bmi       = st.number_input("Preop BMI",              min_value=25.0,  max_value=70.0,  value=38.5,  step=0.1)
    preop_tbwl      = st.number_input("Preop TBWL%",            min_value=-10.0, max_value=35.0,  value=11.5,  step=0.1)

    sex          = st.selectbox("Sex",          ["Female", "Male"])
    race         = st.selectbox("Race",         ["White", "Hispanic", "African_American", "Asian", "Other"])
    surgery_type = st.selectbox("Surgery Type", ["Sleeve", "Bypass", "Revision"])

    st.divider()
    comorbidities = st.multiselect(
        "Comorbidities",
        list(COMORBIDITY_REMISSION.keys()),
        default=[],
        help="Used by the prediction models (years 2-3) and the remission reference panel. "
             "Leave empty if unknown.",
    )

    with st.expander("Optional preop labs & scores (improves preop accuracy)"):
        st.caption(
            "The preop models for years 2-3 were trained with these. "
            "Leave blank if unavailable — blanks are median-imputed."
        )
        lab_hba1c   = st.number_input("HbA1c (%)",            min_value=3.5,  max_value=15.0,   value=None, step=0.1,  key="lab_hba1c")
        lab_glucose = st.number_input("Glucose (mg/dL)",      min_value=40.0, max_value=500.0,  value=None, step=1.0,  key="lab_glucose")
        lab_insulin = st.number_input("Insulin (uIU/mL)",     min_value=0.0,  max_value=300.0,  value=None, step=0.1,  key="lab_insulin")
        lab_tg      = st.number_input("Triglycerides (mg/dL)",min_value=20.0, max_value=1500.0, value=None, step=1.0,  key="lab_tg")
        lab_hdl     = st.number_input("HDL (mg/dL)",          min_value=10.0, max_value=150.0,  value=None, step=1.0,  key="lab_hdl")
        lab_crp     = st.number_input("CRP (mg/dL)",          min_value=0.0,  max_value=50.0,   value=None, step=0.1,  key="lab_crp")
        lab_alt     = st.number_input("ALT (U/L)",            min_value=1.0,  max_value=500.0,  value=None, step=1.0,  key="lab_alt")
        score_bes   = st.number_input("BES score (binge eating, 0-46)", min_value=0.0, max_value=46.0,  value=None, step=1.0, key="score_bes")
        score_iwqol = st.number_input("IWQoL score",          min_value=0.0,  max_value=155.0,  value=None, step=1.0,  key="score_iwqol")
        score_epw   = st.number_input("Epworth score (0-24)", min_value=0.0,  max_value=24.0,   value=None, step=1.0,  key="score_epw")
        preop_visits = st.number_input("Preop clinic visits", min_value=0.0,  max_value=40.0,   value=None, step=1.0,  key="preop_visits")

    # map optional inputs to the cleaned 1A/1B column names
    OPTIONAL_FEATURES = {
        "Preop_HbA1c": lab_hba1c, "Preop_Glucose": lab_glucose,
        "Preop_Insulin": lab_insulin, "Preop_TG": lab_tg,
        "Preop_HDL": lab_hdl, "Preop_CRP": lab_crp, "Preop_ALT": lab_alt,
        "BES_score": score_bes, "IWQoL_score": score_iwqol,
        "Epworth_score": score_epw, "Preop_Visits": preop_visits,
    }
    COMORB_TO_COL = {
        "Type 2 Diabetes (DM)": "DM", "Hypertension (HTN)": "HTN",
        "Hyperlipidemia (HLD)": "Hyperlipidemia", "Sleep Apnea (OSA)": "OSA",
        "GERD": "GERD", "Depression": "Depression",
    }

    with st.expander("Actual postop data (optional)"):
        st.caption(
            "Enter actual measured TBWL% at follow-up visits to refine later-year forecasts. "
            "Leave at 0 to skip a year."
        )
        p_yr1 = st.number_input("TBWL% at year 1 (0=skip)", min_value=0.0, max_value=60.0, value=0.0, step=0.1, key="p1")
        p_yr2 = st.number_input("TBWL% at year 2 (0=skip)", min_value=0.0, max_value=60.0, value=0.0, step=0.1, key="p2")
        p_yr3 = st.number_input("TBWL% at year 3 (0=skip)", min_value=0.0, max_value=60.0, value=0.0, step=0.1, key="p3")
        p_yr4 = st.number_input("TBWL% at year 4 (0=skip)", min_value=0.0, max_value=60.0, value=0.0, step=0.1, key="p4")
    postop_tbwl_input = {yr: v for yr, v in [(1, p_yr1), (2, p_yr2), (3, p_yr3), (4, p_yr4)] if v > 0}

    st.divider()
    compute_btn = st.button("Compute Trajectory", type="primary", use_container_width=True)

# ── main panel ────────────────────────────────────────────────────────────────
st.title("Bariatric Trajectory CDS — Research Prototype")
st.caption(
    "**Research / quality-improvement use only.** Not a medical device. "
    "All predictions carry uncertainty. Defer to the care team for clinical decisions."
)

# ── abbreviation glossary ─────────────────────────────────────────────────────
with st.expander("Abbreviations & definitions"):
    st.markdown("""
| Term | Full name | What it means |
|---|---|---|
| **TBWL%** | Total Body Weight Loss % | Percentage of the patient's starting weight lost since surgery. A 30% TBWL means someone who started at 250 lbs has lost 75 lbs. |
| **FML%** | Fat Mass Loss % | Percentage of the patient's initial fat mass lost. A higher FML% relative to TBWL% means preferential fat loss over lean tissue. |
| **BMI** | Body Mass Index | Weight (lbs) / Height (in)^2 x 703. Proxy for overall adiposity. |
| **BMR** | Basal Metabolic Rate | Calories burned at complete rest, estimated from body composition (kcal/day). |
| **VF** | Visceral Fat | Fat stored around internal organs. Higher visceral fat carries greater metabolic risk. |
| **FFM** | Fat-Free Mass | Lean body mass (muscle, bone, water, organs). Preserved FFM is a marker of healthy weight loss. |
| **FAT%** | Body Fat Percentage | Fat mass as a percentage of total body weight at the initial visit. |
| **Preop TBWL%** | Preoperative TBWL% | Weight lost between first clinic visit and surgery, expressed as % of initial body weight. |
| **R2** | R-squared | How much of the outcome variation the model explains (0 = nothing, 1 = perfect). Drives the green/amber/red reliability tier. |
| **SHAP** | SHapley Additive exPlanations | Shows how much each feature pushed this patient's prediction up or down vs. the average patient. |
| **Green tier** | R2 >= 0.40 | Model explains enough variation for a reliable point estimate. |
| **Amber tier** | R2 0.20-0.40 | Weak signal -- treat as a rough guide, not a precise number. |
| **Red tier** | R2 < 0.20 | Model cannot reliably predict this year. No point estimate shown. |
""")

# ── session state / compute ───────────────────────────────────────────────────
patient = {
    "Age": age, "Sex": sex, "Race": race, "Height": height,
    "Initial_BMI": initial_bmi, "Initial_Weight": initial_wt,
    "Initial_BMR": initial_bmr, "Initial_VF": initial_vf,
    "Initial_FATpct": initial_fatpct, "Initial_FATMASS": initial_fatmass,
    "Initial_FFM": initial_ffm, "Time_to_Surgery": time_to_surg,
    "Surgery_Type": surgery_type, "Preop_BMI": preop_bmi, "Preop_TBWL": preop_tbwl,
}
# optional labs/scores: only pass what was entered (blanks stay median-imputed)
patient.update({col: val for col, val in OPTIONAL_FEATURES.items() if val is not None})
# comorbidity flags: an empty selection means "unknown", not "none"
if comorbidities:
    patient.update({col: (1.0 if label in comorbidities else 0.0)
                    for label, col in COMORB_TO_COL.items()})

if compute_btn:
    with st.spinner("Computing trajectory..."):
        try:
            st.session_state["traj"] = predict_trajectory(
                patient, postop_tbwl=postop_tbwl_input or None
            )
            st.session_state["patient"] = patient
            for key in ("wi_result", "wi_modified", "surg_trajs", "shap_result", "shap_selected"):
                st.session_state.pop(key, None)
        except FileNotFoundError as exc:
            st.error(f"Model artifacts not found.\n\n{exc}")
            st.stop()

if "traj" not in st.session_state:
    st.info("Enter patient features in the sidebar and click **Compute Trajectory**.")
    st.stop()

traj = st.session_state["traj"]
patient = st.session_state["patient"]

if postop_tbwl_input:
    yr_list = ", ".join(f"yr{y}={v:.1f}%" for y, v in sorted(postop_tbwl_input.items()))
    st.success(
        f"Using actual postop measurements: {yr_list}. "
        "Later-year forecasts are conditioned on these values."
    )

tab_traj, tab_whatif, tab_pheno, tab_risk, tab_shap, tab_summary = st.tabs([
    "Trajectory", "What-If", "Phenotype", "Preop Risk", "SHAP Drivers", "Summary"
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Trajectory + BMI milestone + cohort percentile
# ─────────────────────────────────────────────────────────────────────────────
with tab_traj:
    st.subheader("Predicted Weight-Loss Trajectory")

    yr1 = traj["TBWL"][1]
    yr2 = traj["TBWL"][2]
    fml_yr2 = traj["FML"][2]
    reliable_tbwl_yrs = [yr for yr in range(1, 7) if traj["TBWL"][yr]["point"] is not None]
    tier_note = (
        f"years {min(reliable_tbwl_yrs)}-{max(reliable_tbwl_yrs)}"
        if reliable_tbwl_yrs else "no reliable years"
    )

    summary_parts = []
    if yr1["point"] is not None:
        lbs_lost = patient["Initial_Weight"] * (yr1["point"] / 100)
        summary_parts.append(
            f"Based on this patient's baseline profile, the model predicts a **{yr1['point']:.1f}% "
            f"total body weight loss (TBWL%) by year 1** -- approximately **{lbs_lost:.0f} lbs** "
            f"from their starting weight of {patient['Initial_Weight']:.0f} lbs."
        )
    if yr2["point"] is not None:
        direction = "continuing to decline" if yr2["point"] < (yr1["point"] or 0) else "plateauing"
        summary_parts.append(
            f"By year 2 (most reliably predicted), TBWL% is estimated at **{yr2['point']:.1f}%**, "
            f"{direction}."
        )
    if fml_yr2["point"] is not None:
        summary_parts.append(
            f"Fat mass loss at year 2 is predicted at **{fml_yr2['point']:.1f}%** of initial fat mass."
        )
    summary_parts.append(
        f"Reliable predictions (green/amber tier) available for {tier_note}; "
        "years outside this range are unreliable and no point estimate is shown."
    )
    st.info(" ".join(summary_parts))

    # BMI milestone from predicted TBWL
    predicted_bmi_by_yr = {}
    for yr in range(1, 7):
        pt = traj["TBWL"][yr]["point"]
        if pt is not None:
            predicted_bmi_by_yr[yr] = round(patient["Initial_BMI"] * (1 - pt / 100), 1)

    milestone_yr = next(
        (yr for yr in range(1, 7) if predicted_bmi_by_yr.get(yr, 999) <= 27), None
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (outcome, ylabel) in zip(axes, [("TBWL", "TBWL%"), ("FML", "FML%")]):
        for yr in range(1, 7):
            d = traj[outcome][yr]
            color = TIER_COLOR[d["tier"]]
            if d["point"] is not None:
                ax.errorbar(
                    yr, d["point"],
                    yerr=[[d["point"] - d["lo"]], [d["hi"] - d["point"]]],
                    fmt="o", color=color, capsize=5, markersize=7,
                )
                if outcome == "TBWL" and yr == milestone_yr:
                    ax.plot(yr, d["point"], "*", color="gold", markersize=18,
                            zorder=5, label="BMI <= 27 predicted")
            else:
                ax.plot(yr, 0, "x", color=color, markersize=10, markeredgewidth=2)
                ax.text(yr, 1.5, "UNRELIABLE\n(red tier)", ha="center", fontsize=7, color=color)

        pts = [(yr, traj[outcome][yr]["point"]) for yr in range(1, 7)
               if traj[outcome][yr]["point"] is not None]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    "--", color="gray", linewidth=1, alpha=0.5, zorder=0)

        ax.axhline(0, color="lightgray", linewidth=0.8)
        ax.set_xlabel("Year Postoperative")
        ax.set_ylabel(ylabel)
        ax.set_title(outcome)
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.5, 6.5)
        ax.grid(True, alpha=0.25)
        if outcome == "TBWL" and milestone_yr:
            ax.legend(fontsize=9)

    fig.suptitle("Predicted Trajectory with +/-1.96 x RMSE Uncertainty Bands", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    cols = st.columns(3)
    cols[0].markdown("Green -- reliable (R2 >= 0.40)")
    cols[1].markdown("Amber -- weak signal (R2 0.20-0.40)")
    cols[2].markdown("Red -- unreliable, no point prediction (R2 < 0.20)")

    # Honest-accuracy caveats from out-of-fold validation (scripts/validate_models.py
    # and scripts/train_direct_models.py)
    preop_years = [
        yr for yr in range(2, 7)
        if traj["TBWL"][yr].get("mode") in ("cascade", "direct")
        and traj["TBWL"][yr]["point"] is not None
    ]
    if preop_years:
        st.warning(
            "**Preop-mode accuracy caveat.** Without actual follow-up data, years 2+ use "
            "direct preop models (trained on baseline features, labs and behavioral scores; "
            "out-of-fold R2 ~0.17-0.26) — substantially weaker than the tier R2, which only "
            "applies when the prior year's actual value is known (e.g. TBWL yr2: R2 0.52 "
            "with actual yr1 vs 0.26 preop). Uncertainty bands are calibrated to actual "
            "out-of-fold errors — treat the band, not the point, as the prediction. "
            "Entering follow-up measurements in the sidebar restores full model accuracy "
            "for the following year, and entering preop labs/scores sharpens years 2-3."
        )
    if patient["Surgery_Type"] == "Bypass":
        st.caption(
            "Note: out-of-fold validation shows the model **under-predicts TBWL% for Bypass "
            "patients by ~4 pp** on average at years 2-4. The true trajectory is likely toward "
            "the upper half of the displayed band."
        )
    if any(traj["TBWL"][yr]["point"] is not None for yr in [5, 6]):
        st.caption(
            "Note: patients with year 5-6 follow-up were systematically lighter at baseline than "
            "the full cohort (attrition bias, |SMD| up to 0.39) — late-year estimates may not "
            "generalize to heavier patients."
        )

    # BMI milestone summary
    st.divider()
    st.subheader("BMI Milestone Prediction")
    if predicted_bmi_by_yr:
        if milestone_yr:
            bmi_at = predicted_bmi_by_yr[milestone_yr]
            st.success(
                f"This patient is **predicted to reach BMI <= 27 at year {milestone_yr}** "
                f"(predicted BMI: {bmi_at:.1f}). "
                f"Patients in our cohort who reached this milestone showed significantly "
                f"better sustained weight-loss trajectories through year 5."
            )
        else:
            min_bmi_yr = min(predicted_bmi_by_yr, key=predicted_bmi_by_yr.get)
            st.warning(
                f"This patient is **not predicted to reach BMI <= 27** within the reliable prediction window. "
                f"Minimum predicted BMI: **{predicted_bmi_by_yr[min_bmi_yr]:.1f}** at year {min_bmi_yr}."
            )
        bmi_rows = [
            {"Year": yr, "Predicted BMI": f"{bmi:.1f}", "<= 27?": "Yes" if bmi <= 27 else "--"}
            for yr, bmi in sorted(predicted_bmi_by_yr.items())
        ]
        st.dataframe(bmi_rows, hide_index=True)
    else:
        st.info("No reliable TBWL predictions available to compute BMI trajectory.")

    # Cohort percentile ranking
    st.divider()
    st.subheader("Cohort Percentile Ranking")
    dists = _tbwl_distributions()
    if dists is None:
        st.info(
            "CSV not found -- percentile ranking unavailable. "
            "Drop the data file into `data/` to enable this view."
        )
    else:
        from scipy.stats import percentileofscore
        pct_rows = []
        for yr in [1, 2, 3, 4]:
            pt = traj["TBWL"][yr]["point"]
            if pt is not None and yr in dists:
                pct = percentileofscore(dists[yr], pt, kind="rank")
                pct_rows.append({
                    "Year": yr,
                    "Predicted TBWL%": f"{pt:.1f}%",
                    "Cohort N": len(dists[yr]),
                    "Percentile": f"{pct:.0f}th",
                    "Vs. average": (
                        "above average" if pct >= 60 else
                        "average" if pct >= 40 else
                        "below average"
                    ),
                })
        if pct_rows:
            yr2_pct = next((r["Percentile"] for r in pct_rows if r["Year"] == 2), None)
            yr2_n = len(dists.get(2, []))
            if yr2_pct and traj["TBWL"][2]["point"] is not None:
                st.info(
                    f"At year 2, this patient is predicted at **{traj['TBWL'][2]['point']:.1f}% TBWL** -- "
                    f"approximately the **{yr2_pct} percentile** among {yr2_n} patients in this "
                    f"cohort with year-2 follow-up."
                )
            st.dataframe(pct_rows, hide_index=True)
        else:
            st.info("No non-red TBWL years overlap with available cohort data for percentile ranking.")

    # Detailed results table
    st.divider()
    st.caption("Detailed results")
    for outcome in ["TBWL", "FML"]:
        rows = []
        for yr in range(1, 7):
            d = traj[outcome][yr]
            mode = d.get("mode")
            eff_r2 = d.get("cascade_r2")
            rows.append({
                "Year": yr,
                "Tier": d["tier"].upper(),
                "Point": f"{d['point']:.1f}%" if d["point"] is not None else "--",
                "95% CI": f"[{d['lo']:.1f}, {d['hi']:.1f}]" if d["point"] is not None else "--",
                "R2 (S5)": d["r2"],
                "Mode": mode if mode else "--",
                "Effective R2": (
                    f"{eff_r2:.2f}" if eff_r2 is not None
                    else (f"{d['r2']:.2f}" if mode == "conditioned" else "--")
                ),
                "Band": {
                    "calibrated_oof": "calibrated (OOF cascade)",
                    "direct_oof": "calibrated (OOF direct)",
                    "s5_rmse": "1.96 x RMSE",
                }.get(d.get("band_source"), "--"),
                "Note": d["message"],
            })
        st.write(f"**{outcome}%**")
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: What-If (Scenario Planner + Surgery Type Comparison)
# ─────────────────────────────────────────────────────────────────────────────
with tab_whatif:
    st.subheader("Scenario Planner -- What-If Analysis")
    st.caption(
        "Adjust key baseline features to model a counterfactual scenario. "
        "Original prediction is shown in blue; the modified scenario in orange."
    )

    col_sl, col_ch = st.columns([1, 1])
    with col_sl:
        st.markdown("**Adjust features:**")
        wi_preop_tbwl = st.slider(
            "Preop TBWL%", min_value=-10.0, max_value=35.0,
            value=float(patient["Preop_TBWL"]), step=0.5,
            help="What if the patient had achieved a different preoperative weight loss?",
        )
        wi_age = st.slider(
            "Age", min_value=18, max_value=85, value=int(patient["Age"]), step=1,
        )
        wi_initial_bmi = st.slider(
            "Initial BMI", min_value=18.0, max_value=95.0,
            value=float(patient["Initial_BMI"]), step=0.5,
        )
        wi_surgery_type = st.selectbox(
            "Surgery Type",
            ["Sleeve", "Bypass", "Revision"],
            index=["Sleeve", "Bypass", "Revision"].index(patient["Surgery_Type"]),
            key="wi_surgery",
        )

    modified_features = {}
    if wi_preop_tbwl != patient["Preop_TBWL"]:
        modified_features["Preop_TBWL"] = wi_preop_tbwl
    if wi_age != patient["Age"]:
        modified_features["Age"] = wi_age
    if wi_initial_bmi != patient["Initial_BMI"]:
        modified_features["Initial_BMI"] = wi_initial_bmi
    if wi_surgery_type != patient["Surgery_Type"]:
        modified_features["Surgery_Type"] = wi_surgery_type

    with col_ch:
        if modified_features:
            st.markdown("**Changes from original:**")
            label_map = {
                "Preop_TBWL":   ("Preop TBWL%",  f"{patient['Preop_TBWL']:.1f}",  f"{wi_preop_tbwl:.1f}"),
                "Age":          ("Age",            str(patient["Age"]),               str(wi_age)),
                "Initial_BMI":  ("Initial BMI",    f"{patient['Initial_BMI']:.1f}",  f"{wi_initial_bmi:.1f}"),
                "Surgery_Type": ("Surgery",         patient["Surgery_Type"],           wi_surgery_type),
            }
            for key in modified_features:
                name, old_v, new_v = label_map[key]
                st.write(f"- {name}: {old_v} -> {new_v}")
        else:
            st.info("Adjust sliders on the left to model a different scenario.")

    run_whatif = st.button(
        "Run Scenario", type="primary", disabled=not bool(modified_features),
    )

    if run_whatif and modified_features:
        with st.spinner("Running scenario..."):
            wi_result = what_if_analysis(patient, modified_features)
            st.session_state["wi_result"] = wi_result

    if "wi_result" in st.session_state:
        wi_result = st.session_state["wi_result"]
        orig_traj = wi_result["original"]
        mod_traj = wi_result["modified"]

        d2 = wi_result["deltas"]["TBWL"].get(2, {})
        if (d2.get("delta_point") is not None
                and orig_traj["TBWL"][2]["point"] is not None
                and mod_traj["TBWL"][2]["point"] is not None):
            delta2 = d2["delta_point"]
            direction = "higher" if delta2 > 0 else "lower"
            st.info(
                f"Under the modified scenario, year-2 TBWL% is predicted at "
                f"**{mod_traj['TBWL'][2]['point']:.1f}%** vs. "
                f"**{orig_traj['TBWL'][2]['point']:.1f}%** originally -- "
                f"a difference of **{abs(delta2):.1f} pp {direction}**."
            )

        fig_wi, axes_wi = plt.subplots(1, 2, figsize=(12, 5))
        for ax, outcome in zip(axes_wi, ["TBWL", "FML"]):
            for t, label, color, ls in [
                (orig_traj, "Original", "#2980b9", "-"),
                (mod_traj,  "Scenario", "#e67e22", "--"),
            ]:
                pts = [(yr, t[outcome][yr]["point"]) for yr in range(1, 7)
                       if t[outcome][yr]["point"] is not None]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=label, color=color, linestyle=ls, linewidth=2)
            ax.set_xlabel("Year Postoperative")
            ax.set_ylabel(f"{outcome}%")
            ax.set_title(f"{outcome}%")
            ax.set_xticks(range(1, 7))
            ax.legend()
            ax.grid(True, alpha=0.25)
        fig_wi.suptitle("Original vs. Scenario Trajectory", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_wi)
        plt.close(fig_wi)

        st.markdown("**Year-by-year impact (Scenario minus Original)**")
        delta_rows = []
        for yr in range(1, 7):
            d = wi_result["deltas"]["TBWL"][yr]
            o_pt = orig_traj["TBWL"][yr]["point"]
            m_pt = mod_traj["TBWL"][yr]["point"]
            delta_rows.append({
                "Year": yr,
                "Original TBWL%": f"{o_pt:.1f}%" if o_pt is not None else "--",
                "Scenario TBWL%": f"{m_pt:.1f}%" if m_pt is not None else "--",
                "Delta (pp)": f"{d['delta_point']:+.1f}" if d["delta_point"] is not None else "--",
            })
        st.dataframe(delta_rows, hide_index=True, use_container_width=True)

    # Surgery Type Comparison
    st.divider()
    st.subheader("Surgery Type Comparison")
    st.caption(
        "For the same patient profile, compare predicted TBWL% across Sleeve, Bypass, and Revision. "
        "Useful when the surgery type has not been finalised."
    )

    compare_surg_btn = st.button("Compare Surgery Types", type="secondary")
    if compare_surg_btn:
        with st.spinner("Running 3 trajectories..."):
            surg_trajs = {}
            for stype in ["Sleeve", "Bypass", "Revision"]:
                surg_trajs[stype] = predict_trajectory({**patient, "Surgery_Type": stype})
            st.session_state["surg_trajs"] = surg_trajs

    if "surg_trajs" in st.session_state:
        surg_trajs = st.session_state["surg_trajs"]
        SURG_COLORS = {"Sleeve": "#3498db", "Bypass": "#2ecc71", "Revision": "#e74c3c"}

        fig_s, ax_s = plt.subplots(figsize=(9, 5))
        for stype, t in surg_trajs.items():
            pts = [(yr, t["TBWL"][yr]["point"]) for yr in range(1, 7)
                   if t["TBWL"][yr]["point"] is not None]
            if pts:
                xs, ys = zip(*pts)
                ax_s.plot(xs, ys, marker="o", label=stype,
                           color=SURG_COLORS[stype], linewidth=2)
        ax_s.set_xlabel("Year Postoperative")
        ax_s.set_ylabel("TBWL%")
        ax_s.set_title("Surgery Type Comparison -- TBWL%")
        ax_s.set_xticks(range(1, 7))
        ax_s.legend()
        ax_s.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

        yr2_vals = {
            s: surg_trajs[s]["TBWL"][2]["point"]
            for s in surg_trajs
            if surg_trajs[s]["TBWL"][2]["point"] is not None
        }
        if yr2_vals:
            best_s = max(yr2_vals, key=yr2_vals.get)
            worst_s = min(yr2_vals, key=yr2_vals.get)
            if best_s != worst_s:
                st.info(
                    f"At year 2, **{best_s}** produces the highest predicted TBWL% for this patient "
                    f"({yr2_vals[best_s]:.1f}%), vs. **{worst_s}** at {yr2_vals[worst_s]:.1f}%. "
                    f"Surgical choice involves many factors beyond weight loss -- these are model estimates only."
                )
            else:
                st.info("The three surgery types produce similar predicted year-2 TBWL% for this profile.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Phenotype + cluster deep-dive
# ─────────────────────────────────────────────────────────────────────────────
with tab_pheno:
    st.subheader("Trajectory Phenotype Assignment")
    st.warning("Phenotype recipe is **provisional** -- awaiting Dr. Raftopoulos confirmation.")

    pheno_model_path = ARTIFACTS / "phenotype_kmeans.joblib"
    pheno_id = None

    # 1A-aligned clustering needs predicted TBWL yrs 1-5 (all non-red)
    traj_pts = {yr: traj["TBWL"][yr]["point"] for yr in [1, 2, 3, 4, 5]}

    if not pheno_model_path.exists():
        st.info("Phenotype model not found. Run `python scripts/run_clustering.py` first.")
    elif any(v is None for v in traj_pts.values()):
        st.warning(
            "Cannot assign phenotype: predicted TBWL is unavailable for one or more of "
            "years 1-5 (red tier). Use clinical judgement."
        )
    else:
        with st.spinner("Assigning phenotype (UMAP projection)..."):
            result_pheno = assign_phenotype(patient, traj=traj)
        pheno_id = result_pheno["phenotype"]
        pheno_k = result_pheno["k"]
        pheno_n = result_pheno["n_train"]
        short_label, short_desc = phenotype_label(pheno_id, pheno_k)
        yr1_pt, yr5_pt = traj_pts[1], traj_pts[5]
        trend = "declining over time" if yr5_pt < yr1_pt else "sustained or improving over time"

        st.info(
            f"Among the {pheno_n} patients in this cohort, this patient's preop profile and "
            f"predicted TBWL trajectory place them in the **{short_label}** group "
            f"(Cluster {pheno_id} of {pheno_k}). "
            f"Their predicted TBWL% moves from {yr1_pt:.1f}% at year 1 to {yr5_pt:.1f}% at year 5, "
            f"which is {trend}. {short_desc}"
        )

        st.metric("Phenotype Cluster", f"#{pheno_id} of {pheno_k}")
        st.caption(
            f"Clustering (1A-aligned): preop features + predicted TBWL yrs 1-5 -> UMAP -> "
            f"KMeans, k={pheno_k} (silhouette-derived), N={pheno_n}, ordered by ascending "
            f"Preop_TBWL. {result_pheno['note']}"
        )

        # Cluster deep-dive
        st.divider()
        st.subheader("Cluster Deep-Dive -- How This Patient Compares to Their Group")

        cluster_data = _cluster_trajectories()
        if cluster_data is None:
            st.info(
                "Cohort cluster data unavailable -- CSV not found. "
                "Drop the data file into `data/` to enable this view."
            )
        else:
            cl = cluster_data.get(pheno_id, {})
            per_yr = cl.get("per_year", {})
            cl_n = cl.get("n", 0)

            if per_yr:
                fig_cl, ax_cl = plt.subplots(figsize=(9, 5))
                yrs_cl = sorted(per_yr.keys())
                means_cl = [per_yr[yr]["mean"] for yr in yrs_cl]
                sds_cl = [per_yr[yr]["sd"] for yr in yrs_cl]

                ax_cl.errorbar(
                    yrs_cl, means_cl, yerr=sds_cl, fmt="o-",
                    label=f"Cluster {pheno_id} actual (N={cl_n}, mean +/- SD)",
                    color="#2980b9", capsize=5, linewidth=2,
                )
                ax_cl.fill_between(
                    yrs_cl,
                    [m - s for m, s in zip(means_cl, sds_cl)],
                    [m + s for m, s in zip(means_cl, sds_cl)],
                    alpha=0.12, color="#2980b9",
                )

                pts = [(yr, traj["TBWL"][yr]["point"]) for yr in range(1, 7)
                       if traj["TBWL"][yr]["point"] is not None]
                if pts:
                    xs, ys = zip(*pts)
                    ax_cl.plot(
                        xs, ys, "^--",
                        label="This patient (predicted)",
                        color="#e74c3c", linewidth=2, markersize=10,
                    )

                ax_cl.set_xlabel("Year Postoperative")
                ax_cl.set_ylabel("TBWL%")
                ax_cl.set_title(
                    f"Cluster {pheno_id} -- Actual cohort trajectories vs. patient prediction"
                )
                ax_cl.set_xticks(range(1, 7))
                ax_cl.legend()
                ax_cl.grid(True, alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig_cl)
                plt.close(fig_cl)

                if (2 in per_yr and traj["TBWL"][2]["point"] is not None):
                    cl_mean2 = per_yr[2]["mean"]
                    cl_sd2 = per_yr[2]["sd"]
                    pt2 = traj["TBWL"][2]["point"]
                    rel = "above" if pt2 > cl_mean2 else "below"
                    z = (pt2 - cl_mean2) / cl_sd2 if cl_sd2 > 0 else 0
                    st.info(
                        f"The {cl_n} patients in cluster {pheno_id} ('{short_label}') had a mean "
                        f"year-2 TBWL of **{cl_mean2:.1f}% +/- {cl_sd2:.1f}% SD**. "
                        f"This patient's predicted year-2 TBWL is **{pt2:.1f}%** -- "
                        f"{rel} the cluster average by {abs(pt2 - cl_mean2):.1f} pp "
                        f"({abs(z):.1f} SD)."
                    )
            else:
                st.info(f"Cluster {pheno_id} has too few patients for a trajectory reference.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Preop Risk + comorbidity reference panel
# ─────────────────────────────────────────────────────────────────────────────
with tab_risk:
    st.subheader("Preoperative Weight-Loss Risk Assessment")
    risk = assess_preop_risk(patient["Preop_TBWL"])

    at_risk = risk["flag"] == "at_risk"
    gap_abs = abs(risk["gap"])
    if at_risk:
        summary = (
            f"This patient lost **{risk['preop_tbwl_pct']:.1f}%** of their body weight before surgery, "
            f"which is **{gap_abs:.1f} percentage points below** the 10.5% threshold associated with "
            f"better long-term outcomes in this cohort. "
            f"Patients below this threshold showed worse trajectories over years 1-6. "
            f"This flag is intended to prompt a conversation about additional preop support -- "
            f"it is not a contraindication to surgery."
        )
        st.error(f"AT RISK -- {risk['message']}")
    else:
        summary = (
            f"This patient lost **{risk['preop_tbwl_pct']:.1f}%** of their body weight before surgery, "
            f"meeting the 10.5% threshold by {gap_abs:.1f} percentage points. "
            f"Their preop weight loss pattern is consistent with patients who tend to achieve "
            f"stronger postoperative trajectories. Individual outcomes still vary widely."
        )
        st.success(f"ON TRACK -- {risk['message']}")

    st.info(summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Preop TBWL%", f"{risk['preop_tbwl_pct']:.1f}%")
    col2.metric("Threshold", f"{risk['threshold']}%")
    col3.metric("Gap", f"{risk['gap']:+.1f} pp")

    st.caption(
        "The 10.5% threshold is a rule from the manuscript. Patients below this level showed "
        "worse long-term outcomes. This is not a model prediction."
    )

    # Comorbidity reference panel
    st.divider()
    if comorbidities:
        st.subheader("Comorbidity Remission Reference Panel")
        st.caption(
            "Population-level literature estimates only. Not a patient-specific prediction. "
            "Rates vary by surgery type, baseline severity, and individual factors."
        )

        yr2_pt_risk = traj["TBWL"][2]["point"] or traj["TBWL"][1]["point"] or 0
        if yr2_pt_risk >= 25:
            tbwl_band = ">25% TBWL"
        elif yr2_pt_risk >= 20:
            tbwl_band = "20-25% TBWL"
        elif yr2_pt_risk >= 15:
            tbwl_band = "15-20% TBWL"
        else:
            tbwl_band = "<15% TBWL"

        st.info(
            f"Using predicted year-2 TBWL% of **{yr2_pt_risk:.1f}%** -- "
            f"reference column: **{tbwl_band}**."
        )

        comorb_rows = []
        for comorb in comorbidities:
            if comorb in COMORBIDITY_REMISSION:
                ref = COMORBIDITY_REMISSION[comorb]
                comorb_rows.append({
                    "Comorbidity": comorb,
                    f"Est. Remission ({tbwl_band})": ref[tbwl_band],
                    "Source": ref["source"],
                })
        if comorb_rows:
            st.dataframe(comorb_rows, hide_index=True, use_container_width=True)
            st.caption(
                "Remission = partial or complete resolution per each study's definition. "
                "Rates are for Bypass unless noted. Sleeve GERD remission may be lower or worsen."
            )
    else:
        st.info("Select comorbidities in the sidebar to see the literature reference panel.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: SHAP Drivers
# ─────────────────────────────────────────────────────────────────────────────
with tab_shap:
    st.subheader("Feature Importance (SHAP)")
    st.caption(
        "Shows which baseline features drive the prediction up or down relative to the average patient. "
        "Only available for green/amber (non-red) years. "
        "SVR models use KernelExplainer -- may take 1-2 minutes."
    )

    non_red_options = [
        f"{outcome} yr{yr}"
        for outcome in ["TBWL", "FML"]
        for yr in range(1, 7)
        if traj[outcome][yr]["point"] is not None
    ]

    if not non_red_options:
        st.warning("No non-red years available for SHAP explanation.")
    else:
        selected = st.selectbox("Select outcome + year to explain:", non_red_options)
        explain_btn = st.button("Explain Drivers", type="secondary")

        if explain_btn:
            outcome_sel, yr_sel = selected.split(" yr")
            yr_int = int(yr_sel)

            with st.spinner(f"Computing SHAP for {outcome_sel} yr{yr_int}..."):
                try:
                    from src.explain import explain_with_shap
                    shap_result = explain_with_shap(patient, outcome_sel, yr_int, top_n=8)
                    st.session_state["shap_result"] = shap_result
                    st.session_state["shap_selected"] = selected
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

        if "shap_result" in st.session_state and st.session_state.get("shap_selected") == selected:
            shap_result = st.session_state["shap_result"]

            top_pos = shap_result["top_positive"]
            top_neg = shap_result["top_negative"]
            pos_str = " and ".join(d["feature"] for d in top_pos[:2]) if top_pos else "none"
            neg_str = " and ".join(d["feature"] for d in top_neg[:2]) if top_neg else "none"
            base_val = shap_result["base_value"]
            pred_pt = traj[shap_result["outcome"]][shap_result["year"]]["point"]
            direction = "above" if pred_pt and pred_pt > base_val else "below"
            diff = abs(round(pred_pt - base_val, 1)) if pred_pt else 0

            st.info(
                f"The average patient in the training cohort had a {shap_result['outcome']}% of "
                f"**{base_val:.1f}%** at year {shap_result['year']}. "
                f"This patient is predicted at **{pred_pt:.1f}%** -- {diff:.1f} points {direction} average. "
                f"The features pushing their prediction **up** most are {pos_str}; "
                f"the features pulling it **down** most are {neg_str}."
            )

            fig2, ax2 = plt.subplots(figsize=(9, 5))
            drivers = shap_result["top_positive"][::-1] + shap_result["top_negative"]
            features = [d["feature"] for d in drivers]
            values = [d["shap_value"] for d in drivers]
            bar_colors = [TIER_COLOR["green"] if v >= 0 else TIER_COLOR["red"] for v in values]
            ax2.barh(features, values, color=bar_colors)
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("SHAP value (impact on prediction in %)")
            ax2.set_title(
                f"Top drivers -- {shap_result['outcome']} yr{shap_result['year']} "
                f"(tier: {shap_result['tier'].upper()})"
            )
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown(
                f"**Base value:** {base_val:.2f}% (population mean for "
                f"{shap_result['outcome']} yr{shap_result['year']})"
            )
            if MODEL_PERFORMANCE[shap_result["outcome"]][shap_result["year"]]["best_model"] == "SVR":
                st.caption("KernelExplainer used (SVR model) -- SHAP values are approximate.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6: Summary card + download
# ─────────────────────────────────────────────────────────────────────────────
with tab_summary:
    st.subheader("Patient Summary Card")
    st.caption(
        "A one-page summary of key findings. Download as HTML to print or share with the care team. "
        "Contains no patient identifiers."
    )

    summary_pheno_id = None
    summary_pheno_short, summary_pheno_desc, summary_pheno_k = "", "", None
    if (ARTIFACTS / "phenotype_kmeans.joblib").exists() and all(
        traj["TBWL"][yr]["point"] is not None for yr in [1, 2, 3, 4, 5]
    ):
        try:
            _res = assign_phenotype(patient, traj=traj)
            summary_pheno_id = _res["phenotype"]
            summary_pheno_k = _res["k"]
            summary_pheno_short, summary_pheno_desc = phenotype_label(
                summary_pheno_id, summary_pheno_k
            )
        except Exception:
            pass

    risk_sum = assess_preop_risk(patient["Preop_TBWL"])
    shap_for_card = st.session_state.get("shap_result")

    c1, c2, c3 = st.columns(3)
    yr2_pt_s = traj["TBWL"][2]["point"]
    c1.metric("TBWL% yr2", f"{yr2_pt_s:.1f}%" if yr2_pt_s is not None else "--")
    c2.metric("Preop Risk", "At risk" if risk_sum["flag"] == "at_risk" else "On track")
    c3.metric(
        "Phenotype",
        f"Cluster {summary_pheno_id} of {summary_pheno_k}" if summary_pheno_id is not None else "N/A",
    )

    if shap_for_card:
        top_drv = shap_for_card.get("top_positive", [])[:3]
        if top_drv:
            st.caption(f"SHAP top drivers: {', '.join(d['feature'] for d in top_drv)}")
    else:
        st.caption(
            "Tip: run SHAP Drivers first to include top predictive features in the summary card."
        )

    html_card = _generate_summary_html(
        patient, traj, summary_pheno_id, risk_sum, shap_for_card,
        pheno_short=summary_pheno_short, pheno_desc=summary_pheno_desc,
    )

    import streamlit.components.v1 as components
    components.html(html_card, height=520, scrolling=True)

    st.download_button(
        label="Download Summary Card (HTML)",
        data=html_card,
        file_name=f"bariatric_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
    )
