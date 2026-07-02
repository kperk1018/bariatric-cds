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
import streamlit as st

from src.predict import predict_trajectory
from src.phenotype import assign_phenotype
from src.risk import assess_preop_risk
from src.config import MODEL_PERFORMANCE, ARTIFACTS

st.set_page_config(page_title="Bariatric CDS", layout="wide")

# ── colour map for reliability tiers ──────────────────────────────────────────
TIER_COLOR = {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c"}

# ── provisional phenotype descriptions ────────────────────────────────────────
PHENOTYPE_DESC = {
    0: "Minimal loss — lowest average TBWL through year 3. May benefit from additional preop support.",
    1: "Moderate loss, plateauing after year 2. Typical trajectory for this cohort.",
    2: "Moderate-high loss, sustained through year 3.",
    3: "High early loss with continued decline. Strong response to surgery.",
    4: "Highest loss trajectory — strong responders across years 1-3.",
}

PHENOTYPE_SHORT = {
    0: "minimal responder",
    1: "moderate responder (plateauing)",
    2: "moderate-high responder",
    3: "high early responder",
    4: "strongest responder",
}

# ── sidebar — patient inputs ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Patient Features")
    st.caption("Enter baseline values. All fields required.")

    age            = st.number_input("Age (years)",              min_value=18,    max_value=85,    value=43)
    height         = st.number_input("Height (inches)",          min_value=48,    max_value=84,    value=64)
    initial_bmi    = st.number_input("Initial BMI",              min_value=18.0,  max_value=95.0,  value=43.0,  step=0.1)
    initial_wt     = st.number_input("Initial Weight (lbs)",     min_value=100.0, max_value=550.0, value=255.0, step=0.1)
    initial_bmr    = st.number_input("Initial BMR (kcal/day)",   min_value=500,   max_value=4000,  value=1957)
    initial_vf     = st.number_input("Initial Visceral Fat",     min_value=0.0,   max_value=50.0,  value=14.0,  step=0.1)
    initial_fatpct = st.number_input("Initial FAT%",             min_value=15.0,  max_value=72.0,  value=45.0,  step=0.1)
    initial_fatmass = st.number_input("Initial Fat Mass (lbs)",  min_value=20.0,  max_value=320.0, value=115.0, step=0.1)
    initial_ffm    = st.number_input("Initial FFM (lbs)",        min_value=40.0,  max_value=240.0, value=137.0, step=0.1)
    time_to_surg   = st.number_input("Time to Surgery (days)",   min_value=1,     max_value=1500,  value=135)
    preop_bmi      = st.number_input("Preop BMI",                min_value=25.0,  max_value=70.0,  value=38.5,  step=0.1)
    preop_tbwl     = st.number_input("Preop TBWL%",              min_value=-10.0, max_value=35.0,  value=11.5,  step=0.1)

    sex          = st.selectbox("Sex",          ["Female", "Male"])
    race         = st.selectbox("Race",         ["White", "Hispanic", "African_American",
                                                  "Asian", "Other"])
    surgery_type = st.selectbox("Surgery Type", ["Sleeve", "Bypass", "Revision"])

    st.divider()
    compute_btn = st.button("Compute Trajectory", type="primary", use_container_width=True)

# ── main panel ────────────────────────────────────────────────────────────────
st.title("Bariatric Trajectory CDS — Research Prototype")
st.caption(
    "**Research / quality-improvement use only.** Not a medical device. "
    "All predictions carry uncertainty. Defer to the care team for clinical decisions."
)

# ── abbreviation glossary ─────────────────────────────────────────────────────
with st.expander("📖 Abbreviations & definitions"):
    st.markdown("""
| Term | Full name | What it means |
|---|---|---|
| **TBWL%** | Total Body Weight Loss % | Percentage of the patient's starting weight lost since surgery. A 30% TBWL means someone who started at 250 lbs has lost 75 lbs. |
| **FML%** | Fat Mass Loss % | Percentage of the patient's initial fat mass lost. A higher FML% relative to TBWL% means the patient is losing fat preferentially over lean tissue. |
| **BMI** | Body Mass Index | Weight (lbs) ÷ Height (in)² × 703. Used as a proxy for overall adiposity. |
| **BMR** | Basal Metabolic Rate | Calories the body burns at complete rest. Estimated from body composition at the initial visit (kcal/day). |
| **VF** | Visceral Fat | Fat stored around internal organs (measured by body composition scan). Higher visceral fat is associated with metabolic risk. |
| **FFM** | Fat-Free Mass | Lean body mass — muscle, bone, water, organs. Preserved FFM after surgery is a marker of healthy weight loss. |
| **FAT%** | Body Fat Percentage | Fat mass as a percentage of total body weight at the initial visit. |
| **Preop TBWL%** | Preoperative TBWL% | Weight lost between the first clinic visit and the day of surgery, expressed as % of initial body weight. |
| **R²** | R-squared | How much of the variation in outcomes the model explains (0 = nothing, 1 = perfect). Drives the green/amber/red reliability tier. |
| **SHAP** | SHapley Additive exPlanations | A method that shows how much each input feature pushed this patient's prediction up or down relative to the average patient. |
| **Green tier** | R² ≥ 0.40 | Model explains enough variation to show a point estimate with confidence. |
| **Amber tier** | R² 0.20–0.40 | Model signal is weak — treat as a rough guide, not a precise number. |
| **Red tier** | R² < 0.20 | Model cannot reliably predict this outcome for this year. No number shown. |
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

if compute_btn:
    with st.spinner("Computing trajectory..."):
        try:
            st.session_state["traj"] = predict_trajectory(patient)
            st.session_state["patient"] = patient
        except FileNotFoundError as exc:
            st.error(f"Model artifacts not found.\n\n{exc}")
            st.stop()

if "traj" not in st.session_state:
    st.info("Enter patient features in the sidebar and click **Compute Trajectory**.")
    st.stop()

traj = st.session_state["traj"]
patient = st.session_state["patient"]

tab_traj, tab_pheno, tab_risk, tab_shap = st.tabs(
    ["📈 Trajectory", "🔵 Phenotype", "⚠️ Preop Risk", "🔍 SHAP Drivers"]
)

# ── Tab 1: Trajectory ────────────────────────────────────────────────────────
with tab_traj:
    st.subheader("Predicted Weight-Loss Trajectory")

    # ── patient summary paragraph ─────────────────────────────────────────────
    yr1 = traj["TBWL"][1]
    yr2 = traj["TBWL"][2]
    best_tbwl_yr, best_tbwl_val = max(
        ((yr, traj["TBWL"][yr]["point"]) for yr in range(1, 6)
         if traj["TBWL"][yr]["point"] is not None),
        key=lambda x: x[1], default=(None, None)
    )
    fml_yr2 = traj["FML"][2]

    reliable_tbwl_yrs = [yr for yr in range(1, 7) if traj["TBWL"][yr]["point"] is not None]
    tier_note = (
        f"years {min(reliable_tbwl_yrs)}–{max(reliable_tbwl_yrs)}"
        if reliable_tbwl_yrs else "no reliable years"
    )

    weight_lost_yr1 = round(patient["Initial_Weight"] * (yr1["point"] / 100), 1) if yr1["point"] else None
    summary_parts = []
    if yr1["point"] is not None:
        summary_parts.append(
            f"Based on this patient's baseline profile, the model predicts a **{yr1['point']:.1f}% total body weight loss (TBWL%) by year 1** "
            f"— approximately **{weight_lost_yr1:.0f} lbs** from their starting weight of {patient['Initial_Weight']:.0f} lbs."
        )
    if yr2["point"] is not None:
        direction = "continuing to decline" if yr2["point"] < (yr1["point"] or 0) else "plateauing"
        summary_parts.append(
            f"By year 2 (the most reliably predicted point), TBWL% is estimated at **{yr2['point']:.1f}%**, {direction}."
        )
    if fml_yr2["point"] is not None:
        summary_parts.append(
            f"Fat mass loss at year 2 is predicted at **{fml_yr2['point']:.1f}%** of initial fat mass."
        )
    summary_parts.append(
        f"Reliable predictions (green/amber tier) are available for {tier_note}; "
        f"years outside this range are marked unreliable and no point estimate is shown."
    )

    st.info(" ".join(summary_parts))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    outcomes_meta = [("TBWL", "TBWL%"), ("FML", "FML%")]

    for ax, (outcome, ylabel) in zip(axes, outcomes_meta):
        for yr in range(1, 7):
            d = traj[outcome][yr]
            color = TIER_COLOR[d["tier"]]
            if d["point"] is not None:
                ax.errorbar(yr, d["point"], yerr=[[d["point"] - d["lo"]], [d["hi"] - d["point"]]],
                            fmt="o", color=color, capsize=5, markersize=7)
            else:
                ax.plot(yr, 0, "x", color=color, markersize=10, markeredgewidth=2)
                ax.text(yr, 1.5, "UNRELIABLE\n(red tier)", ha="center",
                        fontsize=7, color=color)

        pts = [(yr, traj[outcome][yr]["point"]) for yr in range(1, 7) if traj[outcome][yr]["point"] is not None]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", color="gray", linewidth=1, alpha=0.5, zorder=0)

        ax.axhline(0, color="lightgray", linewidth=0.8)
        ax.set_xlabel("Year Postoperative")
        ax.set_ylabel(ylabel)
        ax.set_title(outcome)
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.5, 6.5)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Predicted Trajectory with ±1.96×RMSE Uncertainty Bands", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    cols = st.columns(3)
    cols[0].markdown("🟢 **Green** — reliable (R² ≥ 0.40)")
    cols[1].markdown("🟡 **Amber** — weak, wide uncertainty (R² 0.20–0.40)")
    cols[2].markdown("🔴 **Red** — unreliable, no point prediction (R² < 0.20)")

    st.divider()
    st.caption("**Detailed results**")
    for outcome in ["TBWL", "FML"]:
        rows = []
        for yr in range(1, 7):
            d = traj[outcome][yr]
            rows.append({
                "Year": yr,
                "Tier": d["tier"].upper(),
                "Point": f"{d['point']:.1f}%" if d["point"] else "—",
                "95% CI": f"[{d['lo']:.1f}, {d['hi']:.1f}]" if d["point"] else "—",
                "R²": d["r2"],
                "Note": d["message"],
            })
        st.write(f"**{outcome}%**")
        st.dataframe(rows, use_container_width=True, hide_index=True)

# ── Tab 2: Phenotype ─────────────────────────────────────────────────────────
with tab_pheno:
    st.subheader("Trajectory Phenotype Assignment")
    st.warning("⚠️ Phenotype recipe is **provisional** — awaiting Dr. Raftopoulos confirmation.")

    pheno_model_path = ARTIFACTS / "phenotype_kmeans.joblib"
    if not pheno_model_path.exists():
        st.info("Phenotype model not found. Run `python scripts/run_clustering.py` first.")
    else:
        traj_vals = {}
        for yr in [1, 2, 3]:
            d = traj["TBWL"][yr]
            if d["point"] is not None:
                traj_vals[f"{yr}yr_Postop_TBWL"] = d["point"]

        if len(traj_vals) < 3:
            st.warning(
                "Cannot assign phenotype: predicted TBWL is unavailable for years 1-3 "
                "(one or more are red tier). Use clinical judgement."
            )
        else:
            result = assign_phenotype(traj_vals)
            pheno_id = result["phenotype"]

            # ── patient summary paragraph ─────────────────────────────────────
            short_label = PHENOTYPE_SHORT.get(pheno_id, f"cluster {pheno_id}")
            yr1_pt = traj_vals.get("1yr_Postop_TBWL", 0)
            yr3_pt = traj_vals.get("3yr_Postop_TBWL", 0)
            trend = "declining over time" if yr3_pt < yr1_pt else "sustained or improving over time"
            st.info(
                f"Among the 149 patients with complete 3-year follow-up in this cohort, this patient's "
                f"predicted trajectory most closely resembles the **{short_label}** group (Cluster {pheno_id} of 5). "
                f"Their predicted TBWL% moves from {yr1_pt:.1f}% at year 1 to {yr3_pt:.1f}% at year 3, which is {trend}. "
                f"{PHENOTYPE_DESC.get(pheno_id, '')}"
            )

            st.metric("Phenotype Cluster", f"#{pheno_id} of 5")
            st.caption(
                f"Clustering: k-means on standardized TBWL% at years 1-3, k=5, complete-case N=149. "
                f"{result['note']}"
            )

# ── Tab 3: Preop Risk ────────────────────────────────────────────────────────
with tab_risk:
    st.subheader("Preoperative Weight-Loss Risk Assessment")
    risk = assess_preop_risk(patient["Preop_TBWL"])

    # ── patient summary paragraph ─────────────────────────────────────────────
    at_risk = risk["flag"] == "at_risk"
    gap_abs = abs(risk["gap"])
    if at_risk:
        summary = (
            f"This patient lost **{risk['preop_tbwl_pct']:.1f}%** of their body weight before surgery, "
            f"which is **{gap_abs:.1f} percentage points below** the 10.5% threshold associated with better long-term outcomes in this cohort. "
            f"Patients who do not reach this threshold have shown worse weight-loss trajectories over years 1–6. "
            f"This flag is intended to prompt a conversation about additional preoperative support — it is not a contraindication to surgery."
        )
        st.error(f"⚠️ **AT RISK** — {risk['message']}")
    else:
        summary = (
            f"This patient lost **{risk['preop_tbwl_pct']:.1f}%** of their body weight before surgery, "
            f"which meets the 10.5% threshold associated with better long-term outcomes in this cohort "
            f"(by {gap_abs:.1f} percentage points). "
            f"Their preoperative weight loss pattern is consistent with patients who tend to achieve stronger postoperative trajectories. "
            f"This is a positive prognostic signal, though individual outcomes still vary widely."
        )
        st.success(f"✅ **ON TRACK** — {risk['message']}")

    st.info(summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Preop TBWL%", f"{risk['preop_tbwl_pct']:.1f}%")
    col2.metric("Threshold", f"{risk['threshold']}%")
    col3.metric("Gap", f"{risk['gap']:+.1f} pp")

    st.caption(
        "The 10.5% threshold is a documented rule from the manuscript — patients below "
        "this level showed worse long-term outcomes. This is not a model prediction."
    )

# ── Tab 4: SHAP Drivers ──────────────────────────────────────────────────────
with tab_shap:
    st.subheader("Feature Importance (SHAP)")
    st.caption(
        "Shows which baseline features drive the prediction up or down relative to the average patient. "
        "Only available for green/amber (non-red) years. "
        "SVR models use KernelExplainer — may take 1-2 minutes."
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

            # ── patient summary paragraph ─────────────────────────────────────
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
                f"This patient is predicted at **{pred_pt:.1f}%** — {diff:.1f} points {direction} average. "
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
            ax2.set_title(f"Top drivers — {outcome_sel} yr{yr_int} "
                          f"(tier: {shap_result['tier'].upper()})")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown(f"**Base value:** {base_val:.2f}% (population mean for {outcome_sel} yr{yr_int})")
            if MODEL_PERFORMANCE[outcome_sel][yr_int]["best_model"] == "SVR":
                st.caption("⚠️ KernelExplainer used (SVR model) — SHAP values are approximate.")
