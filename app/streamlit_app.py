"""Bariatric CDS — Streamlit MVP (Stage 3).

Internal research prototype only. Not a medical device. Not patient-facing.
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
# Ordered 0-4 by ascending year-3 TBWL mean (lowest → highest loss)
# PROVISIONAL — awaiting Dr. Raftopoulos confirmation on clustering recipe.
PHENOTYPE_DESC = {
    0: "Minimal loss — lowest average TBWL through year 3. May benefit from additional preop support.",
    1: "Moderate loss, plateauing after year 2. Typical trajectory for this cohort.",
    2: "Moderate-high loss, sustained through year 3.",
    3: "High early loss with continued decline. Strong response to surgery.",
    4: "Highest loss trajectory — strong responders across years 1-3.",
}

# ── sidebar — patient inputs ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Patient Features")
    st.caption("Enter baseline values. All fields required.")

    age           = st.number_input("Age (years)",            min_value=18,   max_value=80,   value=45)
    height        = st.number_input("Height (cm)",            min_value=140,  max_value=220,  value=170)
    initial_bmi   = st.number_input("Initial BMI",            min_value=30.0, max_value=80.0, value=45.0,  step=0.1)
    initial_wt    = st.number_input("Initial Weight (kg)",    min_value=50.0, max_value=300.0, value=120.0, step=0.1)
    initial_bmr   = st.number_input("Initial BMR (kcal/day)", min_value=1000, max_value=4000,  value=1800)
    initial_vf    = st.number_input("Initial Visceral Fat",   min_value=0.0,  max_value=50.0,  value=15.0,  step=0.1)
    initial_fatpct = st.number_input("Initial FAT%",          min_value=10.0, max_value=70.0,  value=45.0,  step=0.1)
    initial_fatmass = st.number_input("Initial Fat Mass (kg)", min_value=5.0,  max_value=200.0, value=55.0,  step=0.1)
    initial_ffm   = st.number_input("Initial FFM (kg)",       min_value=20.0, max_value=150.0, value=65.0,  step=0.1)
    time_to_surg  = st.number_input("Time to Surgery (months)", min_value=0,  max_value=60,    value=6)
    preop_bmi     = st.number_input("Preop BMI",              min_value=25.0, max_value=80.0,  value=43.0,  step=0.1)
    preop_tbwl    = st.number_input("Preop TBWL%",            min_value=0.0,  max_value=30.0,  value=8.0,   step=0.1)

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

if not compute_btn:
    st.info("Enter patient features in the sidebar and click **Compute Trajectory**.")
    st.stop()

patient = {
    "Age": age, "Sex": sex, "Race": race, "Height": height,
    "Initial_BMI": initial_bmi, "Initial_Weight": initial_wt,
    "Initial_BMR": initial_bmr, "Initial_VF": initial_vf,
    "Initial_FATpct": initial_fatpct, "Initial_FATMASS": initial_fatmass,
    "Initial_FFM": initial_ffm, "Time_to_Surgery": time_to_surg,
    "Surgery_Type": surgery_type, "Preop_BMI": preop_bmi, "Preop_TBWL": preop_tbwl,
}

with st.spinner("Computing trajectory..."):
    try:
        traj = predict_trajectory(patient)
    except FileNotFoundError as exc:
        st.error(f"Model artifacts not found.\n\n{exc}")
        st.stop()

tab_traj, tab_pheno, tab_risk, tab_shap = st.tabs(
    ["📈 Trajectory", "🔵 Phenotype", "⚠️ Preop Risk", "🔍 SHAP Drivers"]
)

# ── Tab 1: Trajectory ────────────────────────────────────────────────────────
with tab_traj:
    st.subheader("Predicted Weight-Loss Trajectory")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    outcomes_meta = [("TBWL", "TBWL%"), ("FML", "FML%")]

    for ax, (outcome, ylabel) in zip(axes, outcomes_meta):
        years, points, los, his, colors = [], [], [], [], []
        for yr in range(1, 7):
            d = traj[outcome][yr]
            color = TIER_COLOR[d["tier"]]
            if d["point"] is not None:
                years.append(yr)
                points.append(d["point"])
                los.append(d["point"] - d["lo"])
                his.append(d["hi"] - d["point"])
                colors.append(color)
                ax.errorbar(yr, d["point"], yerr=[[d["point"] - d["lo"]], [d["hi"] - d["point"]]],
                            fmt="o", color=color, capsize=5, markersize=7)
            else:
                ax.plot(yr, 0, "x", color=color, markersize=10, markeredgewidth=2)
                ax.text(yr, 1.5, "UNRELIABLE\n(red tier)", ha="center",
                        fontsize=7, color=color)

        if len(years) > 1:
            ax.plot(years, points, "--", color="gray", linewidth=1, alpha=0.5, zorder=0)

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

    # Tier legend
    cols = st.columns(3)
    cols[0].markdown(f"🟢 **Green** — reliable (R² ≥ 0.40)")
    cols[1].markdown(f"🟡 **Amber** — weak, wide uncertainty (R² 0.20–0.40)")
    cols[2].markdown(f"🔴 **Red** — unreliable, no point prediction (R² < 0.20)")

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
        # Use predicted yr1-3 TBWL to assign phenotype
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
            st.metric("Phenotype Cluster", f"#{pheno_id} of 5")
            st.info(PHENOTYPE_DESC.get(pheno_id, "Description not available."))
            st.caption(
                f"Clustering: k-means on standardized TBWL% at years 1-3, k=5, complete-case N=149. "
                f"{result['note']}"
            )

# ── Tab 3: Preop Risk ────────────────────────────────────────────────────────
with tab_risk:
    st.subheader("Preoperative Weight-Loss Risk Assessment")
    risk = assess_preop_risk(preop_tbwl)

    if risk["flag"] == "at_risk":
        st.error(f"⚠️ **AT RISK** — {risk['message']}")
    else:
        st.success(f"✅ **ON TRACK** — {risk['message']}")

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
        "Shows which baseline features drive the prediction up or down. "
        "Only available for green/amber (non-red) years. "
        "SVR models use KernelExplainer — may take 1-2 minutes."
    )

    # Build list of non-red (outcome, year) pairs
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

            with st.spinner(f"Computing SHAP for {outcome_sel} yr{yr_int}... (SVR may take ~1-2 min)"):
                try:
                    from src.explain import explain_with_shap
                    shap_result = explain_with_shap(patient, outcome_sel, yr_int, top_n=8)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

            st.markdown(f"**Base value:** {shap_result['base_value']:.2f}% "
                        f"(population mean for {outcome_sel} yr{yr_int})")

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

            if MODEL_PERFORMANCE[outcome_sel][yr_int]["best_model"] == "SVR":
                st.caption("⚠️ KernelExplainer used (SVR model) — SHAP values are approximate.")
