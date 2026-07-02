# CLAUDE.md — Bariatric Clinical Decision Support (research prototype)

Load this every session. Keep it current. Prefer editing a wrong instruction here
over re-explaining it in chat.

## What this is
An **internal research prototype** that helps a surgeon set realistic expectations
for post-bariatric weight-loss trajectories. It predicts TBWL% / FML% over years
1-6, assigns a patient to a trajectory phenotype, flags preop-weight-loss risk, and
explains drivers with SHAP. A natural-language agent wraps these as tools.

**Intended use = research / quality-improvement.** Not a medical device. Every
output states uncertainty and defers to the care team. Do not add anything that
makes this diagnostic or prescriptive.

## Hard rules (do not violate)
- **PHI stays out of `git` and out of the LLM.** `data/` and `artifacts/` are
  gitignored. Never commit the CSV. The agent's tools run models **locally**; only
  derived outputs (predicted %, cluster id, driver names) may go to the LLM. Never
  pass raw patient rows or `ID` to the Anthropic API.
- **`ID` is a study id, not an MRN.** If that ever changes, stop and re-de-identify.
- Fix random seeds everywhere (`random_state=42`). Results must reproduce.
- Every modeling choice gets one line in `docs/METHODS_LOG.md`. This is a research
  artifact headed for a paper; undocumented choices are worthless later.

## Data
- File: `data/patients-data5b-may11data-merge-edited.csv` (drop it here yourself).
- 802 patients, 185 columns. Baseline/preop fields ~complete; postop outcomes attrite
  hard (per Supplementary Table S4): TBWL non-missing = 608/399/254/185/87/83 for
  years 1-6. Body-comp (FML%) is sparser still.
- Outcome columns follow `{n}yr_Postop_{Metric}`, e.g. `3yr_Postop_TBWL`,
  `2yr_Postop_FML%`. Baseline in `src/config.py`.

## Reliability gating (from Supplementary Table S5 — non-negotiable)
Best-model R² by year drives a green/amber/red tier. The tool MUST surface this and
MUST refuse point predictions in red years. Values + tiers live in `src/config.py`
(`MODEL_PERFORMANCE`) and are applied by `src/reliability.py`.
- TBWL: yr1 amber, yr2-4 green, yr5 amber, yr6 red.
- FML%: yr1 red, yr2 green, yr3-4 amber, yr5-6 red.

## Preop threshold rule
PreopTBWL% < **10.5%** is the actionable flag associated with worse long-term
outcomes. Implemented as a rule in `src/risk.py` (no model needed).

## Phenotype / clustering — RECIPE NOT YET FINAL
`src/phenotype.py` holds the current defensible default: k-means on standardized
TBWL% at years 1-3, complete-case (N=149), k=5. **Caveat:** silhouette actually
prefers k=2; k=5 is imposed to match the manuscript. The feature set + patient
inclusion + choice of k are being confirmed with Dr. Raftopoulos (Monday meeting).
Until confirmed, treat phenotypes as provisional. `assign_phenotype` must load the
one frozen fitted model from `artifacts/`, never re-fit per call.

## Build stages (all complete)
1. ✅ Blind clustering exploration (done in chat; ported to `src/phenotype.py`).
2. ✅ **Model layer reproduced and validated.** All 12 per-year models pass the ΔR²
   gate vs S5. Artifacts in `artifacts/`. Single-lag cascade; SVR→RF when N<100;
   amber N<50 skips strict gate. See `docs/METHODS_LOG.md` for full decisions.
3. ✅ Streamlit MVP — `app/streamlit_app.py`. 4 tabs: Trajectory | Phenotype |
   Preop Risk | SHAP Drivers. Run: `streamlit run app/streamlit_app.py`.
4. ✅ Agent layer — `src/agent.py`. PHI guard, audit log, 5 tools (predict, phenotype,
   SHAP, preop risk, what-if). Requires `ANTHROPIC_API_KEY` env var.
   Research Q1a/Q1b completed: `scripts/analyze_bmi_milestone.py`.

## Commands
- `pip install -r requirements.txt`
- `python scripts/run_clustering.py`         # reproduce the k-means phenotype model
- `PYTHONPATH=. python scripts/reproduce_models.py`   # refit + validate all 12 models
- `PYTHONPATH=. python scripts/analyze_bmi_milestone.py`  # Q1a/Q1b analysis
- `streamlit run app/streamlit_app.py`       # launch Streamlit MVP
- `pytest -q`

## Conventions
Python 3, scikit-learn. Small pure functions with typed signatures per tool
(`predict_trajectory`, `assign_phenotype`, `explain_with_shap`, `assess_preop_risk`,
`what_if`). Write a test when you add a tool. Don't invent features not in the CSV —
check `src/config.py` / the dataframe first.
