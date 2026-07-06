# Bariatric CDS — How This Was Built

This document explains how the codebase in this folder was assembled: what data it
started from, what machine learning methods were used and why, and how all the
pieces connect at prediction time. It's meant to be read alongside the repo by
someone who wasn't part of building it.

For the terse, dated, decision-by-decision log, see `docs/METHODS_LOG.md`. This
document is the narrative version of that log.

---

## 1. The data

**Source:** a single CSV, one row per patient — 802 patients, 185 columns
(`data/patients-data5b-may11data-merge-edited.csv`; never committed to git — see
§7). Columns fall into three groups:

- **Baseline / preop** (~complete): age, sex, race, height, initial weight/BMI/BMR,
  body composition (visceral fat, fat mass, fat-free mass, fat %), time to surgery,
  surgery type, preop BMI and preop weight loss.
- **Extended preop** (mostly complete, some sparse): 14 labs (HbA1c, insulin,
  lipid panel, liver enzymes, CRP...), comorbidity flags (diabetes, hypertension,
  hyperlipidemia, sleep apnea, depression, GERD, smoking), prior-surgery history,
  and four psychological/behavioral scores (binge-eating scale, quality-of-life
  score, Epworth sleepiness, adverse-childhood-experiences score).
- **Postoperative outcomes**: `TBWL%` (total body weight loss) and `FML%` (fat
  mass loss), each measured at years 1 through 6.

**The defining problem with this data is attrition, not size.** 802 patients
sounds like plenty, but follow-up drops off a cliff: non-missing TBWL% by year is
608 → 399 → 254 → 185 → 87 → 83. By year 6 there are 83 patients, and — this
matters more than the small count — they are *not* a random 83. Patients who stay
in a bariatric follow-up program for six years tend to be more engaged or doing
better than patients who drop out. This selection bias was later measured
directly (§5) rather than assumed.

A light de-identification guard (`src/data_load.py`) refuses to load the CSV if
any column name looks like a direct identifier (name, MRN, DOB, SSN, address,
phone, email). `ID` is treated as a study id, not a medical record number.

---

## 2. The clinical questions this needed to answer

Four things, each becoming one module in `src/`:

1. **Predict the trajectory.** Given a patient's baseline profile, forecast
   TBWL% and FML% for years 1–6, with a stated confidence interval.
2. **Flag preop risk.** Patients who lose less than 10.5% of body weight before
   surgery (`Preop_TBWL`) have historically shown worse long-term outcomes — this
   is a fixed clinical rule (`src/risk.py`), not a model.
3. **Group patients into a trajectory phenotype** — a small number of "shapes" of
   response (e.g., early strong responder vs. plateau) — so a surgeon can reason
   about a patient by analogy to others like them.
4. **Explain any prediction** in terms of which input features pushed it up or
   down (SHAP), so a number is never just handed over unexplained.

---

## 3. Reliability gating — the rule everything else obeys

Before any model was trained, one rule was fixed and treated as non-negotiable
throughout the project: **a model's R² (how much outcome variance it explains)
determines whether the tool is allowed to show a number at all.**

```
green  : R² ≥ 0.40   → usable, moderate confidence
amber  : R² 0.20–0.40 → show, but as a wide-uncertainty rough guide
red    : R² < 0.20   → refuse the point prediction entirely
```

This comes from a source table of best-model performance per outcome/year
(`MODEL_PERFORMANCE` in `src/config.py`, transcribed from the project's
Supplementary Table S5) and is enforced by `src/reliability.py`. Per the original
table:

- **TBWL%:** yr1 amber, yr2–4 green, yr5 amber, yr6 red
- **FML%:** yr1 red, yr2 green, yr3–4 amber, yr5–6 red

Every prediction returned by the code carries its tier, R², RMSE, and a plain-
English message alongside the number (or alongside `None`, in red years — the
code returns `point: None` rather than ever fabricating a value). This is
enforced structurally: `src/predict.py` checks the gate *before* running any
model, and red years short-circuit to `None` before a model is even loaded.

---

## 4. Building and validating the prediction models (Stage 2)

### 4.1 Shared preprocessing (`src/preprocess.py`)

One encoding path is used for both training and inference, because a mismatch
between the two is the single most common way ML systems silently break in
production. Categorical features (Sex, Race, Surgery_Type) are one-hot encoded;
numeric features are median-imputed, with the imputer fit only on training data
and only ever `.transform()`-ed at inference time.

### 4.2 The original cascade design (`scripts/reproduce_models.py`)

Twelve models were reproduced — TBWL% and FML%, years 1 through 6 — using the
best-performing model type for each outcome/year from the source table
(RandomForest, GradientBoosting, SVR, or XGBoost, depending on the year).

The key design decision here was how to handle the fact that year 2's outcome is
correlated with year 1's, year 3's with year 2's, and so on. The approach taken
was a **single-lag cascade**: each year's model includes the immediately
preceding year's actual outcome as one extra input feature — not every prior
year, just the last one. This was a deliberate tradeoff: requiring every prior
year to be non-missing (an "all-lags" design) collapses the usable training
sample for year 5–6 models to under 20 patients, because a patient needs *every*
intervening year measured, not just the target year. The single most recent lag
captures most of the predictive signal anyway.

At inference, this means: to predict year 3 for a brand-new patient, the model
needs a year-2 value — but a new preop patient doesn't have one yet. The
solution used at first was to **cascade the model's own year-1 prediction into
the year-2 model, then that prediction into year 3**, and so on. This detail
turns out to matter a great deal (§5).

Each of the 12 models was validated against the source table by refitting on
this codebase's own 5-fold cross-validation (`KFold(5, shuffle=True,
random_state=42)`) and checking the refit R² landed within a tolerance of the
published number. This gate is what makes the reproduction defensible rather
than just "trust me" — the script exits with an error if any model's refit
diverges too far, rather than silently continuing. Small-sample years (where
5-fold CV itself is noisy — 8 test patients per fold and a lot of variance) were
given a wider tolerance, documented and justified in `docs/METHODS_LOG.md` rather
than silently loosened.

### 4.3 Phenotyping (`src/phenotype.py`, `scripts/run_clustering.py`)

Patients are grouped into trajectory "phenotypes" via k-means clustering on
standardized TBWL% at years 1–3 (complete-case, N=149 patients with all three
years present).

This is worth being precise about, because it's a mix of "the data decided" and
"a person decided":

- **Which cluster a given patient falls into is genuinely data-driven** — real
  k-means fit on real trajectories, not hand-drawn boundaries.
- **The number of clusters (k=5) is a human override, not what the data
  prefers.** `scripts/run_clustering.py` computes the silhouette score (a
  measure of cluster separation quality) for k = 2 through 8, and the data's own
  answer is k=2. The code fits k=5 anyway, to match an existing manuscript. This
  is flagged directly in the code and surfaced to the end user every time a
  phenotype is shown ("provisional — pending confirmation") rather than buried.
- **Cluster labels 0–4 are then reassigned by ascending mean year-3 TBWL%**, so
  cluster 0 is always whichever group actually lost the least weight and cluster
  4 whichever lost the most — that ordering is real. The plain-English
  descriptions attached to each number in the Streamlit app ("minimal
  responder," "strongest responder") are hand-written prose glossing that
  ordering, not a statistical output themselves.

The fitted clustering model is frozen to a file (`artifacts/phenotype_kmeans.joblib`)
and loaded, never refit, on every subsequent call — so a given patient is
assigned consistently over time.

### 4.4 Explaining predictions (`src/explain.py`)

SHAP (SHapley Additive exPlanations) attributes each prediction to its input
features. Tree-based models (RandomForest, GradientBoosting, XGBoost) use
`TreeExplainer`, which is exact and fast. SVR — being a non-tree model — uses the
slower `KernelExplainer` against a 50-row representative background sample of
the training data, gated behind an explicit "Explain" button in the UI rather
than computed automatically, since it can take a minute or two.

### 4.5 What-if scenarios (`src/what_if.py`)

A thin wrapper that runs `predict_trajectory()` twice — once for the patient's
actual inputs, once with one or more features overridden — and returns the
year-by-year delta between the two, so a surgeon can ask "what if this patient
had reached 15% preop weight loss instead of 8%?"

---

## 5. The honesty audit — the most consequential phase

Once the 12 base models passed their validation gate, a second and more
important question was asked: **do the app's actual, deployed predictions match
the accuracy the source table claims?**

The gap: the 12 models were trained and scored using the *actual* prior-year
outcome as a lag feature. But a brand-new preop patient has no prior-year
outcome yet — the app has to *predict* year 1, then feed that prediction into
the year-2 model as if it were real, then feed that prediction into year 3, and
so on. This recursive cascade was never actually measured under its own
operating conditions; the source table's R² describes a different situation
(where the true lag is known).

`scripts/validate_models.py` runs five checks, all out-of-fold (5-fold, seed 42,
over all 802 patients):

1. **Cascade vs. oracle-lag accuracy.** Refits every model per fold, cascades
   predictions forward exactly as the app does, and compares that to the same
   model given the *true* prior-year lag. Result: cascaded preop accuracy
   collapses relative to the source table — TBWL year 2 dropped from a claimed
   R² of 0.52 to an actual 0.18 in deployment; year 4 dropped to **-0.02**,
   meaning the cascaded prediction was *worse than simply guessing the cohort
   average*. Meanwhile, the oracle-lag condition (true prior year known)
   reproduced or beat the source table, confirming the underlying models
   themselves were fine — the *forecasting strategy* was the problem.
2. **Uncertainty band coverage.** The app's displayed confidence interval
   (±1.96 × source-table RMSE) is supposed to contain the true outcome ~95% of
   the time. Measured empirically under the cascade, it only did so 78–89% of
   the time — the bands were overconfident.
3. **Naive baselines.** Once a prior-year actual value is known, simply carrying
   it forward ("predict year N = actual year N−1") beat the model's cascade
   prediction outright (RMSE 5.4–7.1 vs. 10.5–11.9). This reframed the tool's
   value proposition: the follow-up mode, not the preop-only mode, is where it
   earns its accuracy.
4. **Attrition bias.** Standardized mean differences between patients with vs.
   without late-year follow-up. Year-6 completers were measurably lighter and
   smaller at baseline than the full cohort (several features with |SMD| up to
   0.39) — independent confirmation that the red tier for year 6 is earned, not
   arbitrary.
5. **Subgroup residual bias.** Mean prediction error by sex and surgery type.
   Bypass patients were systematically under-predicted by ~4 percentage points
   at years 2–4; Sleeve patients slightly over-predicted. This is disclosed to
   the user rather than silently absorbed.

### What changed as a result

- **Uncertainty bands were recalibrated.** `src/predict.py` now tags every
  prediction with a `mode`: `"conditioned"` (year 1, or the immediate prior year
  was an actual measurement — the source-table RMSE band is trustworthy here) vs.
  `"cascade"` (the lag is itself a prediction). Cascade-mode years use empirical
  residual quantiles measured in the check above instead of the theoretical
  band, and the bands are allowed to be asymmetric, since they now absorb real
  bias rather than assuming symmetric Gaussian error.
- **Direct models replaced the cascade for preop-only prediction.**
  `scripts/train_direct_models.py` trains each year's model *directly* from
  preop features only — no recursive lag at all — which is the standard fix for
  compounding forecast error. It compares, out-of-fold, six configurations per
  year (15 baseline features vs. an expanded 35-feature set including labs and
  behavioral scores, crossed with RandomForest / HistGradientBoosting / Ridge),
  and adopts the direct model **only where it measurably beat the old cascade**.
  Adopted at TBWL years 2–4 and FML years 2 and 4; rejected (cascade kept) at
  TBWL year 5 and FML year 3, where nothing beat the existing approach. Result:
  preop-mode R² roughly doubled at several years, and the actively-wrong year-4
  prediction (negative R²) became a genuine, if modest, positive signal.
- **The reliability tiers themselves were left untouched.** The green/amber/red
  gate is a fixed rule from the source table (§3) and is treated as
  non-negotiable. Rather than re-tiering years based on the new cascade-honesty
  findings, the UI surfaces an additional "effective R²" caveat alongside the
  official tier, so a user sees both numbers rather than one being silently
  replaced by the other.

---

## 6. How a single prediction actually flows through the code

For a new patient entered into the Streamlit sidebar with no follow-up data yet:

```
app/streamlit_app.py (sidebar inputs: 15 baseline features
                       + optional labs/scores + optional postop actuals)
        │
        ▼
src/predict.py :: predict_trajectory(patient)
        │
        ├─ for each (outcome, year) in {TBWL, FML} × {1..6}:
        │     │
        │     ├─ src/reliability.py :: gate(outcome, year)
        │     │     → if RED: return {point: None, ...}, stop here
        │     │
        │     ├─ decide mode:
        │     │     "conditioned" if year==1 or prior-year actual supplied
        │     │     "direct"      if a direct model was adopted for this year (§5)
        │     │     "cascade"     otherwise (fallback — uses predicted lag)
        │     │
        │     ├─ src/preprocess.py :: encode_patient(...)
        │     │     (one-hot encode + align to the exact column order
        │     │      the model was trained on)
        │     │
        │     ├─ load frozen model + imputer from artifacts/
        │     │     (joblib files written by reproduce_models.py or
        │     │      train_direct_models.py; cached in memory after first load)
        │     │
        │     ├─ model.predict(x)  →  point estimate
        │     │
        │     └─ uncertainty band:
        │           "conditioned" → ±1.96 × source-table RMSE
        │           "cascade"/"direct" → empirical OOF residual quantiles
        │           (artifacts/calibration.joblib, from scripts/validate_models.py
        │            and scripts/train_direct_models.py)
        │
        ▼
returns: { "TBWL": {1: {...}, ..., 6: {...}}, "FML": {...} }
        │
        ├─→ src/phenotype.py :: assign_phenotype(...)   (uses predicted yrs 1-3)
        ├─→ src/risk.py      :: assess_preop_risk(...)  (rule on Preop_TBWL, no model)
        ├─→ src/explain.py   :: explain_with_shap(...)  (on demand, per year)
        └─→ src/what_if.py   :: what_if_analysis(...)   (re-runs predict_trajectory twice)
```

If the surgeon later enters an actual postop measurement (e.g., the real year-1
TBWL% from a follow-up visit), that value is passed into `postop_tbwl={1: ...}`,
which flips year 2 into `"conditioned"` mode — using the model's real, higher
accuracy — and the cascade only continues from year 3 onward.

---

## 7. Governance rules that shaped every design choice

A few rules were fixed at the start and constrained essentially every decision
above:

- **PHI never enters git or the LLM.** The CSV and `artifacts/` (except model
  weight files) are gitignored. Model-serving code runs entirely locally; only
  derived outputs (a predicted percentage, a cluster id, a feature name) are
  ever eligible to reach an LLM — never a raw patient row or the study `ID`.
- **Every random process is seeded (`random_state=42`)** so results reproduce
  exactly on a rerun.
- **Every non-obvious modeling choice gets one line in `docs/METHODS_LOG.md`,
  dated.** This write-up is the narrative expansion of that log; the log itself
  is the authoritative, chronological record.
- **The tool states uncertainty and defers to the care team.** Not a diagnostic
  or prescriptive system — red-tier years get no number, provisional phenotypes
  are labeled provisional, and known biases (Bypass under-prediction, year 5–6
  attrition) are disclosed in the interface rather than smoothed over.

---

## 8. Where to look for more detail

| Question | Where to look |
|---|---|
| Why was this specific model / tolerance / seed chosen? | `docs/METHODS_LOG.md` |
| What does the source performance table say? | `src/config.py` → `MODEL_PERFORMANCE` |
| How is a tier decided? | `src/reliability.py` |
| How does a single prediction get computed? | `src/predict.py` |
| How were the direct models chosen? | `scripts/train_direct_models.py` |
| What did the honesty audit actually find? | `scripts/validate_models.py` (run it — it prints the full table) |
| Is the phenotype clustering final? | No — see `src/phenotype.py` docstring and `CLAUDE.md` |
