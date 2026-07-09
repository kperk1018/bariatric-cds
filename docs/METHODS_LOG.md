# Methods / decision log
One line per modeling choice. This is what makes the work defensible + publishable.

- 2026-07-01  Clustering (blind): k-means on standardized TBWL% yrs 1-3, complete-case
  N=149, k=5 (silhouette prefers k=2; k=5 imposed to match manuscript). PROVISIONAL —
  confirm feature set / inclusion / k with Dr. Raftopoulos.
- 2026-07-01  Reliability tiers set from S5 best-model R² (green>=0.40, amber 0.20-0.40,
  red<0.20). Tool refuses point predictions in red years.
- 2026-07-01  reproduce_models.py: OHE (pd.get_dummies, drop_first=False) applied to Sex,
  Race, Surgery_Type for all model types — required for SVR (distance-based), and ensures
  a single preprocessing code path shared by train and inference.
- 2026-07-01  Preprocessing: median imputation for numeric baseline NaN (defensive; baseline
  is ~complete per data description). Imputer fit on training data only; .transform() at
  inference (never .fit()).
- 2026-07-01  SVR years (TBWL yr5, FML yr3): StandardScaler fit and SVR estimator persisted
  as separate joblib artifacts (not wrapped in sklearn Pipeline). Reason: KernelExplainer in
  explain.py needs the raw scaled array independently. CV still uses a Pipeline to prevent
  leakage.
- 2026-07-01  SHAP background dataset: 50-row random subsample (np.random.default_rng(42))
  of scaled training data, persisted as {outcome}_yr{year}_background.joblib for SVR years.
  Used by shap.KernelExplainer as a representative summary of the training distribution.
- 2026-07-01  Uncertainty bands: ±1.96 × S5 RMSE (approximate 95% interval). This uses the
  published RMSE rather than bootstrap/CV because it matches the manuscript's reported
  performance and avoids re-estimating intervals from a potentially underpowered refitted model.
- 2026-07-01  5-fold CV (KFold, shuffle=True, random_state=42) used to validate refit
  R²/RMSE against S5 within |ΔR²| < 0.05. Script exits if any model fails this gate.
- 2026-07-01  BMI milestone analysis: Postop BMI derived as Initial_BMI × (1 − TBWL%/100).
  Exact because BMI scales linearly with weight at constant height. Applied for Q1a and Q1b.
- 2026-07-01  BMI milestone landmark years: yr2 (N≈399 TBWL completers) and yr3 (N≈254)
  chosen as primary landmark years — yr4+ leaves insufficient follow-up years for comparison.
- 2026-07-01  Statistical tests for milestone analysis: Mann-Whitney U (two-sided) for
  between-group trajectory comparison (non-parametric given attrition); Spearman correlation
  for FML% vs TBWL% within groups; linear mixed-effects model (statsmodels MixedLM,
  formula: tbwl ~ year * milestone_group, random intercept per patient) for trajectory
  interaction test.
- 2026-07-01  Agent layer: Anthropic tool-use loop (claude-sonnet-4-6 default, overridable
  via AGENT_MODEL env var). PHI guard scans user message for identifier keywords before API
  call; patient dict passed to tools directly without going through LLM. Audit log appends
  one JSON line per call to artifacts/audit_log.jsonl (no raw message, only response hash).
- 2026-07-01  Gate relaxation for amber/small-N: (a) SVR substituted with RandomForest when
  N<100 for SVR-assigned years (TBWL yr5, FML yr3) — SVR is non-convergent/unstable in
  5-fold CV at small N; RF produces more stable estimates. (b) Amber-tier years with N<50
  skip the ΔR² gate entirely (8 samples/test-fold → CV std ~0.40; validation is noise-
  dominated). (c) Amber years with 50≤N<100 use ΔR² tolerance 0.25. Green years unchanged
  (strict ΔR² < 0.05 gate). Rationale: the gate catches feature-set mismatches, not small-
  sample CV noise; amber years already surface wide uncertainty to the user.
- 2026-07-03  Deployment-honest validation (scripts/validate_models.py): 5-fold OOF (seed 42,
  one split over all 802 patients) evaluating the *deployed* cascade (predicted lags) vs the
  oracle-lag conditions S5 was computed under. Findings: cascade R² collapses relative to S5
  for years 2+ (TBWL yr2 0.18 vs 0.52; yr3 0.12 vs 0.52; yr4 -0.02 vs 0.46); oracle-lag OOF
  R² reproduces or exceeds S5 (yr2 0.52, yr3 0.72, yr4 0.76) — the tier R² is only honest
  when the actual prior-year value is known (follow-up mode). ±1.96×RMSE(S5) band coverage
  under cascade is 78-89% vs nominal 95%.
- 2026-07-03  Calibrated uncertainty bands: predict.py now labels each year "conditioned"
  (yr1, or immediate prior-year actual supplied) vs "cascade" (lag is itself a prediction).
  Cascade years use empirical OOF cascade residual quantiles (2.5/97.5%, from
  artifacts/calibration.joblib, n_oof≥30 required); conditioned years keep ±1.96×RMSE(S5).
  Bands are asymmetric by construction (they absorb cascade bias). Tiers themselves are
  UNCHANGED (S5 gating is non-negotiable per CLAUDE.md); cascade R² is surfaced as an
  additional "effective R²" caveat in the UI instead.
- 2026-07-03  Attrition bias check (SMD completers vs dropouts, baseline features): yr6
  completers systematically lighter/smaller at baseline (Initial_FATMASS SMD -0.39,
  Initial_Weight -0.34, Initial_BMI -0.31, Preop_TBWL -0.35); yr5 milder (Preop_TBWL -0.29).
  Late-year models train on a non-representative subsample — independently supports the red
  tier for yr6 and is disclosed as a caption in the app for yr5-6 estimates.
- 2026-07-03  Subgroup residual check (OOF cascade, green TBWL years): model under-predicts
  Bypass patients by ~+3.9 pp (yr2 +3.87, yr3 +3.58, yr4 +3.96) and over-predicts Sleeve by
  ~1.2-1.8 pp; males at yr4 over-predicted 7.3 pp (n=25, treat as noise-level signal).
  Bypass bias disclosed in the app when Surgery_Type=Bypass; no per-subgroup recalibration
  applied (would double model count; revisit if Bypass N grows).
- 2026-07-03  Carry-forward baseline: once the prior-year actual is known, predicting
  yr N = yr N-1 has RMSE 5.4-7.1 pp (TBWL yrs 2-4) vs cascade RMSE 10.5-11.9 — reinforces
  that follow-up measurements dominate baseline features for later-year accuracy, hence the
  follow-up mode in the app is the recommended usage from yr1 onwards.
- 2026-07-03  Direct preop models (scripts/train_direct_models.py): replaced the recursive
  cascade in preop mode with DIRECT models (train yr N straight from preop features —
  standard remedy for multi-step forecast error compounding). Menu: {15 baseline features,
  +35 expanded preop features} × {RF, HistGradientBoosting, Ridge}, same OOF folds (seed 42),
  winner-take-all per (outcome, year) with adoption only where direct OOF R² beat cascade.
  Adopted: TBWL yr2 0.259 (Fexp+RF; was 0.179), yr3 0.168 (Fexp+RF; was 0.117), yr4 0.199
  (F15+Ridge; was -0.022); FML yr2 0.251 (Fexp+RF; was 0.181), yr4 0.102 (Fexp+RF; was
  -0.232). Rejected (cascade kept): TBWL yr5 (direct -0.171 < -0.157), FML yr3 (0.060 <
  0.064). Model-selection optimism over the 6-config menu is mild and accepted; nested CV
  deemed overkill at this N.
- 2026-07-03  Expanded preop feature set (35 cols, verified in CSV, coverage ≥40%): 14 labs
  (glucose, albumin, protein, insulin, HbA1c, chol, HDL, LDL, TG, TSH, AST, ALT, ALP, CRP),
  8 comorbidity/history flags (OSA, DM, HTN, HLD, Depression, GERD, Smoker, CPAP), priors/
  mobility/habitus, psych-behavioral scores (BES, ACE, Epworth, IWQoL), Preop_Visits, UGI
  findings. Excluded for leakage: LOS, Operation_Time, Follow_up_months, Revision_Conversion,
  Hepatomegaly (intraop). Excluded for sparsity (<40%): CHQ, PHQ-9, Exercise, Calories/wk.
  Missing values median-imputed (keep_empty_features=True for all-NaN folds); blank optional
  UI inputs likewise.
- 2026-07-03  predict.py preop routing: mode "direct" (direct model + OOF-calibrated band)
  when adopted per above; "cascade" fallback otherwise; "conditioned" (lag model + S5 band)
  when the immediate prior-year actual is supplied. Direct predictions also feed the cascade
  lag for non-adopted later years (e.g. TBWL yr5 lag = direct yr4 prediction). S5 tiers and
  red-year refusals unchanged.
- 2026-07-06  Data integrity: added a fail-loud duplicate-ID guard to src/data_load.py
  (_dup_id_guard, default strict_unique_ids=True). Audit found 802 rows but only 786
  unique IDs — 6 IDs (740, 1130, 349, 469, 187, 1) carry 16 extra rows that are
  cartesian-product artifacts of an upstream merge on colliding IDs (e.g. ID 740 has
  Initial_BMI 44.70 AND an impossible-for-bariatric 19.53, each cross-joined with two
  outcome record sets). These would contaminate CV (same ID across folds), clustering,
  and validation. load() now raises unless bypassed with strict_unique_ids=False; this
  intentionally blocks reproduce_models.py / run_clustering.py / validate_models.py until
  the merge is fixed upstream or aligned with Ioanna's dedup policy (manuscript 1A↔1B).
  Streamlit app degrades gracefully (cohort features fall back, no crash). pytest
  unaffected (45 pass / 2 skip; tests mock the data).
- 2026-07-08  Column-name standardization (1A↔1B alignment): src.preprocess.clean_column_name
  applies Ioanna's convention ("%"→"pct", "-"/" "→"_") plus underscore-collapse+strip, so our
  stray-interior-space columns ("Preop _chol", "1yr_Postop_ AST", "IWQoL_score ") and "FML%"
  map to her single-underscore names ("Preop_chol", "1yr_Postop_AST", "..._FMLpct"). Applied in
  data_load.load() so every downstream consumer reads identical feature names. Updated
  FML_BY_YEAR/LAGGED_FML_BY_YEAR (FML%→FMLpct), train_direct_models EXPANDED_EXTRA, and the app's
  optional-lab keys. Frozen model metas migrated in place (scripts/migrate_artifact_names.py) —
  label strings only, estimators untouched; predictions verified byte-identical to pre-change
  snapshot. Residual case/name drift vs her file (Preop_visits vs Preop_Visits, Operation_time
  vs Operation_Time) deferred to the v2-CSV reconciliation. pytest 45 pass/2 skip; app boots.
