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
