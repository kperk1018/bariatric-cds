# Bariatric Clinical Decision Support (research prototype)

Internal research prototype: predicts post-bariatric TBWL%/FML% trajectories, assigns
trajectory phenotypes, flags preop-weight-loss risk, explains drivers with SHAP, and
(later) exposes these as tools to a natural-language agent.

**Intended use: internal quality-improvement / research only. Not a medical device,
not patient-facing.** See CLAUDE.md for the full spec and hard rules.

## Setup
    pip install -r requirements.txt
    # place the de-identified CSV at data/patients-data5b-may11data-merge-edited-v2.csv
    python scripts/run_clustering.py
    pytest -q

Data and model artifacts are gitignored and must never be committed.
