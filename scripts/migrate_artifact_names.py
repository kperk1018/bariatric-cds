"""One-shot: rename stored column-name strings inside frozen model metas to the
cleaned 1A/1B convention, WITHOUT refitting.

Context: the per-year models were trained when the loader still emitted raw column
names (e.g. ``1yr_Postop_FML%``, ``Preop _chol``). data_load.load now standardizes
names on load. The estimators are numeric and name-agnostic, but predict.py /
explain.py match features by name against each meta's stored ``columns`` /
``lagged_cols`` / ``feature_cols`` / ``num_encoded``. This migrates only those
stored *label strings* so inference stays consistent — the models, and therefore
every prediction, are byte-for-byte unchanged.

Idempotent: re-running is a no-op once names are already clean.

Run:
    PYTHONPATH=. python scripts/migrate_artifact_names.py
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ARTIFACTS
from src.preprocess import clean_column_name

_LIST_KEYS = ("columns", "lagged_cols", "feature_cols", "num_encoded")


def _clean_list(x):
    return [clean_column_name(c) for c in x]


def main() -> None:
    metas = sorted(ARTIFACTS.glob("*_meta.joblib"))
    if not metas:
        print("No *_meta.joblib artifacts found — nothing to migrate.")
        return
    changed = 0
    for path in metas:
        meta = joblib.load(path)
        touched = False
        for key in _LIST_KEYS:
            if key in meta and isinstance(meta[key], list):
                cleaned = _clean_list(meta[key])
                if cleaned != meta[key]:
                    meta[key] = cleaned
                    touched = True
        if touched:
            joblib.dump(meta, path)
            changed += 1
            print(f"  migrated {path.name}")
    print(f"\nDone. {changed}/{len(metas)} metas updated "
          f"({len(metas) - changed} already clean).")


if __name__ == "__main__":
    main()
