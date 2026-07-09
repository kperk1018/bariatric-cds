"""Load the patient CSV with a light de-identification guard."""
import re
import pandas as pd
from src.config import DATA, ID_COL
from src.preprocess import clean_columns


def load(path=DATA, strict_unique_ids: bool = True) -> pd.DataFrame:
    """Load the patient CSV.

    Column names are standardized to the shared 1A/1B cleaning convention
    (see :func:`src.preprocess.clean_column_name`) immediately on load, so every
    downstream consumer reads identical feature names to Ioanna's pipeline.

    Args:
        path: CSV path.
        strict_unique_ids: if True (default), raise on duplicate patient IDs. Set
            False only to deliberately bypass the guard with intent (e.g. inspecting
            the raw, un-deduplicated file); never for training/clustering.
    """
    df = pd.read_csv(path)
    df = clean_columns(df)
    _deid_guard(df)
    if strict_unique_ids:
        _dup_id_guard(df)
    return df


def _deid_guard(df: pd.DataFrame) -> None:
    """Fail loudly if columns that look like direct identifiers slip in."""
    bad = [c for c in df.columns
           if re.search(r"name|mrn|dob|birth|ssn|address|phone|email|record", c, re.I)]
    if bad:
        raise ValueError(f"Possible PHI columns present, refuse to proceed: {bad}")


def _dup_id_guard(df: pd.DataFrame) -> None:
    """Fail loudly if any patient ID repeats.

    Duplicate IDs in this dataset are cartesian-product artifacts of an upstream
    merge on colliding IDs (e.g. ID 740 carries two different demographic profiles —
    Initial_BMI 44.70 and an impossible-for-bariatric 19.53 — each cross-joined with
    two outcome record sets = 4 rows). Left unchecked they contaminate cross-
    validation (same ID across folds), clustering, and validation. We refuse to
    proceed rather than silently dedup, so the merge is fixed upstream / aligned with
    Ioanna's pipeline. Bypass only with load(..., strict_unique_ids=False).
    """
    if ID_COL not in df.columns:
        raise ValueError(f"Expected id column '{ID_COL}' not found; cannot verify uniqueness.")
    counts = df[ID_COL].value_counts()
    dupes = counts[counts > 1]
    if len(dupes) > 0:
        n_extra = int(dupes.sum() - len(dupes))
        detail = ", ".join(f"{idx}(x{int(n)})" for idx, n in dupes.items())
        raise ValueError(
            f"Duplicate patient IDs found: {len(dupes)} id(s), {n_extra} extra row(s). "
            f"These are likely cartesian-product artifacts of an upstream merge on "
            f"colliding IDs and would corrupt CV/clustering/validation. Fix the merge "
            f"upstream (or align with Ioanna's dedup) before training. "
            f"Offending IDs: {detail}. "
            f"To inspect the raw file deliberately, call load(..., strict_unique_ids=False)."
        )
