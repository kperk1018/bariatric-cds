"""Load the patient CSV: column cleaning, de-id guard, and keep-first dedup."""
import re
import warnings
import pandas as pd
from src.config import DATA, ID_COL
from src.preprocess import clean_columns


def load(path=DATA, dedup: bool = True) -> pd.DataFrame:
    """Load the patient CSV.

    Column names are standardized to the shared 1A/1B cleaning convention
    (see :func:`src.preprocess.clean_column_name`) immediately on load, so every
    downstream consumer reads identical feature names to Ioanna's pipeline.

    Duplicate patient IDs are removed with ``keep="first"`` to match 1A
    (Ioanna's ``drop_duplicates(subset=["ID"], keep="first")``). This dataset has
    6 IDs carrying cartesian-product duplicate rows from an upstream merge on
    colliding IDs; keep-first empirically retains the plausible-BMI row (it drops
    the impossible Initial_BMI 19.53 for ID 740). The count is reported via a
    warning, never silently swallowed.

    Args:
        path: CSV path.
        dedup: if True (default), apply keep-first dedup. Set False only to inspect
            the raw, un-deduplicated file (e.g. auditing the duplicates themselves).
    """
    df = pd.read_csv(path)
    df = clean_columns(df)
    _deid_guard(df)
    n_dupes = _count_duplicate_ids(df)  # internal check — reports, does not raise
    if n_dupes and dedup:
        df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)
    return df


def _deid_guard(df: pd.DataFrame) -> None:
    """Fail loudly if columns that look like direct identifiers slip in."""
    bad = [c for c in df.columns
           if re.search(r"name|mrn|dob|birth|ssn|address|phone|email|record", c, re.I)]
    if bad:
        raise ValueError(f"Possible PHI columns present, refuse to proceed: {bad}")


def _count_duplicate_ids(df: pd.DataFrame) -> int:
    """Warn about duplicate patient IDs and return the count of extra rows.

    Retained as an internal integrity check (per the 1A-alignment decision we match
    Ioanna's keep-first dedup rather than fail-loud, so this reports instead of
    raising). Duplicates here are cartesian-product artifacts of an upstream merge
    on colliding IDs.
    """
    if ID_COL not in df.columns:
        raise ValueError(f"Expected id column '{ID_COL}' not found; cannot verify uniqueness.")
    counts = df[ID_COL].value_counts()
    dupes = counts[counts > 1]
    if len(dupes) == 0:
        return 0
    n_extra = int(dupes.sum() - len(dupes))
    detail = ", ".join(f"{idx}(x{int(n)})" for idx, n in dupes.items())
    warnings.warn(
        f"Duplicate patient IDs: {len(dupes)} id(s), {n_extra} extra row(s) "
        f"[{detail}] — removed via keep='first' (matches 1A). These are "
        f"cartesian-product merge artifacts.",
        stacklevel=2,
    )
    return n_extra
