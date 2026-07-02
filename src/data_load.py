"""Load the patient CSV with a light de-identification guard."""
import re
import pandas as pd
from src.config import DATA


def load(path=DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    _deid_guard(df)
    return df


def _deid_guard(df: pd.DataFrame) -> None:
    """Fail loudly if columns that look like direct identifiers slip in."""
    bad = [c for c in df.columns
           if re.search(r"name|mrn|dob|birth|ssn|address|phone|email|record", c, re.I)]
    if bad:
        raise ValueError(f"Possible PHI columns present, refuse to proceed: {bad}")
