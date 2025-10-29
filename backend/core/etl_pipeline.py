# etl_pipeline.py
import pandas as pd
import numpy as np

DATE_HINTS = ("date","year","month","time")

def run_etl(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # normalize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # try parse date-like columns
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in DATE_HINTS):
            with pd.option_context("mode.chained_assignment", None):
                try: df[c] = pd.to_datetime(df[c], errors="ignore")
                except: pass

    # numeric coercion (don’t break categorical)
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c])  # will raise if not numeric
            except: pass

    # drop constant columns
    nunique = df.nunique(dropna=False)
    keep = [c for c in df.columns if nunique[c] > 1]
    return df[keep]