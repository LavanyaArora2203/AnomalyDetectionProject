"""
Rule 2 — Threshold-gaming detection.

Definition: an invoice is threshold-gaming if:
    1. Its amount is within 2% BELOW any known approval threshold, AND
    2. The same vendor has ≥ 3 such invoices targeting the same threshold
       within the same calendar quarter.

Approval thresholds (configurable):
    $5,000 / $10,000 / $25,000 / $50,000 / $100,000

Why quarterly window: gaming patterns cluster in time as fraudsters
exploit the same approval limit repeatedly before controls adapt.
"""

import pandas as pd
import numpy as np

APPROVAL_THRESHOLDS = [5_000, 10_000, 25_000, 50_000, 100_000]
BAND_PCT            = 0.02    # 2% below threshold
MIN_OCCURRENCES     = 3       # per vendor per threshold per quarter


def _nearest_threshold_below(amount: float) -> tuple[float | None, float | None]:
    """
    Return (threshold, gap_pct) if amount falls within BAND_PCT below a threshold.
    gap_pct = (threshold - amount) / threshold
    """
    for t in APPROVAL_THRESHOLDS:
        lower = t * (1 - BAND_PCT)
        if lower <= amount < t:
            gap_pct = (t - amount) / t
            return t, round(gap_pct, 6)
    return None, None


def detect_threshold_gaming(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag threshold-gaming invoices.

    Parameters
    ----------
    df : DataFrame with columns:
         [invoice_id, vendor_id, amount, issue_date, is_fraud, fraud_category]

    Returns
    -------
    df with added columns:
        nearest_threshold_2pct  – threshold hit (None if not in zone)
        gap_pct                 – how far below (0–0.02)
        vendor_qtr_threshold_count  – vendor occurrences same threshold × quarter
        threshold_gaming_flag   – True if count ≥ MIN_OCCURRENCES
    """
    df = df.copy()
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["_year"]  = df["issue_date"].dt.year
    df["_qtr"]   = df["issue_date"].dt.quarter

    # Tag each invoice with its nearest threshold (if in zone)
    parsed = df["amount"].apply(_nearest_threshold_below)
    df["nearest_threshold_2pct"] = parsed.apply(lambda x: x[0])
    df["gap_pct"]                = parsed.apply(lambda x: x[1])

    # Count vendor × threshold × quarter occurrences
    zone = df[df["nearest_threshold_2pct"].notna()].copy()

    if zone.empty:
        df["vendor_qtr_threshold_count"] = 0
        df["threshold_gaming_flag"]      = False
        return df

    counts = (
        zone.groupby(["vendor_id", "nearest_threshold_2pct", "_year", "_qtr"])
        .size()
        .reset_index(name="vendor_qtr_threshold_count")
    )

    df = df.merge(
        counts,
        on=["vendor_id", "nearest_threshold_2pct", "_year", "_qtr"],
        how="left",
    )
    df["vendor_qtr_threshold_count"] = (
        df["vendor_qtr_threshold_count"].fillna(0).astype(int)
    )
    df["threshold_gaming_flag"] = (
        df["nearest_threshold_2pct"].notna()
        & (df["vendor_qtr_threshold_count"] >= MIN_OCCURRENCES)
    )

    return df.drop(columns=["_year", "_qtr"])


def threshold_gaming_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table: one row per (vendor_id, threshold, quarter) combo
    that crosses the MIN_OCCURRENCES threshold.
    """
    flagged = df[df["threshold_gaming_flag"]].copy()
    flagged["quarter"] = (
        pd.to_datetime(flagged["issue_date"]).dt.to_period("Q").astype(str)
    )

    return (
        flagged.groupby(["vendor_id", "nearest_threshold_2pct", "quarter"])
        .agg(
            invoice_count=("invoice_id", "count"),
            total_amount=("amount", "sum"),
            min_gap_pct=("gap_pct", "min"),
            max_gap_pct=("gap_pct", "max"),
            fraud_count=("is_fraud", "sum"),
        )
        .reset_index()
        .sort_values("invoice_count", ascending=False)
    )
