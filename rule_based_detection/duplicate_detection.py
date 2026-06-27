"""
Rule 3 — Fuzzy duplicate invoice detection.

Two invoices are considered duplicates if:
    - same vendor_id (exact, used as blocking key)
    - amount within ±1% of each other
    - issue_date within ±7 days of each other

Implementation uses blocking on vendor_id to limit the O(n²) comparison
to within-vendor pairs only, then applies the fuzzy amount + date criteria.

For large datasets (>100k invoices), upgrade to recordlinkage with
sorted-neighbourhood blocking on (vendor_id, amount_bucket).
"""

import pandas as pd
import numpy as np
from itertools import combinations


AMOUNT_TOLERANCE_PCT = 0.01   # ±1%
DATE_TOLERANCE_DAYS  = 7


def _within_amount_tolerance(a1: float, a2: float, pct: float = AMOUNT_TOLERANCE_PCT) -> bool:
    avg = (a1 + a2) / 2
    if avg == 0:
        return False
    return abs(a1 - a2) / avg <= pct


def _within_date_tolerance(d1, d2, days: int = DATE_TOLERANCE_DAYS) -> bool:
    return abs((d1 - d2).days) <= days


def detect_fuzzy_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect fuzzy duplicate pairs using vendor-id blocking.

    Parameters
    ----------
    df : DataFrame with [invoice_id, vendor_id, amount, issue_date, is_fraud, fraud_category]

    Returns
    -------
    (flagged_invoices_df, pairs_df)
        flagged_invoices_df : original df with added columns:
            duplicate_flag, duplicate_pair_id, duplicate_partner_invoice_id
        pairs_df : one row per detected duplicate pair with match details
    """
    df = df.copy()
    df["issue_date"] = pd.to_datetime(df["issue_date"])

    duplicate_flags  = {}   # invoice_id -> (pair_id, partner_id)
    pairs            = []
    pair_counter     = 0

    # Block by vendor_id — only compare invoices from same vendor
    for vendor_id, grp in df.groupby("vendor_id"):
        grp = grp.sort_values("issue_date").reset_index(drop=True)
        n   = len(grp)
        if n < 2:
            continue

        ids     = grp["invoice_id"].values
        amounts = grp["amount"].values
        dates   = grp["issue_date"].values

        # Only look at pairs within a ±15 day window (sorted → early exit)
        for i in range(n):
            for j in range(i + 1, n):
                date_diff = abs((dates[j] - dates[i]).astype("timedelta64[D]").astype(int))
                if date_diff > DATE_TOLERANCE_DAYS:
                    break   # sorted by date → no later j will be within window

                if _within_amount_tolerance(float(amounts[i]), float(amounts[j])):
                    pair_id = f"DUP-{pair_counter:05d}"
                    pair_counter += 1

                    # Tag both invoices
                    for inv, partner in [(ids[i], ids[j]), (ids[j], ids[i])]:
                        if inv not in duplicate_flags:
                            duplicate_flags[inv] = (pair_id, partner)

                    # Get fraud labels for the pair
                    row_i = grp.iloc[i]
                    row_j = grp.iloc[j]
                    pairs.append({
                        "pair_id":                pair_id,
                        "invoice_id_a":           ids[i],
                        "invoice_id_b":           ids[j],
                        "vendor_id":              vendor_id,
                        "amount_a":               float(amounts[i]),
                        "amount_b":               float(amounts[j]),
                        "amount_diff_pct":        round(abs(float(amounts[i]) - float(amounts[j])) /
                                                        max((float(amounts[i]) + float(amounts[j])) / 2, 1e-9), 6),
                        "issue_date_a":           pd.Timestamp(dates[i]).date(),
                        "issue_date_b":           pd.Timestamp(dates[j]).date(),
                        "date_diff_days":         date_diff,
                        "either_is_fraud":        bool(row_i["is_fraud"]) or bool(row_j["is_fraud"]),
                        "fraud_category_a":       row_i["fraud_category"],
                        "fraud_category_b":       row_j["fraud_category"],
                    })

    # Annotate original dataframe
    df["duplicate_flag"]               = df["invoice_id"].isin(duplicate_flags)
    df["duplicate_pair_id"]            = df["invoice_id"].map(
        lambda x: duplicate_flags[x][0] if x in duplicate_flags else None
    )
    df["duplicate_partner_invoice_id"] = df["invoice_id"].map(
        lambda x: duplicate_flags[x][1] if x in duplicate_flags else None
    )

    pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame()
    return df, pairs_df


def duplicate_summary(pairs_df: pd.DataFrame) -> dict:
    """Quick stats on detected duplicate pairs."""
    if pairs_df.empty:
        return {"total_pairs": 0}
    return {
        "total_pairs":          len(pairs_df),
        "fraud_pairs":          int(pairs_df["either_is_fraud"].sum()),
        "clean_pairs":          int((~pairs_df["either_is_fraud"]).sum()),
        "avg_amount_diff_pct":  round(pairs_df["amount_diff_pct"].mean() * 100, 4),
        "avg_date_diff_days":   round(pairs_df["date_diff_days"].mean(), 2),
    }
