"""
Rule 4 — Shell vendor detection via shared bank accounts.

A vendor is flagged as a shell if its bank_account_id is shared with
at least one other vendor_id. This is a strong structural signal:
legitimate businesses rarely share routing/account numbers.

Additional heuristics layered on top:
    - EIN root clustering (first 6 chars of tax_id shared across ≥2 vendors)
    - High wire-transfer ratio (>90% of payments via Wire)
    - High-risk jurisdiction (BVI, KY, etc.)
    - Short time between vendor creation and first invoice (<60 days)
"""

import pandas as pd
import numpy as np


HIGH_RISK_COUNTRIES = {"BVI", "KY", "PA", "SC", "VG"}
WIRE_RATIO_THRESHOLD      = 0.90
MIN_VENDORS_SHARING_BANK  = 2
NEW_VENDOR_DAYS_THRESHOLD = 60   # days from creation to first invoice


def detect_shell_vendors(
    invoices_df: pd.DataFrame,
    vendors_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect shell vendor clusters.

    Parameters
    ----------
    invoices_df : DataFrame with [invoice_id, vendor_id, amount, issue_date,
                                   payment_method, is_fraud, fraud_category]
    vendors_df  : DataFrame with [vendor_id, bank_account, tax_id, country,
                                   vendor_created_at, shell_cluster_id]

    Returns
    -------
    (flagged_invoices_df, vendor_risk_df)
        flagged_invoices_df : invoices with shell_vendor_flag + score columns
        vendor_risk_df      : one row per vendor with risk breakdown
    """
    inv = invoices_df.copy()
    ven = vendors_df.copy()

    # ── Vendor-level stats from invoices ──────────────────────────────────────
    inv["payment_method_lower"] = inv["payment_method"].str.lower()
    vendor_stats = (
        inv.groupby("vendor_id")
        .agg(
            total_invoices      = ("invoice_id",      "count"),
            wire_count          = ("payment_method_lower",
                                   lambda x: (x == "wire").sum()),
            total_amount        = ("amount",           "sum"),
            first_invoice_date  = ("issue_date",       "min"),
        )
        .reset_index()
    )
    vendor_stats["wire_ratio"] = (
        vendor_stats["wire_count"] / vendor_stats["total_invoices"].clip(lower=1)
    )

    # ── Bank account sharing ──────────────────────────────────────────────────
    bank_sharing = (
        ven[ven["bank_account"].notna()]
        .groupby("bank_account")
        .agg(vendors_sharing_bank=("vendor_id", "nunique"))
        .reset_index()
    )

    # ── EIN root clustering ────────────────────────────────────────────────────
    ven["ein_root"] = ven["tax_id"].astype(str).str[:6]
    ein_sharing = (
        ven.groupby("ein_root")
        .agg(vendors_sharing_ein=("vendor_id", "nunique"))
        .reset_index()
    )

    # ── Assemble vendor risk table ────────────────────────────────────────────
    vr = ven.merge(vendor_stats, on="vendor_id", how="left")
    vr = vr.merge(bank_sharing,  on="bank_account", how="left")
    vr = vr.merge(ein_sharing,   on="ein_root",     how="left")

    # Days vendor to first invoice
    vr["vendor_created_at"]  = pd.to_datetime(vr["vendor_created_at"])
    vr["first_invoice_date"] = pd.to_datetime(vr["first_invoice_date"])
    vr["days_to_first_inv"]  = (
        vr["first_invoice_date"] - vr["vendor_created_at"]
    ).dt.days

    # ── Risk scoring (0–5) ────────────────────────────────────────────────────
    vr["score_shared_bank"]  = (vr["vendors_sharing_bank"].fillna(1) >= MIN_VENDORS_SHARING_BANK).astype(int) * 2
    vr["score_shared_ein"]   = (vr["vendors_sharing_ein"].fillna(1)  >= 2).astype(int)
    vr["score_wire_ratio"]   = (vr["wire_ratio"].fillna(0)           >= WIRE_RATIO_THRESHOLD).astype(int)
    vr["score_high_risk_ctry"]= vr["country"].str.upper().isin(HIGH_RISK_COUNTRIES).astype(int)
    vr["score_new_vendor"]   = (
        vr["days_to_first_inv"].notna() &
        (vr["days_to_first_inv"] <= NEW_VENDOR_DAYS_THRESHOLD)
    ).astype(int)

    vr["shell_risk_score"] = (
        vr["score_shared_bank"]
        + vr["score_shared_ein"]
        + vr["score_wire_ratio"]
        + vr["score_high_risk_ctry"]
        + vr["score_new_vendor"]
    )

    # Primary flag: shared bank account (structural, highest precision)
    vr["shell_bank_flag"]  = vr["vendors_sharing_bank"].fillna(1) >= MIN_VENDORS_SHARING_BANK
    # Composite flag: score >= 3 (bank + any other signal)
    vr["shell_composite_flag"] = vr["shell_risk_score"] >= 3

    # ── Propagate to invoice level ────────────────────────────────────────────
    flag_cols = vr[["vendor_id", "shell_bank_flag", "shell_composite_flag",
                    "shell_risk_score", "vendors_sharing_bank",
                    "score_shared_ein", "score_wire_ratio",
                    "score_high_risk_ctry", "score_new_vendor"]]

    inv_flagged = inv.merge(flag_cols, on="vendor_id", how="left")
    inv_flagged["shell_bank_flag"]      = inv_flagged["shell_bank_flag"].fillna(False)
    inv_flagged["shell_composite_flag"] = inv_flagged["shell_composite_flag"].fillna(False)
    inv_flagged["shell_risk_score"]     = inv_flagged["shell_risk_score"].fillna(0)

    return inv_flagged, vr


def shell_cluster_summary(vendor_risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group flagged shell vendors by bank_account to show clusters.
    """
    flagged = vendor_risk_df[vendor_risk_df["shell_bank_flag"]].copy()
    if flagged.empty:
        return pd.DataFrame()

    return (
        flagged.groupby("bank_account")
        .agg(
            vendor_count        = ("vendor_id",        "count"),
            vendor_ids          = ("vendor_id",         list),
            total_invoices      = ("total_invoices",    "sum"),
            total_amount        = ("total_amount",      "sum"),
            countries           = ("country",           lambda x: list(set(x))),
            max_shell_risk_score= ("shell_risk_score",  "max"),
            has_known_cluster   = ("shell_cluster_id",  lambda x: x.notna().any()),
        )
        .reset_index()
        .sort_values("vendor_count", ascending=False)
    )
