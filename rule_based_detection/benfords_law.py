"""
Rule 1 — Benford's Law test on invoice amounts.

Benford's Law predicts the distribution of leading digits in naturally
occurring financial data. The expected probability of leading digit d is:
    P(d) = log10(1 + 1/d)

Genuine invoice amounts follow this closely. Fabricated amounts (round
numbers, repeated patterns) deviate significantly.

Output per vendor:
    - observed leading-digit counts
    - chi-squared statistic vs. Benford distribution
    - p-value
    - flag: chi2 > threshold (default 15.51 = chi2 critical at df=8, p=0.05)
    - minimum count guard: skip vendors with < 20 invoices (unreliable chi2)
"""

import numpy as np
import pandas as pd
from scipy import stats

# Benford's expected probabilities for digits 1–9
BENFORD_PROBS = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
BENFORD_PROBS /= BENFORD_PROBS.sum()   # normalise (should already sum to ~1)

# Chi-squared critical value: df=8, alpha=0.05 → 15.507
CHI2_CRITICAL_05 = stats.chi2.ppf(0.95, df=8)
CHI2_CRITICAL_01 = stats.chi2.ppf(0.99, df=8)   # stricter: 20.090

MIN_INVOICES = 20   # minimum invoices per vendor for reliable chi2 test


def _leading_digit(amount: float) -> int | None:
    """Return the first significant digit (1–9) of an amount, or None."""
    if amount <= 0:
        return None
    s = f"{amount:.10f}".lstrip("0").replace(".", "")
    for ch in s:
        if ch != "0":
            return int(ch)
    return None


def compute_vendor_benford_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Benford's Law chi-squared deviation score per vendor.

    Parameters
    ----------
    df : DataFrame with columns [vendor_id, amount, is_fraud, fraud_category]

    Returns
    -------
    DataFrame with one row per vendor:
        vendor_id, n_invoices, chi2_stat, p_value, benford_flag,
        benford_flag_strict, observed_d1..d9, expected_d1..d9, mse
    """
    df = df.copy()
    df["leading_digit"] = df["amount"].apply(_leading_digit)
    df = df.dropna(subset=["leading_digit"])
    df["leading_digit"] = df["leading_digit"].astype(int)

    rows = []
    for vendor_id, grp in df.groupby("vendor_id"):
        n = len(grp)
        if n < MIN_INVOICES:
            continue

        # Observed counts for digits 1–9
        counts = grp["leading_digit"].value_counts().reindex(range(1, 10), fill_value=0).values
        expected = BENFORD_PROBS * n

        # Chi-squared: use scipy directly (handles near-zero expected cells)
        chi2, p_val = stats.chisquare(counts, f_exp=expected)

        # Mean squared error of proportions (scale-invariant alternative)
        obs_prop = counts / n
        mse = np.mean((obs_prop - BENFORD_PROBS) ** 2)

        row = {
            "vendor_id":          vendor_id,
            "n_invoices":         n,
            "chi2_stat":          round(chi2, 4),
            "p_value":            round(p_val, 6),
            "benford_flag":       chi2 > CHI2_CRITICAL_05,
            "benford_flag_strict":chi2 > CHI2_CRITICAL_01,
            "mse_proportion":     round(mse, 8),
        }
        # Observed & expected per digit
        for i, d in enumerate(range(1, 10)):
            row[f"obs_d{d}"] = int(counts[i])
            row[f"exp_d{d}"] = round(expected[i], 2)

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values("chi2_stat", ascending=False).reset_index(drop=True)
    return result


def join_benford_flags_to_invoices(
    invoices: pd.DataFrame,
    benford_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Propagate vendor-level Benford flags down to invoice level.
    Vendors with < MIN_INVOICES get benford_flag = False (untestable).
    """
    flags = benford_scores[["vendor_id", "chi2_stat", "p_value",
                             "benford_flag", "benford_flag_strict",
                             "mse_proportion"]].copy()
    return invoices.merge(flags, on="vendor_id", how="left").fillna(
        {"benford_flag": False, "benford_flag_strict": False,
         "chi2_stat": 0.0, "p_value": 1.0, "mse_proportion": 0.0}
    )
