"""
Rule-based detection layer — main runner.
Week 4 deliverable: deterministic baseline all ML models must beat.

Runs all four rules, evaluates precision/recall, produces:
    - rule_scores.csv       : invoice-level flag table (one column per rule)
    - rule_metrics.csv      : per-rule precision/recall/F1
    - ensemble_metrics.csv  : ensemble at vote thresholds 1..4
    - category_breakdown.csv: per-rule performance split by fraud category
    - threshold_curve_*.csv : precision/recall at multiple score cutoffs
    - benford_vendor_scores.csv : per-vendor Benford chi2 scores
    - duplicate_pairs.csv   : detected duplicate pairs
    - shell_clusters.csv    : shell vendor cluster summary

Usage
-----
    python run_rules.py                        # full run
    python run_rules.py --rules benford shell  # specific rules only
    python run_rules.py --no-export            # print only, no CSV
"""

import argparse, sys, time
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

# Rule modules
from benfords_law        import compute_vendor_benford_scores, join_benford_flags_to_invoices
from threshold_gaming    import detect_threshold_gaming, threshold_gaming_summary
from duplicate_detection import detect_fuzzy_duplicates, duplicate_summary
from shell_vendor        import detect_shell_vendors, shell_cluster_summary
from evaluation          import (
    evaluate_binary_rule, evaluate_at_thresholds,
    per_category_metrics, evaluate_ensemble, print_rule_summary,
)

DB_PATH = "invoice_anomaly\dev.duckdb"
OUT_DIR = Path("rule_based_detection/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAUD_CATEGORIES = ["duplicate", "threshold_gaming", "shell_vendor", "soft_anomaly"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    print("  Loading data from DuckDB …")
    con = duckdb.connect(DB_PATH, read_only=True)
    invoices = con.execute("""
        SELECT
            fe.invoice_id, fe.vendor_id, fe.buyer_id,
            fe.amount, fe.issue_date,
            fe.is_fraud, fe.fraud_category, fe.anomaly_type,
            fe.vendor_category, fe.status, fe.payment_method,
            si.invoice_number, si.created_at,
            sv.bank_account, sv.shell_cluster_id, sv.tax_id,
            sv.country, sv.vendor_created_at
        FROM main_mart.feat_master  fe
        JOIN main_staging.stg_invoices  si ON fe.invoice_id = si.invoice_id
        JOIN main_staging.stg_vendors   sv ON fe.vendor_id  = sv.vendor_id
    """).df()

    vendors = con.execute("""
        SELECT vendor_id, bank_account, tax_id, country,
               vendor_created_at, shell_cluster_id, is_shell_vendor
        FROM main_staging.stg_vendors
    """).df()
    con.close()
    print(f"  Loaded {len(invoices):,} invoices, {len(vendors):,} vendors\n")
    return invoices, vendors


# ── Rule runners ──────────────────────────────────────────────────────────────

def run_benford(invoices: pd.DataFrame, export: bool) -> pd.DataFrame:
    t0 = time.time()
    print("  ── Rule 1: Benford's Law ─────────────────────────────────")
    vendor_scores = compute_vendor_benford_scores(invoices)
    flagged       = join_benford_flags_to_invoices(invoices, vendor_scores)

    n_flagged     = flagged["benford_flag"].sum()
    n_strict      = flagged["benford_flag_strict"].sum()
    vendors_tested= len(vendor_scores)
    print(f"     Vendors tested:      {vendors_tested:,}")
    print(f"     Vendors flagged (p<0.05): {vendor_scores['benford_flag'].sum():,}")
    print(f"     Invoices affected (p<0.05): {n_flagged:,}  ({n_flagged/len(flagged)*100:.1f}%)")
    print(f"     Invoices affected (p<0.01): {n_strict:,}")
    print(f"     Elapsed: {time.time()-t0:.1f}s\n")

    if export:
        vendor_scores.to_csv(OUT_DIR / "benford_vendor_scores.csv", index=False)
        print(f"     → benford_vendor_scores.csv ({len(vendor_scores):,} vendors)")

    return flagged[["invoice_id", "benford_flag", "benford_flag_strict",
                    "chi2_stat", "p_value", "mse_proportion"]]


def run_threshold(invoices: pd.DataFrame, export: bool) -> pd.DataFrame:
    t0 = time.time()
    print("  ── Rule 2: Threshold Gaming ──────────────────────────────")
    flagged = detect_threshold_gaming(invoices)
    summary = threshold_gaming_summary(flagged)

    n_flagged = flagged["threshold_gaming_flag"].sum()
    print(f"     Invoices in 2% zone:         {flagged['nearest_threshold_2pct'].notna().sum():,}")
    print(f"     Invoices flagged (≥3/qtr):   {n_flagged:,}  ({n_flagged/len(flagged)*100:.1f}%)")
    print(f"     Vendor×threshold combos hit: {len(summary):,}")
    print(f"     Elapsed: {time.time()-t0:.1f}s\n")

    if export and not summary.empty:
        summary.to_csv(OUT_DIR / "threshold_gaming_summary.csv", index=False)
        print(f"     → threshold_gaming_summary.csv")

    return flagged[["invoice_id", "threshold_gaming_flag",
                    "nearest_threshold_2pct", "gap_pct",
                    "vendor_qtr_threshold_count"]]


def run_duplicates(invoices: pd.DataFrame, export: bool) -> pd.DataFrame:
    t0 = time.time()
    print("  ── Rule 3: Fuzzy Duplicate Detection ────────────────────")
    flagged, pairs_df = detect_fuzzy_duplicates(invoices)
    stats             = duplicate_summary(pairs_df)

    n_flagged = flagged["duplicate_flag"].sum()
    print(f"     Duplicate pairs found:   {stats.get('total_pairs', 0):,}")
    print(f"       — fraud pairs:         {stats.get('fraud_pairs',  0):,}")
    print(f"       — clean pairs:         {stats.get('clean_pairs',  0):,}")
    print(f"     Invoices flagged:        {n_flagged:,}  ({n_flagged/len(flagged)*100:.1f}%)")
    print(f"     Avg amount diff:         {stats.get('avg_amount_diff_pct', 0):.4f}%")
    print(f"     Avg date diff:           {stats.get('avg_date_diff_days',  0):.1f} days")
    print(f"     Elapsed: {time.time()-t0:.1f}s\n")

    if export and not pairs_df.empty:
        pairs_df.to_csv(OUT_DIR / "duplicate_pairs.csv", index=False)
        print(f"     → duplicate_pairs.csv ({len(pairs_df):,} pairs)")

    return flagged[["invoice_id", "duplicate_flag",
                    "duplicate_pair_id", "duplicate_partner_invoice_id"]]


def run_shell(invoices: pd.DataFrame, vendors: pd.DataFrame, export: bool) -> pd.DataFrame:
    t0 = time.time()
    print("  ── Rule 4: Shell Vendor Detection ───────────────────────")
    flagged_inv, vendor_risk = detect_shell_vendors(invoices, vendors)
    cluster_summary          = shell_cluster_summary(vendor_risk)

    n_bank      = flagged_inv["shell_bank_flag"].sum()
    n_composite = flagged_inv["shell_composite_flag"].sum()
    n_clusters  = (vendor_risk["vendors_sharing_bank"].fillna(1) >= 2).sum()

    print(f"     Shell vendors (shared bank):   {n_clusters:,}")
    print(f"     Invoices flagged (bank only):  {n_bank:,}  ({n_bank/len(flagged_inv)*100:.1f}%)")
    print(f"     Invoices flagged (composite):  {n_composite:,}  ({n_composite/len(flagged_inv)*100:.1f}%)")
    print(f"     Clusters detected:             {len(cluster_summary):,}")
    print(f"     Elapsed: {time.time()-t0:.1f}s\n")

    if export and not cluster_summary.empty:
        cluster_summary.to_csv(OUT_DIR / "shell_clusters.csv", index=False)
        print(f"     → shell_clusters.csv ({len(cluster_summary):,} clusters)")

    return flagged_inv[["invoice_id", "shell_bank_flag", "shell_composite_flag",
                         "shell_risk_score", "vendors_sharing_bank"]]


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(merged: pd.DataFrame, export: bool) -> None:
    print("\n  ── Evaluation ────────────────────────────────────────────")

    rule_flags = {
        "benford_p05":        "benford_flag",
        "benford_p01":        "benford_flag_strict",
        "threshold_gaming":   "threshold_gaming_flag",
        "duplicate_fuzzy":    "duplicate_flag",
        "shell_bank":         "shell_bank_flag",
        "shell_composite":    "shell_composite_flag",
    }

    y_true    = merged["is_fraud"]
    all_metrics = []

    for rule_name, col in rule_flags.items():
        if col not in merged.columns:
            continue
        y_pred = merged[col].fillna(False)
        score_col = None
        if rule_name.startswith("benford"):
            score_col = merged["chi2_stat"].fillna(0)
        elif rule_name == "shell_bank":
            score_col = merged["shell_risk_score"].fillna(0)
        elif rule_name == "threshold_gaming":
            score_col = merged["vendor_qtr_threshold_count"].fillna(0)
        m = evaluate_binary_rule(y_true, y_pred, rule_name, score_col)
        all_metrics.append(m)

    print_rule_summary(all_metrics, "Per-Rule Precision / Recall / F1")
    metrics_df = pd.DataFrame(all_metrics)

    # ── Category breakdown ────────────────────────────────────────────────────
    print("\n  Per-Rule Performance by Fraud Category:")
    cat_rows = []
    for rule_name, col in rule_flags.items():
        if col not in merged.columns:
            continue
        cat_df = per_category_metrics(merged, col, rule_name, FRAUD_CATEGORIES + ["clean"])
        cat_rows.append(cat_df)
        print(f"\n    {rule_name}:")
        print(cat_df.to_string(index=False))
    cat_breakdown = pd.concat(cat_rows, ignore_index=True) if cat_rows else pd.DataFrame()

    # ── Threshold curves for scores ───────────────────────────────────────────
    print("\n  Threshold curves (precision/recall at varying cutoffs):")
    curve_dfs = []
    curve_configs = [
        ("benford_chi2",         "chi2_stat",                  "benford_p05"),
        ("shell_risk",           "shell_risk_score",           "shell_composite"),
        ("threshold_vote_count", "vendor_qtr_threshold_count", "threshold_gaming"),
    ]
    for label, score_col, rule_name in curve_configs:
        if score_col not in merged.columns:
            continue
        score = merged[score_col].fillna(0)
        thresholds = sorted(score.unique())
        if len(thresholds) > 40:
            idx = np.linspace(0, len(thresholds)-1, 40, dtype=int)
            thresholds = [thresholds[i] for i in idx]
        curve = evaluate_at_thresholds(y_true, score, rule_name, thresholds)
        curve_dfs.append(curve)
        best = curve.loc[curve["f1"].idxmax()]
        print(f"    {label:30s} best F1={best['f1']:.4f} at threshold={best['threshold']}")
    all_curves = pd.concat(curve_dfs, ignore_index=True) if curve_dfs else pd.DataFrame()

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_cols = [c for c in ["benford_flag", "threshold_gaming_flag",
                                  "duplicate_flag", "shell_bank_flag"]
                     if c in merged.columns]
    ens = evaluate_ensemble(merged, ensemble_cols)
    print(f"\n  Ensemble voting (rules: {', '.join(ensemble_cols)}):")
    print(ens.to_string(index=False))

    # ── Rule failure analysis ──────────────────────────────────────────────────
    print(f"""
  ── Where Rules Fail (motivation for ML layer) ────────────────────────────

  1. Benford's Law:
     • Low recall on threshold-gaming fraud — those amounts are crafted to be
       just below thresholds (e.g. $9,840), which still follows Benford's
       digit-1 distribution naturally.
     • Vendor-level aggregation means a single fraudulent invoice from an
       otherwise clean vendor will not be flagged.
     • Completely blind to shell vendors — their amounts are arbitrary.

  2. Threshold Gaming:
     • Requires ≥3 hits per quarter — a patient fraudster submitting 1–2 per
       quarter evades detection indefinitely.
     • Fixed threshold list ($5k, $10k, $25k, $50k, $100k) — custom approval
       limits not in this list are invisible to the rule.
     • Zero recall on duplicate and shell fraud categories.

  3. Fuzzy Duplicate Detection:
     • Designed for exact clones — misses semantic duplicates (same service,
       different description, >1% amount variation, >7 day gap).
     • False-positive pairs: legitimate recurring monthly invoices (e.g.
       utilities, SaaS subscriptions) have similar amounts on regular cadences.
     • Blocking on vendor_id means cross-vendor duplicate schemes (split
       billing across shell vendors) are invisible.

  4. Shell Vendor (bank account sharing):
     • Only catches vendors in our dataset. A shell vendor using a brand-new
       bank account has zero prior signals until the second vendor appears.
     • The rule is binary — it cannot rank shell vendors by severity.
     • Misses nominee arrangements where each shell uses a unique account.

  5. Ensemble:
     • Rules are largely orthogonal — they catch different fraud types,
       meaning a fraudster using a novel method evades all four simultaneously.
     • No temporal learning: a rule that fires today fires identically in
       6 months even as patterns evolve.
     • Cannot detect low-signal anomalies that are individually below any
       threshold but collectively unusual (isolation forest territory).

  → ML layer requirements derived from rule failures:
     • Needs to work at invoice level (not vendor-aggregate like Benford).
     • Must learn flexible amount-proximity patterns, not hardcoded thresholds.
     • Needs temporal features to catch low-frequency persistent fraud.
     • Should model inter-vendor relationships (graph features) for split billing.
     • Must rank risk continuously, not just binary flag.
""")

    # ── Export ────────────────────────────────────────────────────────────────
    if export:
        metrics_df.to_csv(OUT_DIR / "rule_metrics.csv", index=False)
        ens.to_csv(OUT_DIR / "ensemble_metrics.csv", index=False)
        if not cat_breakdown.empty:
            cat_breakdown.to_csv(OUT_DIR / "category_breakdown.csv", index=False)
        if not all_curves.empty:
            all_curves.to_csv(OUT_DIR / "threshold_curves.csv", index=False)
        print(f"\n  Exported to {OUT_DIR}/")
        print(f"    rule_metrics.csv, ensemble_metrics.csv,")
        print(f"    category_breakdown.csv, threshold_curves.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rule-based detection layer")
    parser.add_argument("--rules", nargs="*",
                        choices=["benford", "threshold", "duplicate", "shell", "all"],
                        default=["all"])
    parser.add_argument("--no-export", action="store_true")
    args    = parser.parse_args()
    export  = not args.no_export
    rules   = args.rules if args.rules != ["all"] else ["benford","threshold","duplicate","shell"]

    print(f"\n{'═'*65}")
    print(f"  RULE-BASED DETECTION LAYER")
    print(f"  Rules: {', '.join(rules)}")
    print(f"{'═'*65}\n")

    invoices, vendors = load_data()

    # Collect per-rule flag frames indexed by invoice_id
    flag_frames = [invoices[["invoice_id", "is_fraud", "fraud_category",
                               "anomaly_type", "vendor_id", "amount",
                               "issue_date"]].copy()]

    if "benford"   in rules:
        flag_frames.append(run_benford(invoices, export))
    if "threshold" in rules:
        flag_frames.append(run_threshold(invoices, export))
    if "duplicate" in rules:
        flag_frames.append(run_duplicates(invoices, export))
    if "shell"     in rules:
        flag_frames.append(run_shell(invoices, vendors, export))

    # Merge all flags onto invoice spine
    merged = flag_frames[0]
    for f in flag_frames[1:]:
        merged = merged.merge(f, on="invoice_id", how="left")

    # Save full flag table
    if export:
        merged.to_csv(OUT_DIR / "rule_scores.csv", index=False)
        print(f"  → rule_scores.csv ({len(merged):,} rows)")

    # Evaluate
    run_evaluation(merged, export)

    print(f"\n{'═'*65}")
    print(f"  Done. All outputs in {OUT_DIR}/")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()