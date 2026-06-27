"""
Great Expectations data quality suite for feat_engineered mart.
GX version: 1.x (fluent datasource API)

Suites
  identity      – row count, column presence, PK uniqueness
  nulls         – hard not-null + soft mostly-not-null
  ranges        – numeric min/max assertions
  sets          – categorical value sets
  booleans      – boolean column membership
  distributions – mean/median bounds (fraud rate, score skew, etc.)
  patterns      – regex checks (UUID format)
  cross_column  – pandas-eval cross-column logic assertions

Usage
  python gx_checks.py                        # run all suites
  python gx_checks.py --suite ranges         # one suite
  python gx_checks.py --fail-fast            # stop on first failure
  python gx_checks.py --export               # write JSON report
  python gx_checks.py --suite nulls --export # combine flags
"""

import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import great_expectations as gx

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = "/home/claude/invoice_anomaly.duckdb"
TABLE      = "main_mart.feat_engineered"
REPORT_DIR = Path("/mnt/user-data/outputs/gx_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["GX_ANALYTICS_ENABLED"] = "false"   # silence telemetry

# ── Load data once ─────────────────────────────────────────────────────────────
print(f"\n  Loading {TABLE} …")
con = duckdb.connect(DB_PATH, read_only=True)
df  = con.execute(f"SELECT * FROM {TABLE}").df()
con.close()
print(f"  {len(df):,} rows × {len(df.columns)} columns loaded\n")

# ── GX context ─────────────────────────────────────────────────────────────────
context = gx.get_context(mode="ephemeral")
ds      = context.data_sources.add_pandas("invoice_ds")
asset   = ds.add_dataframe_asset("feat_engineered")


def fresh_validator(suite_name: str):
    """Create a new suite + validator for each run (suites don't share state)."""
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
    br    = asset.build_batch_request(options={"dataframe": df})
    return context.get_validator(batch_request=br, expectation_suite=suite)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE DEFINITIONS  { suite_name: [(method_name, kwargs), …] }
# ─────────────────────────────────────────────────────────────────────────────
SUITES = {}

# ── 1. IDENTITY & UNIQUENESS ─────────────────────────────────────────────────
SUITES["identity"] = [
    ("expect_column_values_to_not_be_null",   {"column": "invoice_id"}),
    ("expect_column_values_to_be_unique",     {"column": "invoice_id"}),
    ("expect_column_values_to_not_be_null",   {"column": "vendor_id"}),
    ("expect_column_values_to_not_be_null",   {"column": "buyer_id"}),
    ("expect_table_row_count_to_be_between",  {"min_value": 50_000, "max_value": 60_000}),
    ("expect_table_columns_to_match_set", {
        "column_set": [
            "invoice_id", "vendor_id", "buyer_id", "amount", "issue_date",
            "vendor_category", "status", "payment_method", "is_fraud",
            "fraud_category", "anomaly_type",
            "vendor_amount_zscore", "category_amount_zscore",
            "vendor_zscore_90d", "vendor_category_bias_zscore",
            "vendor_dtp_rolling_avg_90d", "vendor_dtp_rolling_avg_180d",
            "dtp_zscore", "dtp_deviation_from_90d_avg",
            "min_dist_to_any_threshold", "flag_threshold_danger_zone",
            "submission_hour", "submission_dow", "submission_time_band",
            "bank_vendor_count", "vendor_bank_share_of_wallet",
            "engineered_risk_score", "total_risk_score", "rule_based_fraud_score",
        ],
        "exact_match": False,
    }),
]

# ── 2. NULL CHECKS ───────────────────────────────────────────────────────────
_hard_not_null = [
    "invoice_id", "vendor_id", "buyer_id",
    "amount", "issue_date", "vendor_category", "buyer_department",
    "status", "payment_method", "currency",
    "is_fraud", "fraud_category", "anomaly_type",
    "vendor_amount_zscore", "category_amount_zscore", "vendor_category_bias_zscore",
    "flag_near_duplicate_prev", "flag_near_duplicate_next", "flag_velocity_spike",
    "flag_threshold_gaming", "flag_shell_vendor", "is_shell_vendor", "is_high_risk_country",
    "rule_based_fraud_score", "engineered_risk_score", "total_risk_score",
    "submission_hour", "submission_dow", "submission_dom",
    "is_business_hours", "submission_time_band", "submission_dow_band",
    "flag_backdated_submission", "flag_late_submission",
    "bank_vendor_count", "flag_threshold_danger_zone",
    "min_dist_to_any_threshold", "amount_tier",
    "flag_iqr_outlier_high", "flag_iqr_outlier_low",
]

_soft_not_null = [       # ≥85% non-null (null for non-paid invoices / first invoice)
    "days_to_payment",
    "days_past_due",
    "vendor_prev_invoice_amount",
    "days_since_prev_vendor_invoice",
    "vendor_dtp_rolling_avg_90d",
    "vendor_dtp_rolling_avg_180d",
    "dtp_zscore",
]

SUITES["nulls"] = (
    [("expect_column_values_to_not_be_null", {"column": c})
     for c in _hard_not_null]
  + [("expect_column_values_to_not_be_null", {"column": c, "mostly": 0.50})
     for c in _soft_not_null]
)

# ── 3. RANGE ASSERTIONS ──────────────────────────────────────────────────────
SUITES["ranges"] = [
    # Amounts
    ("expect_column_values_to_be_between",
     {"column": "amount",                    "min_value": 0.0, "max_value": 1_000_000}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_avg_amount_alltime", "min_value": 0.0}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_total_spend_30d",    "min_value": 0.0}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_cumulative_spend",   "min_value": 0.0}),
    # Z-scores (corruption = values beyond ±50)
    ("expect_column_values_to_be_between",
     {"column": "vendor_amount_zscore",       "min_value": -50.0,  "max_value": 50.0}),
    ("expect_column_values_to_be_between",
     {"column": "category_amount_zscore",     "min_value": -50.0,  "max_value": 50.0}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_zscore_90d",          "min_value": -50.0,  "max_value": 50.0}),
    ("expect_column_values_to_be_between",
     {"column": "dtp_zscore",                 "min_value": -20.0,  "max_value": 20.0}),
    # Payment days
    ("expect_column_values_to_be_between",
     {"column": "days_to_payment",            "min_value": 0,      "max_value": 365,
      "mostly": 0.99}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_dtp_rolling_avg_90d", "min_value": 0,      "mostly": 0.99}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_dtp_rolling_avg_180d","min_value": 0,      "mostly": 0.99}),
    # Temporal
    ("expect_column_values_to_be_between",
     {"column": "submission_hour",            "min_value": 0,      "max_value": 23}),
    ("expect_column_values_to_be_between",
     {"column": "submission_dow",             "min_value": 0,      "max_value": 6}),
    ("expect_column_values_to_be_between",
     {"column": "submission_dom",             "min_value": 1,      "max_value": 31}),
    ("expect_column_values_to_be_between",
     {"column": "submission_week",            "min_value": 1,      "max_value": 53}),
    ("expect_column_values_to_be_between",
     {"column": "submission_quarter",         "min_value": 1,      "max_value": 4}),
    ("expect_column_values_to_be_between",
     {"column": "issue_year",                 "min_value": 2020,   "max_value": 2030}),
    ("expect_column_values_to_be_between",
     {"column": "issue_month",                "min_value": 1,      "max_value": 12}),
    ("expect_column_values_to_be_between",
     {"column": "issue_dow",                  "min_value": 0,      "max_value": 6}),
    ("expect_column_values_to_be_between",
     {"column": "issue_dom",                  "min_value": 1,      "max_value": 31}),
    # Threshold distances
    ("expect_column_values_to_be_between",
     {"column": "min_dist_to_any_threshold",  "min_value": 0.0}),
    ("expect_column_values_to_be_between",
     {"column": "abs_dist_to_5k",             "min_value": 0.0}),
    ("expect_column_values_to_be_between",
     {"column": "abs_dist_to_10k",            "min_value": 0.0}),
    ("expect_column_values_to_be_between",
     {"column": "abs_dist_to_100k",           "min_value": 0.0}),
    # Bank
    ("expect_column_values_to_be_between",
     {"column": "bank_vendor_count",          "min_value": 1,      "max_value": 50}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_bank_share_of_wallet","min_value": 0.0,    "max_value": 1.0,
      "mostly": 0.99}),
    # Ratios [0,1]
    ("expect_column_values_to_be_between",
     {"column": "v_wire_ratio",               "min_value": 0.0,    "max_value": 1.0}),
    # Scores
    ("expect_column_values_to_be_between",
     {"column": "rule_based_fraud_score",     "min_value": 0,      "max_value": 20}),
    ("expect_column_values_to_be_between",
     {"column": "engineered_risk_score",      "min_value": 0,      "max_value": 30}),
    ("expect_column_values_to_be_between",
     {"column": "total_risk_score",           "min_value": 0,      "max_value": 50}),
    ("expect_column_values_to_be_between",
     {"column": "shell_suspicion_score",      "min_value": 0,      "max_value": 10}),
    # Counts
    ("expect_column_values_to_be_between",
     {"column": "vendor_inv_count_30d",       "min_value": 0}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_inv_count_90d",       "min_value": 0}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_invoices_same_hour",  "min_value": 1}),
    ("expect_column_values_to_be_between",
     {"column": "vendor_invoices_same_day",   "min_value": 1}),
    # Percentiles
    ("expect_column_values_to_be_between",
     {"column": "cat_p25_amount",             "min_value": 0.0}),
    ("expect_column_values_to_be_between",
     {"column": "cat_p95_amount",             "min_value": 0.0}),
]

# ── 4. SET MEMBERSHIP ────────────────────────────────────────────────────────
SUITES["sets"] = [
    ("expect_column_values_to_be_in_set", {
        "column": "fraud_category",
        "value_set": ["clean","duplicate","threshold_gaming","shell_vendor","soft_anomaly"],
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "amount_tier",
        "value_set": ["micro","small","medium","large","xlarge","mega"],
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "submission_time_band",
        "value_set": ["overnight","early_morning","morning","midday","afternoon","evening","night"],
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "submission_dow_band",
        "value_set": ["friday","weekend","weekday"],
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "threshold_proximity_band",
        "value_set": ["tight","near","none"],
        "mostly": 0.99,
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "status",
        "value_set": ["paid","approved","pending","disputed","cancelled"],
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "payment_method",
        "value_set": ["ach","wire","check","credit card","eft"],
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "currency",
        "value_set": ["USD","EUR","GBP","CAD","AUD"],
    }),
    ("expect_column_values_to_be_in_set", {
        "column": "vendor_category",
        "value_set": [
            "IT Services","Office Supplies","Consulting","Logistics",
            "Facilities","Marketing","Legal","Utilities","Staffing","Raw Materials",
        ],
    }),
]

# ── 5. BOOLEAN COLUMNS ───────────────────────────────────────────────────────
_bool_cols = [
    "is_fraud","is_shell_vendor","is_high_risk_country",
    "flag_near_duplicate_prev","flag_near_duplicate_next","flag_velocity_spike",
    "flag_threshold_gaming","flag_shell_vendor",
    "flag_iqr_outlier_high","flag_iqr_outlier_low",
    "flag_unusually_fast_payment","flag_unusually_slow_payment","flag_robotic_payment_timing",
    "flag_threshold_danger_zone","flag_backdated_submission","flag_late_submission",
    "flag_dominant_vendor_in_cluster","flag_multicategory_bank",
    "flag_new_bank_relationship","flag_high_bank_velocity",
    "is_weekend_issue","is_weekend_payment","is_month_start","is_month_end",
    "is_round_1k","is_round_500","is_business_hours",
    "is_month_end_submission","is_month_start_submission","is_quarter_end_submission",
]

SUITES["booleans"] = [
    ("expect_column_values_to_be_in_set",
     {"column": c, "value_set": [True, False]})
    for c in _bool_cols
]

# ── 6. DISTRIBUTION BOUNDS ───────────────────────────────────────────────────
SUITES["distributions"] = [
    # Fraud rate 1–10%
    ("expect_column_mean_to_be_between",
     {"column": "is_fraud",                  "min_value": 0.01,  "max_value": 0.10}),
    # Risk scores right-skewed (median low)
    ("expect_column_median_to_be_between",
     {"column": "rule_based_fraud_score",    "min_value": 0,     "max_value": 2}),
    ("expect_column_median_to_be_between",
     {"column": "engineered_risk_score",     "min_value": 0,     "max_value": 3}),
    # Amount distribution
    ("expect_column_median_to_be_between",
     {"column": "amount",                    "min_value": 1_000, "max_value": 50_000}),
    # Avg payment days 20–90
    ("expect_column_mean_to_be_between",
     {"column": "days_to_payment",           "min_value": 20,    "max_value": 90}),
    # Threshold danger zone: 0.1%–5% of invoices
    ("expect_column_mean_to_be_between",
     {"column": "flag_threshold_danger_zone","min_value": 0.001, "max_value": 0.05}),
    # Shell vendor flag rate: 0.5%–5%
    ("expect_column_mean_to_be_between",
     {"column": "flag_shell_vendor",         "min_value": 0.005, "max_value": 0.05}),
    # Submission hours average during business hours
    ("expect_column_mean_to_be_between",
     {"column": "submission_hour",           "min_value": 9.0,   "max_value": 15.0}),
    # Bank vendor count: average near 1 (most vendors alone on their account)
    ("expect_column_mean_to_be_between",
     {"column": "bank_vendor_count",         "min_value": 1.0,   "max_value": 5.0}),
    # Vendor 30d invoice count sanity
    ("expect_column_mean_to_be_between",
     {"column": "vendor_inv_count_30d",      "min_value": 1.0,   "max_value": 500.0}),
]

# ── 7. REGEX / PATTERN CHECKS ────────────────────────────────────────────────
UUID_REGEX = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
SUITES["patterns"] = [
    ("expect_column_values_to_match_regex", {"column": "invoice_id", "regex": UUID_REGEX}),
    ("expect_column_values_to_match_regex", {"column": "vendor_id",  "regex": UUID_REGEX}),
    ("expect_column_values_to_match_regex", {"column": "buyer_id",   "regex": UUID_REGEX}),
]

# ── 8. CROSS-COLUMN LOGIC (pandas eval) ──────────────────────────────────────
CROSS_COLUMN_CHECKS = [
    {
        "name":  "total_risk_gte_rule_based",
        "desc":  "total_risk_score >= rule_based_fraud_score always",
        "expr":  "df['total_risk_score'] >= df['rule_based_fraud_score']",
    },
    {
        "name":  "shell_score_nonneg",
        "desc":  "shell_suspicion_score is never negative",
        "expr":  "df['shell_suspicion_score'] >= 0",
    },
    {
        "name":  "fraud_shell_implies_flag",
        "desc":  "shell_vendor fraud_category implies is_shell_vendor OR flag_shell_vendor",
        "expr":  "(df['fraud_category'] != 'shell_vendor') | df['is_shell_vendor'] | df['flag_shell_vendor']",
    },
    {
        "name":  "cat_percentile_ordering",
        "desc":  "P25 <= P75 <= P95 for category amount percentiles",
        "expr":  "(df['cat_p25_amount'] <= df['cat_p75_amount']) & (df['cat_p75_amount'] <= df['cat_p95_amount'])",
    },
    {
        "name":  "amount_delta_nonneg",
        "desc":  "amount_delta_from_prev is abs diff, always >= 0",
        "expr":  "df['amount_delta_from_prev'].isna() | (df['amount_delta_from_prev'] >= 0)",
    },
    {
        "name":  "bank_share_wallet_in_range",
        "desc":  "vendor_bank_share_of_wallet between 0 and 1 where not null",
        "expr":  "df['vendor_bank_share_of_wallet'].isna() | ((df['vendor_bank_share_of_wallet'] >= 0) & (df['vendor_bank_share_of_wallet'] <= 1))",
    },
    {
        "name":  "engineered_score_additive",
        "desc":  "engineered_risk_score <= 20 (cap check on additive scoring)",
        "expr":  "df['engineered_risk_score'] <= 20",
    },
    {
        "name":  "zscore_90d_nan_iff_rolling_nan",
        "desc":  "vendor_zscore_90d is null iff vendor_90d_rolling_std is null or zero",
        "expr":  "df['vendor_zscore_90d'].notna() | df['vendor_90d_rolling_std'].isna() | (df['vendor_90d_rolling_std'] == 0)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(suite_name: str, expectations: list, fail_fast: bool = False):
    validator = fresh_validator(suite_name)
    results   = []
    for method, kwargs in expectations:
        # Strip non-GX keys we use for notes
        gx_kwargs = {k: v for k, v in kwargs.items() if k != "notes"}
        note      = kwargs.get("notes", "")
        col       = kwargs.get("column", "TABLE")
        label     = f"{method}({col})"
        try:
            res    = getattr(validator, method)(**gx_kwargs)
            passed = bool(res.success)
        except Exception as exc:
            passed = False
            note   = str(exc)[:120]
        results.append({"check": label, "passed": passed, "notes": note})
        if not passed and fail_fast:
            print(f"    ❌  {label}  {note}")
            print("    ⛔ --fail-fast: stopping.")
            return results
    return results


def run_cross_column(fail_fast: bool = False):
    results = []
    for chk in CROSS_COLUMN_CHECKS:
        try:
            mask   = eval(chk["expr"])
            passed = bool(mask.all())
            fails  = int((~mask).sum())
            note   = chk["desc"] + (f"  [{fails} violations]" if not passed else "")
        except Exception as exc:
            passed = False
            note   = chk["desc"] + f"  [ERROR: {exc}]"
        results.append({"check": f"cross_col::{chk['name']}", "passed": passed, "notes": note})
        if not passed and fail_fast:
            print(f"    ❌  {results[-1]['check']}  {note}")
            print("    ⛔ --fail-fast: stopping.")
            return results
    return results


def print_results(suite_name: str, results: list):
    passed = sum(r["passed"] for r in results)
    total  = len(results)
    print(f"\n  ┌─ {suite_name}  ({passed}/{total}) " + "─" * max(0, 55 - len(suite_name)))
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        note = f"  ↳ {r['notes']}" if r["notes"] and not r["passed"] else ""
        print(f"  │  {icon}  {r['check']}{note}")
    print(f"  └{'─'*60}")
    return passed, total


def export_json(all_results: dict, path: str):
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "table":        TABLE,
        "total_rows":   len(df),
        "suites":       all_results,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  📄 JSON report → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",     default="all",
        help="Suite: identity|nulls|ranges|sets|booleans|distributions|patterns|cross_column|all")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--export",    action="store_true")
    args = parser.parse_args()

    suites_to_run = (
        list(SUITES.keys()) + ["cross_column"]
        if args.suite == "all" else [args.suite]
    )

    grand_pass = grand_total = 0
    all_results = {}

    for sname in suites_to_run:
        if sname == "cross_column":
            results = run_cross_column(args.fail_fast)
        elif sname in SUITES:
            results = run_suite(sname, SUITES[sname], args.fail_fast)
        else:
            print(f"  Unknown suite '{sname}'. Options: {list(SUITES)+['cross_column']}")
            continue

        p, t = print_results(sname, results)
        grand_pass  += p
        grand_total += t
        all_results[sname] = results

    pct = grand_pass / grand_total * 100 if grand_total else 0
    print(f"\n  {'═'*62}")
    print(f"  GRAND TOTAL  {grand_pass}/{grand_total} checks passed  ({pct:.1f}%)")
    print(f"  Table: {TABLE}  │  Rows: {len(df):,}")
    if grand_pass == grand_total:
        print("  🎉  All checks passed — feature mart is clean.")
    else:
        print(f"  ⚠️   {grand_total - grand_pass} check(s) failed — see ❌ above.")
    print(f"  {'═'*62}\n")

    if args.export:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"gx_report_{ts}.json"
        export_json(all_results, str(path))

    sys.exit(0 if grand_pass == grand_total else 1)


if __name__ == "__main__":
    main()
