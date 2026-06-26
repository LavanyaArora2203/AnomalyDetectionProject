"""
Fraud injection for invoice anomaly detection dataset.
Injects three fraud patterns with full audit trail:
  1. Near-duplicate invoices  (same vendor, ~same amount ±0.01, close dates)
  2. Threshold-gaming invoices (amounts just below approval thresholds)
  3. Shell vendor clusters    (fake vendors sharing bank accounts / tax IDs)
"""

import uuid, random, math
import numpy as np
import pandas as pd
from datetime import timedelta
from copy import deepcopy

random.seed(99)
np.random.seed(99)

OUT = "/mnt/user-data/outputs"

# ── Load existing data ────────────────────────────────────────────────────────
print("Loading data...")
inv = pd.read_csv(f"{OUT}/invoices.csv", parse_dates=["issue_date","due_date","payment_date","created_at"])
ven = pd.read_csv(f"{OUT}/vendors.csv")
buy = pd.read_csv(f"{OUT}/buyers.csv")
li  = pd.read_csv(f"{OUT}/line_items.csv")

# Normalise anomaly_type — treat NaN as empty string
inv["anomaly_type"] = inv["anomaly_type"].fillna("")

# We'll collect new rows here
new_invoices  = []
new_vendors   = []
new_line_items= []

# ─────────────────────────────────────────────────────────────────────────────
# FRAUD TYPE 1 — Near-duplicate invoices
# Strategy: pick 300 paid invoices, clone each with:
#   • amount ±0.01 (different enough to dodge exact-match dedup)
#   • invoice number with a suffix tweak
#   • issue_date shifted 1-7 days
#   • fresh UUID, same vendor/buyer
#   • Both original and clone labelled fraud_duplicate
# ─────────────────────────────────────────────────────────────────────────────
print("Injecting near-duplicate invoices...")

N_DUPE_PAIRS = 300
paid_inv = inv[inv["status"] == "paid"].sample(N_DUPE_PAIRS, random_state=7).copy()

dupe_clone_ids = []
original_ids   = paid_inv["id"].tolist()

for _, orig in paid_inv.iterrows():
    delta      = random.choice([-0.01, 0.01])
    new_amount = round(float(orig["amount"]) + delta, 2)
    date_shift = timedelta(days=random.randint(1, 7))
    new_issue  = pd.Timestamp(orig["issue_date"]) + date_shift
    new_due    = pd.Timestamp(orig["due_date"])   + date_shift
    pay_shift  = timedelta(days=random.randint(0, 3))
    new_pay    = (pd.Timestamp(orig["payment_date"]) + pay_shift
                  if pd.notna(orig["payment_date"]) else pd.NaT)
    # Slight invoice number mutation: append letter suffix
    suffix     = random.choice(["A","B","R","C"])
    new_inv_no = str(orig["invoice_number"]) + suffix

    clone_id = str(uuid.uuid4())
    dupe_clone_ids.append(clone_id)

    new_invoices.append({
        **orig.to_dict(),
        "id":             clone_id,
        "invoice_number": new_inv_no,
        "amount":         new_amount,
        "issue_date":     new_issue.date(),
        "due_date":       new_due.date(),
        "payment_date":   new_pay.date() if pd.notna(new_pay) else None,
        "anomaly_type":   "fraud_duplicate",
        "created_at":     new_issue,
    })

    # Clone line items (split proportionally to new amount)
    orig_li = li[li["invoice_id"] == orig["id"]].copy()
    if not orig_li.empty:
        ratio = new_amount / float(orig["amount"]) if float(orig["amount"]) != 0 else 1
        for _, row in orig_li.iterrows():
            new_total = round(float(row["total"]) * ratio, 2)
            new_unit  = round(new_total / max(float(row["quantity"]), 1), 4)
            new_line_items.append({
                **row.to_dict(),
                "id":         str(uuid.uuid4()),
                "invoice_id": clone_id,
                "total":      new_total,
                "unit_price": new_unit,
            })

# Tag originals too so both sides of the pair are labelled
inv.loc[inv["id"].isin(original_ids), "anomaly_type"] = "fraud_duplicate"

# ─────────────────────────────────────────────────────────────────────────────
# FRAUD TYPE 2 — Threshold-gaming (approval limit evasion)
# Common approval thresholds:  $5k, $10k, $25k, $50k, $100k
# Strategy: create 200 invoices whose amounts are $1–$200 below a threshold,
#   attributed to a mix of existing vendors with high invoice volume.
#   Each invoice gets status "approved" (sailed through auto-approval).
# ─────────────────────────────────────────────────────────────────────────────
print("Injecting threshold-gaming invoices...")

THRESHOLDS = [5_000, 10_000, 25_000, 50_000, 100_000]
N_THRESHOLD = 200

# Pick vendors that appear frequently (more realistic — active fraudsters reuse vendors)
vendor_counts = inv["vendor_id"].value_counts()
top_vendors   = vendor_counts[vendor_counts >= 10].index.tolist()
buyer_ids     = buy["id"].tolist()

# Date range same as dataset
date_range_days = (pd.Timestamp("2024-12-30") - pd.Timestamp("2022-01-01")).days

for i in range(N_THRESHOLD):
    threshold   = random.choice(THRESHOLDS)
    gap         = random.uniform(1, 200)          # $1–$200 below threshold
    amount      = round(threshold - gap, 2)
    vendor_id   = random.choice(top_vendors)
    buyer_id    = random.choice(buyer_ids)
    issue_date  = pd.Timestamp("2022-01-01") + timedelta(days=random.randint(0, date_range_days))
    due_date    = issue_date + timedelta(days=random.choice([30, 45, 60]))
    payment_date= issue_date + timedelta(days=random.randint(20, 55))

    inv_id = str(uuid.uuid4())
    new_invoices.append({
        "id":             inv_id,
        "vendor_id":      vendor_id,
        "buyer_id":       buyer_id,
        "invoice_number": f"INV-{random.randint(100000,999999)}",
        "amount":         amount,
        "currency":       "USD",
        "issue_date":     issue_date.date(),
        "due_date":       due_date.date(),
        "payment_date":   payment_date.date(),
        "status":         "approved",
        "payment_method": random.choice(["ACH","Wire","EFT"]),
        "anomaly_type":   f"fraud_threshold_{threshold}",
        "created_at":     issue_date,
    })

    # Single line item matching the full amount
    new_line_items.append({
        "id":            str(uuid.uuid4()),
        "invoice_id":    inv_id,
        "description":   random.choice(["Consulting services","Advisory fee","Professional services",
                                        "Project deliverable","Service contract"]),
        "quantity":      1,
        "unit_price":    amount,
        "total":         amount,
        "category_code": "CONSULTING",
    })

# ─────────────────────────────────────────────────────────────────────────────
# FRAUD TYPE 3 — Shell vendor clusters
# 15 clusters, each with 3-6 shell vendors that share:
#   • bank_account (added as a new column to vendors)
#   • tax_id pattern (same root EIN, different suffix)
#   • Similar company name patterns (e.g. "Alpha Services LLC" / "Alpha Solutions Inc")
# Each shell vendor submits 8-20 invoices totalling large amounts.
# ─────────────────────────────────────────────────────────────────────────────
print("Injecting shell vendor clusters...")

N_CLUSTERS   = 15
NAME_ROOTS   = [
    "Apex","Nexus","Vantage","Summit","Crest","Pinnacle","Meridian",
    "Zenith","Atlas","Orion","Titan","Vertex","Horizon","Arcadia","Solaris"
]
NAME_SUFFIXES= ["Services","Solutions","Consulting","Enterprises","Group",
                "Partners","Ventures","Associates","Holdings","Systems"]
LEGAL_FORMS  = ["LLC","Inc","Corp","Ltd","LP"]
COUNTRIES    = ["US","US","US","SG","BVI"]  # BVI = red flag jurisdiction

# Add bank_account col to vendors if not present
if "bank_account" not in ven.columns:
    ven["bank_account"] = None
if "shell_cluster_id" not in ven.columns:
    ven["shell_cluster_id"] = None

shell_vendor_ids_all = []

for cluster_idx in range(N_CLUSTERS):
    root        = NAME_ROOTS[cluster_idx]
    cluster_id  = f"SHELL_CLUSTER_{cluster_idx+1:02d}"
    # Shared bank account (same routing+account → dead giveaway when joined)
    shared_bank = f"ROUT-{random.randint(100000000,999999999)}-ACCT-{random.randint(10000000,99999999)}"
    # Base EIN, variants differ only in last 2 digits
    base_ein    = f"{random.randint(10,99)}-{random.randint(1000000,9000000)}"

    n_shells = random.randint(3, 6)
    cluster_vendor_ids = []

    for j in range(n_shells):
        suffix_ein = random.randint(0, 99)
        tax_id     = base_ein[:-2] + f"{suffix_ein:02d}"  # same root, rotated suffix
        name       = f"{root} {random.choice(NAME_SUFFIXES)} {random.choice(LEGAL_FORMS)}"
        vid        = str(uuid.uuid4())
        cluster_vendor_ids.append(vid)
        shell_vendor_ids_all.append(vid)

        new_vendors.append({
            "id":               vid,
            "name":             name,
            "tax_id":           tax_id,
            "country":          random.choice(COUNTRIES),
            "category":         "Consulting",
            "is_active":        True,
            "created_at":       pd.Timestamp("2022-01-01") + timedelta(days=random.randint(0,180)),
            "bank_account":     shared_bank,
            "shell_cluster_id": cluster_id,
        })

    # Each shell vendor submits invoices
    for vid in cluster_vendor_ids:
        n_inv = random.randint(8, 20)
        for _ in range(n_inv):
            amount     = round(random.uniform(8_000, 95_000), 2)  # stay under 100k threshold
            buyer_id   = random.choice(buyer_ids)
            issue_date = pd.Timestamp("2022-01-01") + timedelta(days=random.randint(0, date_range_days))
            due_date   = issue_date + timedelta(days=45)
            pay_date   = issue_date + timedelta(days=random.randint(30, 60))
            inv_id     = str(uuid.uuid4())

            new_invoices.append({
                "id":             inv_id,
                "vendor_id":      vid,
                "buyer_id":       buyer_id,
                "invoice_number": f"INV-{random.randint(100000,999999)}",
                "amount":         amount,
                "currency":       "USD",
                "issue_date":     issue_date.date(),
                "due_date":       due_date.date(),
                "payment_date":   pay_date.date(),
                "status":         "paid",
                "payment_method": "Wire",
                "anomaly_type":   f"fraud_shell_{cluster_id}",
                "created_at":     issue_date,
            })

            new_line_items.append({
                "id":            str(uuid.uuid4()),
                "invoice_id":    inv_id,
                "description":   random.choice(["Strategic consulting","Advisory retainer",
                                                "Management consulting","Business development"]),
                "quantity":      1,
                "unit_price":    amount,
                "total":         amount,
                "category_code": "CONSULTING",
            })

# ─────────────────────────────────────────────────────────────────────────────
# Merge everything back
# ─────────────────────────────────────────────────────────────────────────────
print("Merging and saving...")

# Ensure vendors table has the new columns for all rows
if "bank_account" not in ven.columns:
    ven["bank_account"] = None
if "shell_cluster_id" not in ven.columns:
    ven["shell_cluster_id"] = None

new_inv_df = pd.DataFrame(new_invoices)
new_ven_df = pd.DataFrame(new_vendors)
new_li_df  = pd.DataFrame(new_line_items)

inv_final = pd.concat([inv, new_inv_df], ignore_index=True)
ven_final = pd.concat([ven, new_ven_df], ignore_index=True)
li_final  = pd.concat([li,  new_li_df],  ignore_index=True)

inv_final.to_csv(f"{OUT}/invoices.csv",    index=False)
ven_final.to_csv(f"{OUT}/vendors.csv",     index=False)
li_final.to_csv(f"{OUT}/line_items.csv",   index=False)
# buyers unchanged

# ─────────────────────────────────────────────────────────────────────────────
# Fraud manifest — one row per fraud invoice with full metadata
# ─────────────────────────────────────────────────────────────────────────────
fraud_mask   = inv_final["anomaly_type"].str.startswith("fraud_", na=False)
fraud_df     = inv_final[fraud_mask][["id","vendor_id","buyer_id","invoice_number",
                                       "amount","issue_date","status","anomaly_type"]].copy()
fraud_df.to_csv(f"{OUT}/fraud_manifest.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fraud Injection Summary ──────────────────────────────────────")
print(f"  Total invoices now:  {len(inv_final):>8,}")
print(f"  Total vendors now:   {len(ven_final):>8,}")
print(f"  Total line items:    {len(li_final):>8,}")

print(f"\n  Fraud breakdown (anomaly_type):")
fraud_counts = inv_final[fraud_mask]["anomaly_type"].value_counts()
# Consolidate shell cluster labels for display
shell_total = fraud_counts[fraud_counts.index.str.startswith("fraud_shell")].sum()
dup_total   = (fraud_counts.get("fraud_duplicate", 0))
thresh_cols = fraud_counts[fraud_counts.index.str.startswith("fraud_threshold")]

print(f"    fraud_duplicate           {dup_total:>5,}  (300 originals + 300 clones)")
for t, c in thresh_cols.items():
    label = t.replace("fraud_threshold_","  below $")
    print(f"    threshold gaming {label:<10} {c:>5,}")
print(f"    fraud_shell (all clusters) {shell_total:>5,}  across {N_CLUSTERS} clusters")
print(f"\n  Shell vendor manifest:")
shell_vendors = ven_final[ven_final["shell_cluster_id"].notna()]
for cid, grp in shell_vendors.groupby("shell_cluster_id"):
    shared = grp["bank_account"].iloc[0]
    print(f"    {cid}: {len(grp)} vendors, shared bank={shared[:30]}…")

total_fraud = fraud_counts.sum()
total_inv   = len(inv_final)
print(f"\n  Overall fraud rate: {total_fraud}/{total_inv} = {total_fraud/total_inv*100:.2f}%")
print(f"\n  Files updated: invoices.csv, vendors.csv, line_items.csv")
print(f"  New file:      fraud_manifest.csv")
