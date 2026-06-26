##issue in parent directory
import uuid
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta, date

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_VENDORS   = 500
N_BUYERS    = 80
N_INVOICES  = 50_000
START_DATE  = date(2022, 1, 1)
END_DATE    = date(2024, 12, 31)
TOTAL_DAYS  = (END_DATE - START_DATE).days

# ── Vendor categories with typical amount distributions ──────────────────────
VENDOR_CATEGORIES = {
    "IT Services":        dict(mu=15_000, sigma=12_000, weight=0.18),
    "Office Supplies":    dict(mu=800,    sigma=500,    weight=0.12),
    "Consulting":         dict(mu=45_000, sigma=30_000, weight=0.10),
    "Logistics":          dict(mu=5_500,  sigma=4_000,  weight=0.14),
    "Facilities":         dict(mu=9_000,  sigma=6_000,  weight=0.10),
    "Marketing":          dict(mu=25_000, sigma=18_000, weight=0.09),
    "Legal":              dict(mu=60_000, sigma=40_000, weight=0.05),
    "Utilities":          dict(mu=3_200,  sigma=1_200,  weight=0.08),
    "Staffing":           dict(mu=70_000, sigma=50_000, weight=0.07),
    "Raw Materials":      dict(mu=120_000,sigma=80_000, weight=0.07),
}

CURRENCIES   = ["USD"] * 60 + ["EUR"] * 20 + ["GBP"] * 10 + ["CAD"] * 5 + ["AUD"] * 5
PAYMENT_METHODS = ["ACH", "Wire", "Check", "Credit Card", "EFT"]
STATUSES     = ["paid", "paid", "paid", "paid", "approved", "pending", "disputed", "cancelled"]

COUNTRIES = ["US", "US", "US", "US", "GB", "DE", "FR", "CA", "AU", "IN", "SG"]

DEPARTMENTS = [
    "Engineering", "Marketing", "Finance", "Operations",
    "HR", "Legal", "Sales", "IT", "Procurement", "R&D"
]

# ── Seasonal weight per month (index 1-12) ───────────────────────────────────
# Q4 spike (Oct-Dec), Q1 slowdown, summer dip
MONTH_WEIGHTS = {
    1: 0.065, 2: 0.060, 3: 0.080, 4: 0.085, 5: 0.085, 6: 0.075,
    7: 0.065, 8: 0.065, 9: 0.080, 10: 0.090, 11: 0.095, 12: 0.100,
}

# Day-of-month weights: spikes at start (1-5) and end (25-31) of month
def dom_weight(d):
    if d <= 5:   return 1.8
    if d >= 25:  return 1.6
    return 1.0

# ── Helpers ──────────────────────────────────────────────────────────────────
def random_date_with_seasonality():
    """Pick a random issue_date respecting monthly + day-of-month seasonality."""
    # Build a simple weighted sample over all days in range
    # We precompute a lookup; for 50k draws, sample from pre-built pool
    pass  # Handled via pre-built pool below

def lognormal_amount(mu, sigma):
    """Return a positive invoice amount from a lognormal approximation."""
    ln_mu    = np.log(mu**2 / np.sqrt(sigma**2 + mu**2))
    ln_sigma = np.sqrt(np.log(1 + (sigma / mu)**2))
    val = np.random.lognormal(ln_mu, ln_sigma)
    return max(round(float(val), 2), 10.0)

def payment_delay_days(category, status):
    """Realistic net-30/60/90 delays plus late payments."""
    if status in ("cancelled", "disputed"):
        return None
    base = random.choices([30, 45, 60, 90], weights=[0.4, 0.25, 0.25, 0.10])[0]
    # Staffing & consulting paid faster; raw materials slower
    if category in ("Staffing", "IT Services"):   base = max(15, base - 15)
    if category in ("Raw Materials", "Legal"):     base += 15
    late = random.random()
    if late < 0.15:   base += random.randint(10, 45)   # late
    elif late < 0.05: base -= random.randint(5, 15)    # early
    return max(1, base)

# ── 1. Vendors ────────────────────────────────────────────────────────────────
print("Generating vendors...")
cat_names   = list(VENDOR_CATEGORIES.keys())
cat_weights = [VENDOR_CATEGORIES[c]["weight"] for c in cat_names]

vendors = []
for _ in range(N_VENDORS):
    cat = random.choices(cat_names, weights=cat_weights)[0]
    vendors.append({
        "id":         str(uuid.uuid4()),
        "name":       fake.company(),
        "tax_id":     fake.ein(),
        "country":    random.choice(COUNTRIES),
        "category":   cat,
        "is_active":  random.random() > 0.05,
        "created_at": fake.date_time_between(start_date="-5y", end_date="-2y"),
    })

vendors_df = pd.DataFrame(vendors)

# ── 2. Buyers ─────────────────────────────────────────────────────────────────
print("Generating buyers...")
buyers = []
for _ in range(N_BUYERS):
    buyers.append({
        "id":         str(uuid.uuid4()),
        "name":       fake.name(),
        "department": random.choice(DEPARTMENTS),
        "country":    random.choice(["US", "US", "US", "GB", "CA"]),
        "created_at": fake.date_time_between(start_date="-6y", end_date="-2y"),
    })

buyers_df = pd.DataFrame(buyers)

# ── 3. Pre-build seasonal date pool ──────────────────────────────────────────
print("Building seasonal date pool...")
all_days = [START_DATE + timedelta(days=d) for d in range(TOTAL_DAYS)]
day_weights = [MONTH_WEIGHTS[d.month] * dom_weight(d.day) for d in all_days]
total_w = sum(day_weights)
day_probs = [w / total_w for w in day_weights]

issue_dates_sampled = np.random.choice(len(all_days), size=N_INVOICES, replace=True, p=day_probs)

# ── 4. Invoices ───────────────────────────────────────────────────────────────
print("Generating invoices...")

vendor_ids = vendors_df["id"].tolist()
vendor_cats = dict(zip(vendors_df["id"], vendors_df["category"]))
buyer_ids  = buyers_df["id"].tolist()

# Some vendors get disproportionately more invoices (power law)
vendor_popularity = np.random.pareto(1.5, N_VENDORS) + 1
vendor_popularity /= vendor_popularity.sum()

invoices = []
for i in range(N_INVOICES):
    vendor_id = np.random.choice(vendor_ids, p=vendor_popularity)
    cat       = vendor_cats[vendor_id]
    dist      = VENDOR_CATEGORIES[cat]

    issue_date   = all_days[issue_dates_sampled[i]]
    delay        = payment_delay_days(cat, "paid")  # rough; refined below
    status       = random.choices(STATUSES)[0]
    amount       = lognormal_amount(dist["mu"], dist["sigma"])
    currency     = random.choice(CURRENCIES)
    due_date     = issue_date + timedelta(days=random.choices([30, 45, 60, 90], weights=[0.4, 0.25, 0.25, 0.10])[0])

    if status == "paid":
        pd_delay      = payment_delay_days(cat, status)
        payment_date  = issue_date + timedelta(days=pd_delay) if pd_delay else None
    else:
        payment_date  = None

    # Inject ~3% anomalies: duplicate amounts, round numbers, weekend payments
    anomaly_type = None
    if random.random() < 0.015:    # round-number invoices
        amount = round(amount / 1000) * 1000
        anomaly_type = "round_amount"
    elif random.random() < 0.01:   # duplicate invoice scenario — same vendor, same amount, close date
        anomaly_type = "potential_duplicate"
    elif payment_date and payment_date.weekday() >= 5 and random.random() < 0.3:
        anomaly_type = "weekend_payment"

    invoices.append({
        "id":             str(uuid.uuid4()),
        "vendor_id":      vendor_id,
        "buyer_id":       random.choice(buyer_ids),
        "invoice_number": f"INV-{fake.numerify('######')}",
        "amount":         amount,
        "currency":       currency,
        "issue_date":     issue_date,
        "due_date":       due_date,
        "payment_date":   payment_date,
        "status":         status,
        "payment_method": random.choice(PAYMENT_METHODS),
        "anomaly_type":   anomaly_type,   # ground-truth label for ML training
        "created_at":     datetime.combine(issue_date, datetime.min.time()) + timedelta(hours=random.randint(8, 18)),
    })

invoices_df = pd.DataFrame(invoices)

# ── 5. Line items (avg 3 per invoice) ────────────────────────────────────────
print("Generating line items...")

LINE_DESCS = {
    "IT Services":     ["Software license", "Cloud hosting", "Support contract", "Implementation fee", "API access"],
    "Office Supplies": ["Paper ream", "Toner cartridge", "Desk accessories", "Printer ink", "Stationery"],
    "Consulting":      ["Strategy advisory", "Project management", "Due diligence", "Workshop facilitation", "Report"],
    "Logistics":       ["Freight shipping", "Warehousing", "Last-mile delivery", "Customs clearance", "Fuel surcharge"],
    "Facilities":      ["Cleaning services", "Security patrol", "HVAC maintenance", "Electrical repair", "Landscaping"],
    "Marketing":       ["Campaign management", "Creative design", "Media buying", "SEO services", "Analytics"],
    "Legal":           ["Contract review", "Litigation support", "Compliance audit", "IP filing", "Retainer fee"],
    "Utilities":       ["Electricity", "Water", "Gas", "Internet", "Phone"],
    "Staffing":        ["Contract labor", "Recruitment fee", "Temp staff week", "Onboarding fee", "Payroll processing"],
    "Raw Materials":   ["Steel coil", "Aluminum sheet", "Resin pellets", "Copper wire", "Chemical compound"],
}

line_items = []
inv_ids    = invoices_df["id"].tolist()
inv_amounts = dict(zip(invoices_df["id"], invoices_df["amount"]))
inv_vendor  = dict(zip(invoices_df["id"], invoices_df["vendor_id"]))

for inv_id in inv_ids:
    cat   = vendor_cats[inv_vendor[inv_id]]
    descs = LINE_DESCS.get(cat, ["Service rendered", "Goods supplied"])
    n_items = random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.30, 0.25, 0.12, 0.08])[0]
    total_inv = inv_amounts[inv_id]

    # Split invoice amount across line items
    splits = np.random.dirichlet(np.ones(n_items))
    for j, split in enumerate(splits):
        item_total = round(total_inv * split, 2)
        qty        = random.choices([1, 2, 3, 5, 10, 20], weights=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03])[0]
        unit_price = round(item_total / qty, 4)
        desc       = random.choice(descs)
        line_items.append({
            "id":            str(uuid.uuid4()),
            "invoice_id":    inv_id,
            "description":   desc,
            "quantity":      qty,
            "unit_price":    unit_price,
            "total":         item_total,
            "category_code": cat.upper().replace(" ", "_")[:20],
        })

line_items_df = pd.DataFrame(line_items)

# ── 6. Save CSVs ──────────────────────────────────────────────────────────────
print("Saving CSVs...")
out = "/data"
vendors_df.to_csv(f"{out}/vendors.csv",     index=False)
buyers_df.to_csv(f"{out}/buyers.csv",       index=False)
invoices_df.to_csv(f"{out}/invoices.csv",   index=False)
line_items_df.to_csv(f"{out}/line_items.csv", index=False)

# ── 7. Summary stats ──────────────────────────────────────────────────────────
print("\n── Dataset Summary ──────────────────────────────────────────")
print(f"  Vendors:     {len(vendors_df):>8,}")
print(f"  Buyers:      {len(buyers_df):>8,}")
print(f"  Invoices:    {len(invoices_df):>8,}")
print(f"  Line items:  {len(line_items_df):>8,}")
print(f"\n  Invoice amount stats (USD equiv):")
print(f"    Min:    ${invoices_df['amount'].min():>12,.2f}")
print(f"    Median: ${invoices_df['amount'].median():>12,.2f}")
print(f"    Mean:   ${invoices_df['amount'].mean():>12,.2f}")
print(f"    P95:    ${invoices_df['amount'].quantile(0.95):>12,.2f}")
print(f"    Max:    ${invoices_df['amount'].max():>12,.2f}")
print(f"\n  Status breakdown:")
for s, c in invoices_df["status"].value_counts().items():
    print(f"    {s:<12} {c:>6,}  ({c/N_INVOICES*100:.1f}%)")
print(f"\n  Anomaly labels injected:")
at = invoices_df["anomaly_type"].value_counts(dropna=False)
for t, c in at.items():
    label = str(t) if t is not None else "none"
    print(f"    {label:<22} {c:>6,}")
print(f"\n  Date range: {invoices_df['issue_date'].min()} → {invoices_df['issue_date'].max()}")
print(f"\n  Files written to {out}/")
print("    vendors.csv, buyers.csv, invoices.csv, line_items.csv")
