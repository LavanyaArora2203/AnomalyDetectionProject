import random
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from faker import Faker

from config import *

fake = Faker("en_IN")

random.seed(42)
np.random.seed(42)

# ==========================================================
# Configuration
# ==========================================================

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)

TOTAL_DAYS = (END_DATE - START_DATE).days + 1

CURRENCY = "INR"

PAYMENT_METHODS = [
    "Bank Transfer",
    "NEFT",
    "RTGS",
    "IMPS"
]

DEPARTMENTS = [
    "Finance",
    "IT",
    "Operations",
    "Manufacturing",
    "HR",
    "Procurement",
    "Marketing",
    "Administration"
]

REQUESTERS = [
    f"EMP{i:04d}"
    for i in range(1,301)
]

APPROVERS = [
    f"MGR{i:03d}"
    for i in range(1,41)
]

DESCRIPTION_MAP = {

    "Office Supplies":[
        "Printer Paper",
        "Office Chairs",
        "Printer Toner",
        "Pens and Stationery",
        "Whiteboard Supplies"
    ],

    "IT Equipment":[
        "Dell Laptop",
        "Network Switch",
        "Server Hardware",
        "27 Inch Monitor",
        "Keyboard and Mouse"
    ],

    "Software":[
        "Microsoft 365 License",
        "Cloud Subscription",
        "Database License",
        "Firewall License",
        "ERP Subscription"
    ],

    "Manufacturing":[
        "Steel Components",
        "Hydraulic Pumps",
        "Industrial Bearings",
        "Assembly Materials",
        "Machine Spare Parts"
    ],

    "Logistics":[
        "Transportation Charges",
        "Freight Charges",
        "Warehouse Rental",
        "Courier Services"
    ],

    "Maintenance":[
        "Generator Servicing",
        "Machine Maintenance",
        "HVAC Servicing",
        "Electrical Repairs"
    ],

    "Furniture":[
        "Office Desk",
        "Conference Table",
        "Storage Cabinet",
        "Office Sofa"
    ],

    "Consulting":[
        "Technical Consulting",
        "Audit Services",
        "Training Services",
        "Legal Advisory"
    ]
}
MONTH_WEIGHTS = {

    1:0.08,
    2:0.07,
    3:0.14,
    4:0.08,
    5:0.06,
    6:0.07,
    7:0.08,
    8:0.08,
    9:0.09,
    10:0.09,
    11:0.09,
    12:0.07
}

all_dates = [
    START_DATE + timedelta(days=i)
    for i in range(TOTAL_DAYS)
]

weights = [
    MONTH_WEIGHTS[d.month]
    for d in all_dates
]

weights = np.array(weights)
weights = weights / weights.sum()
def generate_invoice_date():

    idx = np.random.choice(
        len(all_dates),
        p=weights
    )

    return all_dates[idx]

def generate_amount(category):

    params = VENDOR_CATEGORIES[category]

    amount = np.random.lognormal(
        params["mu"],
        params["sigma"]
    )

    return round(amount,2)

def generate_quantity(category):

    if category == "Manufacturing":
        return random.randint(100,500)

    elif category == "Office Supplies":
        return random.randint(20,200)

    elif category == "IT Equipment":
        return random.randint(1,10)

    elif category == "Furniture":
        return random.randint(1,20)

    else:
        return random.randint(1,50)
    

def generate_payment_date(invoice_date, payment_terms):

    due_date = invoice_date + timedelta(days=payment_terms)

    delay = random.randint(-3,10)

    payment_date = due_date + timedelta(days=delay)

    return due_date, payment_date


def generate_submission_timestamp(invoice_date):

    return fake.date_time_between(
        start_date=datetime.combine(invoice_date, datetime.min.time()),
        end_date=datetime.combine(invoice_date, datetime.max.time())
    )


def calculate_tax(amount):

    gst_rate = random.uniform(0.17,0.19)

    return round(amount * gst_rate,2)


def calculate_discount(amount):

    rate = random.choice([0,0,0.05,0.10])

    return round(amount * rate,2)


# ==========================================================
# Load Vendor Master
# ==========================================================

vendors_df = pd.read_csv("data/vendors.csv")

vendor_lookup = (
    vendors_df
    .set_index("vendor_id")
    .to_dict(orient="index")
)

vendor_ids = vendors_df["vendor_id"].tolist()


# ==========================================================
# Vendor Popularity
# ==========================================================

vendor_popularity = np.random.pareto(
    a=1.5,
    size=len(vendor_ids)
)

vendor_popularity += 1

vendor_popularity = (
    vendor_popularity /
    vendor_popularity.sum()
)

# ==========================================================
# Generate Invoices
# ==========================================================

invoices = []

for i in range(1, INVOICE_COUNT + 1):

    invoice_id = f"INV{i:06d}"

    vendor_id = np.random.choice(
        vendor_ids,
        p=vendor_popularity
    )

    vendor = vendor_lookup[vendor_id]

    category = vendor["vendor_category"]

    payment_terms = int(vendor["payment_terms"])


    invoice_date = generate_invoice_date()

    submission_timestamp = generate_submission_timestamp(
        invoice_date
    )

    due_date, payment_date = generate_payment_date(
        invoice_date,
        payment_terms
    )

    amount = generate_amount(category)

    quantity = generate_quantity(category)

    unit_price = round(
        amount / quantity,
        2
    )

    tax_amount = calculate_tax(amount)

    discount = calculate_discount(amount)


    description = random.choice(
        DESCRIPTION_MAP[category]
    )


    requester = random.choice(
        REQUESTERS
    )

    approver = random.choice(
        APPROVERS
    )

    department = random.choice(
        DEPARTMENTS
    )


    purchase_order_id = (
        f"PO{random.randint(100000,999999)}"
    )

    contract_id = (
        f"CTR{random.randint(10000,99999)}"
    )

    invoice = {

        "invoice_id": invoice_id,

        "vendor_id": vendor_id,

        "invoice_date": invoice_date,

        "submission_timestamp": submission_timestamp,

        "payment_date": payment_date,

        "due_date": due_date,

        "amount": amount,

        "quantity": quantity,

        "unit_price": unit_price,

        "currency": CURRENCY,

        "requester": requester,

        "approver": approver,

        "purchase_order_id": purchase_order_id,

        "contract_id": contract_id,

        "payment_terms": payment_terms,

        "payment_method": random.choice(
            PAYMENT_METHODS
        ),

        "invoice_status": "Paid",

        "department": department,

        "description": description,

        "tax_amount": tax_amount,

        "discount": discount
    }

    invoices.append(invoice)


# ==========================================================
# Convert to DataFrame
# ==========================================================

invoices_df = pd.DataFrame(invoices)
# ==========================================================
# Data Validation
# ==========================================================

print("\nValidating generated invoices...")

# Check duplicate invoice IDs
assert invoices_df["invoice_id"].is_unique, \
    "Duplicate invoice IDs found!"

# Check for missing values
assert invoices_df.isnull().sum().sum() == 0, \
    "Missing values detected!"

# Amount should always be positive
assert (invoices_df["amount"] > 0).all(), \
    "Negative invoice amount found!"

# Quantity must be positive
assert (invoices_df["quantity"] > 0).all(), \
    "Invalid quantity found!"

# Unit price must be positive
assert (invoices_df["unit_price"] > 0).all(), \
    "Invalid unit price found!"

# Payment date should never be before invoice date
assert (
    invoices_df["payment_date"] >= invoices_df["invoice_date"]
).all(), \
    "Payment date before invoice date!"

# Due date should never be before invoice date
assert (
    invoices_df["due_date"] >= invoices_df["invoice_date"]
).all(), \
    "Due date before invoice date!"

# Currency check
assert (
    invoices_df["currency"] == "INR"
).all(), \
    "Unexpected currency detected!"

print("All validation checks passed.")

# ==========================================================
# Generate Line Items
# ==========================================================

line_items = []

for _, row in invoices_df.iterrows():

    line_items.append({

        "invoice_id": row["invoice_id"],

        "line_number": 1,

        "description": row["description"],

        "quantity": row["quantity"],

        "unit_price": row["unit_price"],

        "line_total": row["amount"]

    })

line_items_df = pd.DataFrame(line_items)

# ==========================================================
# Dataset Summary
# ==========================================================

print("\n================ Dataset Summary ================")

print(f"Total Vendors       : {vendors_df.shape[0]:,}")
print(f"Total Invoices      : {invoices_df.shape[0]:,}")
print(f"Total Line Items    : {line_items_df.shape[0]:,}")

print("\nInvoice Amount Statistics")

print(invoices_df["amount"].describe())

print("\nVendor Categories")

print(
    vendors_df["vendor_category"]
    .value_counts()
)

print("\nDepartments")

print(
    invoices_df["department"]
    .value_counts()
)

print("\nPayment Terms")

print(
    invoices_df["payment_terms"]
    .value_counts()
)

print("\nDate Range")

print(
    invoices_df["invoice_date"].min(),
    " --> ",
    invoices_df["invoice_date"].max()
)
# ==========================================================
# Save Output
# ==========================================================

import os

os.makedirs("output", exist_ok=True)

# vendors_df.to_csv(
#     "output/vendors.csv",
#     index=False
# )

invoices_df.to_csv(
    "data/invoices.csv",
    index=False
)

line_items_df.to_csv(
    "data/line_items.csv",
    index=False
)

print("\nFiles written successfully.")

print("data/vendors.csv")

print("data/invoices.csv")

print("data/line_items.csv")