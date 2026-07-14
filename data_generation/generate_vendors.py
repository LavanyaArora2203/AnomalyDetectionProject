from faker import Faker
import pandas as pd
import random
from datetime import datetime
from config import *

fake = Faker("en_IN")

random.seed(42)

vendors = []

for i in range(1, VENDOR_COUNT + 1):

    vendor_id = f"V{i:04d}"

    category = random.choice(list(VENDOR_CATEGORIES.keys()))

    vendor = {
        "vendor_id": vendor_id,

        "vendor_name": fake.company(),

        "vendor_category": category,

        "bank_account_id": fake.bban(),

        "vendor_address": fake.address().replace("\n", ", "),

        "onboarding_date": fake.date_between(
            start_date="-5y",
            end_date="-6m"
        ),

        "payment_terms": random.choice(PAYMENT_TERMS)
    }

    vendors.append(vendor)

vendors_df = pd.DataFrame(vendors)

vendors_df.to_csv(
    "data/vendors.csv",
    index=False
)

print(vendors_df.head())
print(f"\nGenerated {len(vendors_df)} vendors.")