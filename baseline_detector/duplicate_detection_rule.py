import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
# Maximum allowed percentage difference in invoice amount
ABSOLUTE_AMOUNT_TOLERANCE = 1.00      # 1%

# Maximum difference between invoice dates
DATE_WINDOW_DAYS = 7

# Minimum description similarity
DESCRIPTION_SIMILARITY = 90

def load_invoice_data():

    invoices = pd.read_csv("data/invoices.csv")

    invoices["invoice_date"] = pd.to_datetime(
        invoices["invoice_date"],
        errors="coerce"
    )

    return invoices

def prepare_data(df):

    return df.sort_values(

        ["vendor_id", "invoice_date"]

    ).reset_index(drop=True)


def generate_candidate_pairs(df):

    candidate_pairs = []

    grouped = df.groupby("vendor_id")

    for vendor_id, vendor_df in grouped:

        vendor_df = vendor_df.reset_index(drop=True)

        for i in range(len(vendor_df)):

            current = vendor_df.iloc[i]

            for j in range(i + 1, len(vendor_df)):

                compare = vendor_df.iloc[j]

                day_difference = (

                    compare["invoice_date"]

                    -

                    current["invoice_date"]

                ).days

                if day_difference > DATE_WINDOW_DAYS:

                    break

                candidate_pairs.append(

                    (current, compare)

                )

    return candidate_pairs

def amount_similarity(amount1, amount2):

    difference = abs(amount1 - amount2)

    percentage = difference / max(amount1, amount2)

    return percentage

def description_similarity(desc1, desc2):

    return fuzz.token_sort_ratio(

        str(desc1),

        str(desc2)

    )

def detect_duplicates(candidate_pairs):

    duplicates = []

    for inv1, inv2 in candidate_pairs:

        amount_diff = amount_similarity(

            inv1["amount"],

            inv2["amount"]

        )

        if amount_diff > ABSOLUTE_AMOUNT_TOLERANCE:

            continue

        similarity = description_similarity(

            inv1["description"],

            inv2["description"]

        )

        if similarity < DESCRIPTION_SIMILARITY:

            continue

        duplicates.append({

            "invoice_id": inv1["invoice_id"],

            "matched_invoice": inv2["invoice_id"],

            "vendor_id": inv1["vendor_id"],

            "amount_difference": round(

                amount_diff,

                2

            ),

            "days_difference": abs(

                (

                    inv2["invoice_date"]

                    -

                    inv1["invoice_date"]

                ).days

            ),

            "description_similarity": similarity,

            "duplicate_flag": 1

        })

    return pd.DataFrame(duplicates)

def save_results(df):

    output_dir = Path("data/outputs")

    output_dir.mkdir(exist_ok=True)

    df.to_csv(

        output_dir /

        "duplicate_results.csv",

        index=False

    )

def amount_difference(amount1, amount2):
    """
    Returns the absolute difference between two invoice amounts.
    """
    return abs(amount1 - amount2)
def main():

    invoices = load_invoice_data()

    invoices = prepare_data(invoices)

    candidate_pairs = generate_candidate_pairs(

        invoices

    )

    duplicate_df = detect_duplicates(

        candidate_pairs

    )

    save_results(duplicate_df)

    print()

    print("Duplicate Detection Complete")

    print()

    print(

        f"Candidate pairs checked : {len(candidate_pairs)}"

    )

    print(

        f"Duplicates found : {len(duplicate_df)}"

    )

    print()

    print(

        duplicate_df.head()

    )


if __name__ == "__main__":

    main()


    