import pandas as pd
import numpy as np
from pathlib import Path
BENFORD_DISTRIBUTION = {
    1: 0.301,
    2: 0.176,
    3: 0.125,
    4: 0.097,
    5: 0.079,
    6: 0.067,
    7: 0.058,
    8: 0.051,
    9: 0.046,
}
MIN_INVOICES_PER_VENDOR = 30

def load_invoice_data():

    data_path = Path("data/invoices.csv")

    invoices = pd.read_csv(data_path)

    return invoices

def extract_leading_digit(amount):

    amount = abs(float(amount))

    while amount >= 10:
        amount //= 10

    return int(amount)

def add_leading_digit(df):

    df = df.copy()

    df["leading_digit"] = df["amount"].apply(extract_leading_digit)

    return df

def compute_vendor_distribution(df):

    distributions = {}

    vendor_counts = {}

    for vendor_id, vendor_df in df.groupby("vendor_id"):

        # Skip vendors with too few invoices
        if len(vendor_df) < MIN_INVOICES_PER_VENDOR:
            continue

        counts = (
            vendor_df["leading_digit"]
            .value_counts(normalize=True)
            .reindex(range(1, 10), fill_value=0)
        )

        distributions[vendor_id] = counts

        vendor_counts[vendor_id] = len(vendor_df)

    return distributions, vendor_counts

def calculate_chi_square(observed_distribution):

    chi_square = 0.0

    for digit in range(1, 10):

        observed = observed_distribution[digit]

        expected = BENFORD_DISTRIBUTION[digit]

        chi_square += ((observed - expected) ** 2) / expected

    return chi_square

def score_all_vendors(vendor_distributions, vendor_counts):

    scores = []

    for vendor_id, distribution in vendor_distributions.items():

        chi_square = calculate_chi_square(distribution)

        scores.append({

            "vendor_id": vendor_id,

            "invoice_count": vendor_counts[vendor_id],

            "chi_square_score": round(chi_square, 4)

        })

    return pd.DataFrame(scores)

BENFORD_THRESHOLD = 2.0
# or BENFORD_THRESHOLD = 1.5

# threshold = mean + 2 * standard deviation
def apply_threshold(scores_df):

    mean_score = scores_df["chi_square_score"].mean()

    std_score = scores_df["chi_square_score"].std()

    threshold = mean_score + (2 * std_score)

    scores_df["benford_flag"] = (

        scores_df["chi_square_score"] > threshold

    ).astype(int)

    return scores_df, threshold

def save_results(results):

    output_path = Path("data/outputs")

    output_path.mkdir(exist_ok=True)

    results.to_csv(

        output_path / "benford_results.csv",

        index=False

    )



# def main():

#     invoices = load_invoice_data()

#     invoices = add_leading_digit(invoices)

#     vendor_distributions, vendor_counts = compute_vendor_distribution(invoices)

#     scores = score_all_vendors(
#     vendor_distributions,
#     vendor_counts
# )

#     scores, threshold = apply_threshold(scores)

#     save_results(scores)

#     print(f"Benford threshold: {threshold:.3f}")

#     print(scores.sort_values("chi_square_score", ascending=False).head())

# if __name__ == "__main__":
#     main()

