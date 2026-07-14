import pandas as pd
from pathlib import Path

# Company approval threshold
APPROVAL_THRESHOLD = 50000

# Consider invoices within 2% below threshold
THRESHOLD_MARGIN = 0.02

# Vendor must submit at least 3 such invoices in a quarter
MIN_THRESHOLD_INVOICES = 3

def load_invoice_data():

    data_path = Path("data/invoices.csv")

    invoices = pd.read_csv(data_path)

    invoices["invoice_date"] = pd.to_datetime(
    invoices["invoice_date"],
    errors="coerce"
)

    return invoices

def calculate_threshold_distance(df):

    df = df.copy()

    df["distance_from_threshold"] = (
        APPROVAL_THRESHOLD - df["amount"]
    )

    return df

def filter_near_threshold(df):

    lower_limit = APPROVAL_THRESHOLD * (
        1 - THRESHOLD_MARGIN
    )

    suspicious = df[

        (df["amount"] >= lower_limit)

        &

        (df["amount"] < APPROVAL_THRESHOLD)

    ].copy()

    return suspicious

def add_quarter(df):

    df = df.copy()

    df["quarter"] = (
        df["invoice_date"]
        .dt
        .to_period("Q")
        .astype(str)
    )

    return df

def count_vendor_threshold_invoices(df):

    counts = (

        df.groupby(

            ["vendor_id", "quarter"]

        )

        .size()

        .reset_index(

            name="vendor_threshold_count"

        )

    )

    return counts

def merge_counts(df, counts):

    return df.merge(

        counts,

        on=["vendor_id", "quarter"],

        how="left"

    )
def apply_threshold_rule(df):

    df = df.copy()

    df["threshold_flag"] = (

        df["vendor_threshold_count"]

        >=

        MIN_THRESHOLD_INVOICES

    ).astype(int)

    return df


def save_results(df):

    output_dir = Path("data/outputs")

    output_dir.mkdir(exist_ok=True)

    df.to_csv(

        output_dir / "threshold_results.csv",

        index=False

    )

def main():

    invoices = load_invoice_data()

    invoices = calculate_threshold_distance(invoices)

    suspicious = filter_near_threshold(invoices)

    suspicious = add_quarter(suspicious)

    counts = count_vendor_threshold_invoices(suspicious)

    suspicious = merge_counts(

        suspicious,

        counts

    )

    suspicious = apply_threshold_rule(

        suspicious

    )

    save_results(suspicious)

    print()

    print("Threshold Gaming Detection Completed")

    print()

    print(

        "Candidate invoices :",

        len(suspicious)

    )

    print(

        "Flagged invoices :",

        suspicious["threshold_flag"].sum()

    )

    print()

    print(

        suspicious

        .sort_values(

            "vendor_threshold_count",

            ascending=False

        )

        .head(10)

    )


if __name__ == "__main__":

    main()
