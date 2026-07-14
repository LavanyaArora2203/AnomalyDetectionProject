import pandas as pd
from pathlib import Path

MIN_SHARED_VENDORS = 2
def load_vendor_data():

    vendors = pd.read_csv("data/vendors.csv")

    return vendors

def count_shared_bank_accounts(df):

    counts = (

        df.groupby("bank_account_id")["vendor_id"]

        .nunique()

        .reset_index(name="vendor_count")

    )

    return counts

def merge_counts(vendors, counts):

    return vendors.merge(

        counts,

        on="bank_account_id",

        how="left"

    )
def detect_shell_vendors(df):

    df = df.copy()

    df["shell_vendor_flag"] = (

        df["vendor_count"]

        >= MIN_SHARED_VENDORS

    ).astype(int)

    return df

def save_results(df):

    output_dir = Path("data/outputs")

    output_dir.mkdir(exist_ok=True)

    df.to_csv(

        output_dir /

        "shell_vendor_results.csv",

        index=False

    )
def main():

    vendors = load_vendor_data()

    counts = count_shared_bank_accounts(

        vendors

    )

    vendors = merge_counts(

        vendors,

        counts

    )

    vendors = detect_shell_vendors(

        vendors

    )

    save_results(vendors)

    print()

    print("Shell Vendor Detection Complete")

    print()

    print(

        "Flagged vendors :",

        vendors["shell_vendor_flag"].sum()

    )

    print()

    print(

        vendors[

            vendors["shell_vendor_flag"] == 1

        ].head(20)

    )


if __name__ == "__main__":

    main()