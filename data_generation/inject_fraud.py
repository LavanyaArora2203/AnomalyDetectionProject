import json
import random
from datetime import timedelta

import numpy as np
import pandas as pd


fraud_labels = pd.DataFrame(columns=[
    "invoice_id",
    "fraud_label",
    "fraud_type",
    "source_invoice",
    "parameters"
])


def inject_duplicate_invoices(
    invoices_df: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    n_duplicates: int = 100,
    amount_delta: float = 0.01,
    max_date_shift: int = 5,
):
    """
    Create near-duplicate invoices by copying existing invoices
    while making tiny modifications that can bypass naive
    duplicate detection systems.

    Returns
    -------
    updated_invoices_df
    updated_fraud_labels
    """

    invoices = invoices_df.copy()
    labels = fraud_labels.copy()

    sampled = invoices.sample(
        n=n_duplicates,
        replace=False,
        random_state=42
    )
    max_id = (
        invoices["invoice_id"]
        .str.replace("INV", "", regex=False)
        .astype(int)
        .max()
    )

    next_invoice_number = max_id + 1
    duplicates = []

    for _, row in sampled.iterrows():

        duplicate = row.copy()
        duplicate["invoice_id"] = (
            f"INV{next_invoice_number:06d}"
        )

        next_invoice_number += 1

        shift = random.randint(
            1,
            max_date_shift
        )

        duplicate["invoice_date"] = (
            pd.to_datetime(
                duplicate["invoice_date"]
            )
            + timedelta(days=shift)
        )
        duplicate["submission_timestamp"] = (
            pd.to_datetime(
                duplicate["submission_timestamp"]
            )
            + timedelta(days=shift)
        )

        duplicate["due_date"] = (
            pd.to_datetime(
                duplicate["due_date"]
            )
            + timedelta(days=shift)
        )
        duplicate["payment_date"] = (
            pd.to_datetime(
                duplicate["payment_date"]
            )
            + timedelta(days=shift)
        )

        duplicate["amount"] = round(
            duplicate["amount"] + amount_delta,
            2
        )

        gst_rate = (
            duplicate["tax_amount"] /
            row["amount"]
        )

        duplicate["tax_amount"] = round(
            duplicate["amount"] * gst_rate,
            2
        )

        duplicate["unit_price"] = round(
            duplicate["amount"] /
            duplicate["quantity"],
            2
        )
        duplicates.append(duplicate)

        labels.loc[len(labels)] = {

            "invoice_id":
                duplicate["invoice_id"],

            "fraud_label":
                1,

            "fraud_type":
                "Duplicate Invoice",

            "source_invoice":
                row["invoice_id"],

            "parameters":
                json.dumps({

                    "amount_delta":
                        amount_delta,

                    "date_shift_days":
                        shift,

                    "same_vendor":
                        True,

                    "same_description":
                        True

                })

        }

        duplicates_df = pd.DataFrame(
        duplicates
    )

    invoices = pd.concat(
        [
            invoices,
            duplicates_df
        ],
        ignore_index=True
    )
    return invoices, labels

invoices_df = pd.read_csv(
    "data/invoices.csv"
)

fraud_labels = pd.DataFrame(columns=[
    "invoice_id",
    "fraud_label",
    "fraud_type",
    "source_invoice",
    "parameters"
])

invoices_df, fraud_labels = inject_duplicate_invoices(
    invoices_df,
    fraud_labels,
    n_duplicates=100
)

invoices_df.to_csv(
    "data/invoices.csv",
    index=False
)

fraud_labels.to_csv(
    "data/fraud_labels.csv",
    index=False
)