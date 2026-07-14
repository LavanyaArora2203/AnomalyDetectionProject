import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/outputs")

BENFORD_FILE = OUTPUT_DIR / "benford_results.csv"
THRESHOLD_FILE = OUTPUT_DIR / "threshold_results.csv"
DUPLICATE_FILE = OUTPUT_DIR / "duplicate_results.csv"
SHELL_VENDOR_FILE = OUTPUT_DIR / "shell_vendor_results.csv"

INVOICE_FILE = Path("data/invoices.csv")

def load_data():

    invoices = pd.read_csv(INVOICE_FILE)

    benford = pd.read_csv(BENFORD_FILE)

    threshold = pd.read_csv(THRESHOLD_FILE)

    duplicate = pd.read_csv(DUPLICATE_FILE)

    shell = pd.read_csv(SHELL_VENDOR_FILE)
    shell.columns = shell.columns.str.strip()

    return invoices, benford, threshold, duplicate, shell


def initialize_predictions(invoices):
    predictions = invoices[["invoice_id", "vendor_id"]].copy()

    predictions["threshold_flag"] = 0
    predictions["duplicate_flag"] = 0
    predictions["shell_vendor_flag"] = 0

    return predictions

def merge_benford(predictions, benford):

    predictions = predictions.merge(

        benford[

            ["vendor_id", "benford_flag"]

        ],

        on="vendor_id",

        how="left"

    )

    predictions["benford_flag"] = (

        predictions["benford_flag"]

        .fillna(0)

        .astype(int)

    )
    # print(predictions.columns.tolist())

    return predictions

def merge_shell_vendor(predictions, shell):

    predictions = predictions.merge(

        shell[

            ["vendor_id", "shell_vendor_flag"]

        ],

        on="vendor_id",

        how="left"

    )

    predictions["shell_vendor_flag"] = (

        predictions["shell_vendor_flag"]

        .fillna(0)

        .astype(int)

    )

    return predictions

def merge_threshold(predictions, threshold):

    threshold = threshold[

        ["invoice_id", "threshold_flag"]

    ]

    predictions = predictions.merge(

        threshold,

        on="invoice_id",

        how="left"

    )

    predictions["threshold_flag"] = (

        predictions["threshold_flag"]

        .fillna(0)

        .astype(int)

    )

    return predictions

def merge_duplicate(predictions, duplicate):

    duplicate = duplicate[

        ["invoice_id", "duplicate_flag"]

    ].drop_duplicates()

    predictions = predictions.merge(

        duplicate,

        on="invoice_id",

        how="left"

    )

    predictions["duplicate_flag"] = (

        predictions["duplicate_flag"]

        .fillna(0)

        .astype(int)

    )

    return predictions

def compute_final_prediction(predictions):

    predictions = predictions.copy()

    predictions["final_prediction"] = (

        (
            predictions["benford_flag"] == 1
        )

        |

        (
            predictions["threshold_flag"] == 1
        )

        |

        (
            predictions["duplicate_flag"] == 1
        )

        |

        (
            predictions["shell_vendor_flag"] == 1
        )

    ).astype(int)

    return predictions

def save_predictions(predictions):

    output_path = OUTPUT_DIR / "baseline_predictions.csv"

    predictions.to_csv(

        output_path,

        index=False

    )

    print(f"\nSaved predictions to: {output_path}")

def print_summary(predictions):

    print("\n========== Baseline Rule Summary ==========\n")

    print(

        f"Total invoices           : {len(predictions)}"

    )

    print(

        f"Benford flags            : {predictions['benford_flag'].sum()}"

    )

    print(

        f"Threshold flags          : {predictions['threshold_flag'].sum()}"

    )

    print(

        f"Duplicate flags          : {predictions['duplicate_flag'].sum()}"

    )

    print(

        f"Shell vendor flags       : {predictions['shell_vendor_flag'].sum()}"

    )

    print(

        f"Final flagged invoices   : {predictions['final_prediction'].sum()}"

    )

    print("\n==========================================")

def main():

    invoices, benford, threshold, duplicate, shell = load_data()

    predictions = initialize_predictions(invoices)

    predictions = merge_benford(

        predictions,

        benford

    )

    predictions = merge_shell_vendor(

        predictions,

        shell

    )

    predictions = merge_threshold(

        predictions,

        threshold

    )

    predictions = merge_duplicate(

        predictions,

        duplicate

    )

    predictions = compute_final_prediction(

        predictions

    )

    save_predictions(predictions)

    print_summary(predictions)


if __name__ == "__main__":

    main()


##evaluate baseline and run baseline are left


