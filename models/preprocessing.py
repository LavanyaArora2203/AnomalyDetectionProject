import duckdb
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

DATABASE_PATH = Path("dev.duckdb")
TABLE_NAME = "invoice_features"


def load_feature_table():

    conn = duckdb.connect(DATABASE_PATH)

    df = conn.execute(

        f"SELECT * FROM {TABLE_NAME}"

    ).fetchdf()

    conn.close()

    return df

METADATA_COLUMNS = [

    "invoice_id",

    "vendor_id",

    "vendor_name",

    "invoice_date",

    "submission_timestamp",

    "payment_date",

    "due_date",

    "purchase_order_id",

    "contract_id",

    "description",

    "requester",

    "approver"

]


def separate_metadata(df):

    metadata = df[METADATA_COLUMNS].copy()

    features = df.drop(

        columns=METADATA_COLUMNS

    )

    return features, metadata

def identify_feature_types(df):

    numeric_columns = df.select_dtypes(

        include=["int64", "float64"]

    ).columns.tolist()

    categorical_columns = df.select_dtypes(

        include=["object"]

    ).columns.tolist()

    return numeric_columns, categorical_columns

def inspect_dataset(df, numeric_columns, categorical_columns):

    print("\n========== Feature Summary ==========")

    print(f"Rows               : {len(df)}")
    print(f"Columns            : {len(df.columns)}")
    print(f"Numeric Features   : {len(numeric_columns)}")
    print(f"Categorical Features: {len(categorical_columns)}")

    print("\nMissing Values")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\n=====================================")
def create_numeric_pipeline():

    numeric_pipeline = Pipeline(

        steps=[

            (

                "imputer",

                SimpleImputer(

                    strategy="median"

                )

            ),

            (

                "scaler",

                StandardScaler()

            )

        ]

    )

    return numeric_pipeline

def create_categorical_pipeline():

    categorical_pipeline = Pipeline(

        steps=[

            (

                "imputer",

                SimpleImputer(

                    strategy="most_frequent"

                )

            ),

            (

                "encoder",

                OneHotEncoder(

                    handle_unknown="ignore",

                    sparse_output=False

                )

            )

        ]

    )

    return categorical_pipeline

def build_preprocessor(

    numeric_columns,

    categorical_columns

):

    numeric_pipeline = create_numeric_pipeline()

    categorical_pipeline = create_categorical_pipeline()

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "numeric",

                numeric_pipeline,

                numeric_columns

            ),

            (

                "categorical",

                categorical_pipeline,

                categorical_columns

            )

        ]

    )

    return preprocessor

def preprocess_features(

    features,

    preprocessor

):

    X = preprocessor.fit_transform(

        features

    )

    return X

from pathlib import Path

ARTIFACT_DIR = Path("artifacts")

ARTIFACT_DIR.mkdir(exist_ok=True)
def save_preprocessor(

    preprocessor

):

    joblib.dump(

        preprocessor,

        ARTIFACT_DIR /

        "preprocessor.pkl"

    )

    print(

        "Saved preprocessor."

    )

def save_feature_matrix(

    X

):

    joblib.dump(

        X,

        ARTIFACT_DIR /

        "feature_matrix.pkl"

    )

    print(

        "Saved feature matrix."

    )

def save_metadata(

    metadata

):

    metadata.to_csv(

        ARTIFACT_DIR /

        "metadata.csv",

        index=False

    )

    print(

        "Saved metadata."

    )

def save_feature_names(
    preprocessor
):

    feature_names = preprocessor.get_feature_names_out()

    joblib.dump(

        feature_names,

        ARTIFACT_DIR /
        "feature_names.pkl"

    )

    print("Saved feature names.")

def main():

    print("=" * 70)
    print("Invoice Feature Preprocessing Pipeline")
    print("=" * 70)

    # --------------------------------------------------
    # Load Feature Table
    # --------------------------------------------------

    df = load_feature_table()

    print(f"\nLoaded Dataset : {df.shape}")

    # --------------------------------------------------
    # Separate Metadata
    # --------------------------------------------------

    features, metadata = separate_metadata(df)

    print(f"\nFeature Matrix Shape : {features.shape}")
    print(f"Metadata Shape       : {metadata.shape}")

    # --------------------------------------------------
    # Identify Column Types
    # --------------------------------------------------

    numeric_columns, categorical_columns = identify_feature_types(
        features
    )

    inspect_dataset(
        features,
        numeric_columns,
        categorical_columns
    )

    # --------------------------------------------------
    # Build Preprocessor
    # --------------------------------------------------

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    # --------------------------------------------------
    # Transform Dataset
    # --------------------------------------------------

    print("Transforming features...")

    X = preprocess_features(
        features,
        preprocessor
    )

    print(f"\nProcessed Feature Matrix Shape : {X.shape}")

    # --------------------------------------------------
    # Save Artifacts
    # --------------------------------------------------

    save_preprocessor(preprocessor)
    save_feature_names(
    preprocessor
)

    save_feature_matrix(X)

    save_metadata(metadata)

    print("\nAll preprocessing artifacts saved successfully!")

    print("\nArtifact Directory")

    print(ARTIFACT_DIR)


if __name__ == "__main__":
    main()
