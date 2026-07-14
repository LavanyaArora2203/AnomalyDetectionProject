# Next Implementation

# I recommend the next coding task be a new script:

# models/
#     tune_isolation_forest.py

# This script will:

# Train dozens of Isolation Forest models.
# Sweep different contamination rates, tree counts, and sample sizes.
# Evaluate each run against fraud_labels.csv.
# Log every experiment to MLflow.
# Produce a ranked results table (CSV) showing the best-performing configurations.

# This is a much stronger addition than immediately implementing LOF because it demonstrates a complete experimentation workflow rather than just another algorithm.