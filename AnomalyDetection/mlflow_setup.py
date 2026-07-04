"""
MLflow tracking server setup for invoice anomaly detection.
Uses SQLite backend (required in MLflow 3.x — file store is deprecated).

Run once:
    python mlflow_setup.py

Start UI in a separate terminal:
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
    then open http://localhost:5000
"""

import json, sys
from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent
MLRUNS_DIR      = PROJECT_ROOT / "mlruns"
DB_PATH         = MLRUNS_DIR / "mlflow.db"
TRACKING_URI    = f"sqlite:///{DB_PATH.resolve()}"
EXPERIMENT_NAME = "invoice_anomaly_detection"

RULE_BASELINE = {
    "rule_shell_bank_f1":           0.697,
    "rule_shell_bank_precision":    1.000,
    "rule_shell_bank_recall":       0.535,
    "rule_ensemble_2of4_f1":        0.154,
    "rule_ensemble_2of4_precision": 0.108,
    "rule_ensemble_2of4_recall":    0.266,
    "total_invoices":               51420,
    "total_fraud_invoices":         1720,
    "fraud_rate_pct":               3.35,
}

TAGS = {
    "project":    "invoice_anomaly_detection",
    "layer":      "ml_anomaly_detection",
    "dataset":    "synthetic_51k_invoices",
}


def main():
    print(f"\n{'═'*60}")
    print("  MLflow tracking server setup  (v3.x SQLite backend)")
    print(f"{'═'*60}\n")

    # ── Create dirs ──────────────────────────────────────────────────────────
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"  Tracking URI : {TRACKING_URI}")

    # ── Create/get experiment ────────────────────────────────────────────────
    client   = MlflowClient(tracking_uri=TRACKING_URI)
    existing = client.get_experiment_by_name(EXPERIMENT_NAME)
    if existing:
        exp_id = existing.experiment_id
        print(f"  Experiment   : '{EXPERIMENT_NAME}' exists (id={exp_id})")
    else:
        exp_id = mlflow.create_experiment(EXPERIMENT_NAME, tags=TAGS)
        print(f"  Experiment   : '{EXPERIMENT_NAME}' created (id={exp_id})")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Log rule baseline as reference run ───────────────────────────────────
    with mlflow.start_run(run_name="rule_based_baseline",
                          tags={**TAGS, "model_type": "rule_baseline"}):
        mlflow.log_params({
            "model_type":        "rule_baseline",
            "rules_used":        "benford,threshold_gaming,duplicate,shell_bank",
            "ensemble_strategy": "vote_2_of_4",
        })
        mlflow.log_metrics({
            "precision":              RULE_BASELINE["rule_ensemble_2of4_precision"],
            "recall":                 RULE_BASELINE["rule_ensemble_2of4_recall"],
            "f1":                     RULE_BASELINE["rule_ensemble_2of4_f1"],
            "best_rule_f1":           RULE_BASELINE["rule_shell_bank_f1"],
            "best_rule_precision":    RULE_BASELINE["rule_shell_bank_precision"],
            "best_rule_recall":       RULE_BASELINE["rule_shell_bank_recall"],
            "total_invoices":         RULE_BASELINE["total_invoices"],
            "fraud_invoices":         RULE_BASELINE["total_fraud_invoices"],
            "fraud_rate_pct":         RULE_BASELINE["fraud_rate_pct"],
            "alerts_per_week_budget": 50,
            "catch_rate_at_50_alerts":0.0,
        })
        baseline_json = PROJECT_ROOT / "baseline_metrics.json"
        baseline_json.write_text(json.dumps(RULE_BASELINE, indent=2))
        mlflow.log_artifact(str(baseline_json))

    print("  Baseline run : logged ✓")

    # ── Write mlflow.env for training scripts ────────────────────────────────
    env_path = PROJECT_ROOT / "mlflow.env"
    env_path.write_text(
        f"MLFLOW_TRACKING_URI={TRACKING_URI}\n"
        f"MLFLOW_EXPERIMENT_NAME={EXPERIMENT_NAME}\n"
        f"MLFLOW_EXPERIMENT_ID={exp_id}\n"
    )
    print(f"  Env file     : mlflow.env ✓")

    # ── Verify: list runs ────────────────────────────────────────────────────
    runs = client.search_runs(experiment_ids=[exp_id])
    print(f"  Runs logged  : {len(runs)}")

    print(f"""
{'─'*60}
  SETUP COMPLETE
{'─'*60}

  Start the MLflow UI (in a separate terminal):

    cd {PROJECT_ROOT}
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

  Then open:  http://localhost:5000

  Tracking URI (copy this into every training script):
    {TRACKING_URI}

{'─'*60}
""")


if __name__ == "__main__":
    main()
