"""
Rule evaluation: precision, recall, F1 at multiple thresholds.

Computes per-rule and ensemble metrics across fraud categories.
Documents exactly where rules succeed and fail — the motivation for ML.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score,
    confusion_matrix, classification_report,
)


# ── Per-rule evaluation ────────────────────────────────────────────────────────

def evaluate_binary_rule(
    y_true: pd.Series,
    y_pred: pd.Series,
    rule_name: str,
    score_col: pd.Series | None = None,
) -> dict:
    """
    Compute precision, recall, F1, and confusion matrix for a binary flag.
    Optionally compute AUC-ROC if a continuous score is provided.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        "rule":         rule_name,
        "tp":           int(tp),
        "fp":           int(fp),
        "tn":           int(tn),
        "fn":           int(fn),
        "precision":    round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":       round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":           round(f1_score(y_true, y_pred,        zero_division=0), 4),
        "flagged_total":int(y_pred.sum()),
        "flagged_pct":  round(y_pred.mean() * 100, 2),
        "auc_roc":      None,
    }
    if score_col is not None:
        try:
            result["auc_roc"] = round(roc_auc_score(y_true, score_col), 4)
        except Exception:
            pass
    return result


def evaluate_at_thresholds(
    y_true: pd.Series,
    score: pd.Series,
    rule_name: str,
    thresholds: list | None = None,
) -> pd.DataFrame:
    """
    Evaluate a continuous score at multiple cutoff thresholds.
    Returns a DataFrame useful for plotting precision-recall curves.
    """
    if thresholds is None:
        thresholds = sorted(score.dropna().unique())
        # Sample up to 50 thresholds evenly
        if len(thresholds) > 50:
            idx = np.linspace(0, len(thresholds) - 1, 50, dtype=int)
            thresholds = [thresholds[i] for i in idx]

    y_true_int = y_true.astype(int)
    rows = []
    for t in thresholds:
        y_pred = (score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true_int == 1)).sum())
        fp = int(((y_pred == 1) & (y_true_int == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_int == 1)).sum())
        tn = int(((y_pred == 0) & (y_true_int == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows.append({
            "rule":      rule_name,
            "threshold": t,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "flagged":   int(y_pred.sum()),
        })
    return pd.DataFrame(rows)


# ── Per-fraud-category breakdown ──────────────────────────────────────────────

def per_category_metrics(
    df: pd.DataFrame,
    rule_flag_col: str,
    rule_name: str,
    fraud_categories: list | None = None,
) -> pd.DataFrame:
    """
    Break down rule performance by fraud_category.
    Reveals which fraud types a rule catches and which it misses.
    """
    if fraud_categories is None:
        fraud_categories = ["duplicate", "threshold_gaming", "shell_vendor",
                            "soft_anomaly", "clean"]

    rows = []
    for cat in fraud_categories:
        if cat == "clean":
            subset = df[df["fraud_category"] == "clean"]
            label  = 0
        else:
            subset = df[df["fraud_category"].isin([cat, "clean"])]
            label  = (subset["fraud_category"] == cat).astype(int)

        if len(subset) == 0:
            continue

        flagged = subset[rule_flag_col].fillna(False).astype(int)
        if cat == "clean":
            # For clean subset: any flag = false positive
            rows.append({
                "rule":           rule_name,
                "fraud_category": "clean (FP rate)",
                "total":          len(subset),
                "flagged":        int(flagged.sum()),
                "flag_rate":      round(flagged.mean() * 100, 2),
                "precision":      None,
                "recall":         None,
            })
        else:
            true_positives = int((flagged * label).sum())
            all_fraud      = int(label.sum())
            all_flagged    = int(flagged.sum())
            prec = true_positives / all_flagged if all_flagged > 0 else 0
            rec  = true_positives / all_fraud   if all_fraud   > 0 else 0
            rows.append({
                "rule":           rule_name,
                "fraud_category": cat,
                "total":          len(subset),
                "flagged":        all_flagged,
                "flag_rate":      round(flagged.mean() * 100, 2),
                "precision":      round(prec, 4),
                "recall":         round(rec,  4),
            })
    return pd.DataFrame(rows)


# ── Ensemble rule evaluation ──────────────────────────────────────────────────

def evaluate_ensemble(df: pd.DataFrame, rule_flag_cols: list) -> pd.DataFrame:
    """
    Evaluate the ensemble at different vote thresholds (1-of-N, 2-of-N, …).
    Shows how combining rules trades precision vs recall.
    """
    y_true = df["is_fraud"].astype(int)
    vote_count = df[rule_flag_cols].fillna(False).astype(int).sum(axis=1)

    rows = []
    for min_votes in range(1, len(rule_flag_cols) + 1):
        y_pred = (vote_count >= min_votes).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows.append({
            "min_votes_to_flag": min_votes,
            "rules_needed":      f"{min_votes}/{len(rule_flag_cols)}",
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision":  round(prec, 4),
            "recall":     round(rec,  4),
            "f1":         round(f1,   4),
            "flagged":    int(y_pred.sum()),
            "flagged_pct":round(y_pred.mean() * 100, 2),
        })
    return pd.DataFrame(rows)


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_rule_summary(metrics: list[dict], title: str = "Rule Performance") -> None:
    w = 90
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")
    hdr = f"  {'Rule':<35} {'Prec':>7} {'Recall':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'Flagged%':>9}"
    print(hdr)
    print(f"  {'-'*85}")
    for m in metrics:
        auc = f"  AUC={m['auc_roc']}" if m.get("auc_roc") else ""
        print(
            f"  {m['rule']:<35} {m['precision']:>7.4f} {m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
            f"{m['flagged_pct']:>8.2f}%{auc}"
        )
    print(f"{'═'*w}")
