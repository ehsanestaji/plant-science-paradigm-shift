"""
G2 (deferred) — Score filled-in classifier_validation_sample.csv.

This script is written now but not run until the domain-expert returns the
filled-in ground-truth CSV.  It will exit with a clear error if any
true_organism or true_paradigm cell is blank.

Usage
-----
python3 -u -m src.analysis.paper_a.hardening.validation_scorer
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_CSV = (
    PROJECT_ROOT
    / "results/paper_a/supplementary/hardening/classifier_validation_sample.csv"
)


def print_confusion_matrix(labels, cm, title="Confusion Matrix"):
    print(f"\n### {title}")
    header = "| Predicted→ | " + " | ".join(str(l) for l in labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    print(header)
    print(sep)
    for i, true_label in enumerate(labels):
        row = f"| **{true_label}** | " + " | ".join(str(cm[i, j]) for j in range(len(labels))) + " |"
        print(row)


def bias_check(df: pd.DataFrame, classifier: str):
    """Fisher's exact test: does accuracy differ by confidence level?"""
    pred_col = f"predicted_{classifier}"
    true_col = f"true_{classifier}"
    conf_col = f"{classifier}_confidence"

    if conf_col not in df.columns:
        return

    df = df.copy()
    df["correct"] = df[pred_col] == df[true_col]
    low_conf = df[df[conf_col] < 0.9]["correct"]
    high_conf = df[df[conf_col] >= 0.9]["correct"]

    if len(low_conf) == 0 or len(high_conf) == 0:
        return

    table = [
        [low_conf.sum(), (~low_conf).sum()],
        [high_conf.sum(), (~high_conf).sum()],
    ]
    _, p = fisher_exact(table)
    print(f"\nAccuracy bias by confidence (Fisher's exact, {classifier}): p = {p:.4f}")
    if p < 0.05:
        print("  ⚠  Significant accuracy difference between confidence groups.")
    else:
        print("  ✓  No significant accuracy difference between confidence groups.")


def score_classifier(df: pd.DataFrame, pred_col: str, true_col: str, conf_col: str, name: str):
    print(f"\n{'='*60}")
    print(f"CLASSIFIER: {name.upper()}")
    print(f"{'='*60}")

    y_true = df[true_col]
    y_pred = df[pred_col]

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"  Accuracy:       {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Cohen's κ:      {kappa:.4f}")
    print(f"  Macro-avg F1:   {macro_f1:.4f}")

    labels = sorted(y_true.unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    print(f"\nPer-class metrics:")
    hdr = f"  {'Label':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}"
    print(hdr)
    print("  " + "-" * 60)
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<35} {precision[i]:>6.3f} {recall[i]:>6.3f} {f1[i]:>6.3f} {support[i]:>6}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print_confusion_matrix(labels, cm, title=f"{name} Confusion Matrix")


def main():
    if not SAMPLE_CSV.exists():
        print(f"ERROR: sample file not found: {SAMPLE_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SAMPLE_CSV, dtype=str)

    # Guard: check for blank ground truth
    blank_org = df["true_organism"].isna() | (df["true_organism"].str.strip() == "")
    blank_par = df["true_paradigm"].isna() | (df["true_paradigm"].str.strip() == "")

    if blank_org.any():
        n_blank = blank_org.sum()
        print(
            f"ERROR: {n_blank} rows have blank 'true_organism'. "
            "Fill in all ground-truth labels before running this scorer.",
            file=sys.stderr,
        )
        sys.exit(1)

    if blank_par.any():
        n_blank = blank_par.sum()
        print(
            f"ERROR: {n_blank} rows have blank 'true_paradigm'. "
            "Fill in all ground-truth labels before running this scorer.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(df)} reviewed papers. Computing scores …")

    # Organism classifier
    score_classifier(
        df,
        pred_col="predicted_organism",
        true_col="true_organism",
        conf_col="organism_confidence",
        name="organism",
    )
    bias_check(df, "organism")

    # Paradigm classifier
    score_classifier(
        df,
        pred_col="predicted_paradigm",
        true_col="true_paradigm",
        conf_col="paradigm_confidence",
        name="paradigm",
    )
    bias_check(df, "paradigm")

    print("\n\nDone.  Copy the output above into the methods supplement.")


if __name__ == "__main__":
    main()
