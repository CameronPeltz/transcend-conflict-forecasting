"""
Generates the real, small derived-data artifacts the HTML report's
visualizations are drawn from: a precision-recall curve for the best
configuration per horizon, false-positive-rate-vs-threshold points
(directly relevant to the program's "reduce false-positive rate by
half" Month 6 target), and a naive frequency-only baseline for
comparison (same discipline as the rest of this project: always show
whether the modeled approach beats a naive heuristic, not just that it
beats chance).
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score, roc_auc_score

from train_discrete_event_model import (FEATURE_SETS, HORIZONS, MIN_TRAIN_ISSUE_DATES, N_FOLDS,
                                         rolling_folds, fit_predict_ensemble)

BEST_FEATURE_SET = "cell_plus_spatial"  # best or tied-best on both horizons per the sweep


def get_predictions(df, feature_cols, label_col):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_y_true, all_y_score = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0:
            continue
        scores = fit_predict_ensemble(train[feature_cols].fillna(0).values, train[label_col].values,
                                       test[feature_cols].fillna(0).values)
        all_y_true.append(test[label_col].values)
        all_y_score.append(scores)
    return np.concatenate(all_y_true), np.concatenate(all_y_score)


def naive_frequency_baseline(df, label_col):
    """Score = cell_count_90d alone, min-max normalized -- the simplest
    possible real heuristic ('this cell has been busy lately'), for a
    fair floor to compare the real ensemble against."""
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_y_true, all_y_score = [], []
    for train_cutoff, test_dates in folds:
        test = df[df["issue_date"].isin(test_dates)]
        if len(test) == 0 or test[label_col].sum() == 0:
            continue
        raw = test["cell_count_90d"].values.astype(float)
        score = raw / (raw.max() + 1e-9)
        all_y_true.append(test[label_col].values)
        all_y_score.append(score)
    return np.concatenate(all_y_true), np.concatenate(all_y_score)


if __name__ == "__main__":
    df = pd.read_csv("../data/discrete_event_candidates.csv", parse_dates=["issue_date"])
    out = {}
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        print(f"Computing PR curve + threshold sweep for {horizon} ({BEST_FEATURE_SET})...", flush=True)
        y_true, y_score = get_predictions(df, FEATURE_SETS[BEST_FEATURE_SET], label_col)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        # subsample the curve to a manageable number of points for the chart
        idx = np.linspace(0, len(precision) - 1, min(60, len(precision))).astype(int)

        fp_by_threshold = []
        for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            pred = (y_score >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            fp_by_threshold.append({
                "threshold": t, "false_positive_rate": float(fp / max(fp + tn, 1)),
                "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
            })

        print(f"  computing naive frequency-only baseline for {horizon}...", flush=True)
        nb_true, nb_score = naive_frequency_baseline(df, label_col)

        out[horizon] = {
            "pr_curve": [{"precision": float(precision[i]), "recall": float(recall[i])} for i in idx],
            "fp_by_threshold": fp_by_threshold,
            "ensemble_ap": float(average_precision_score(y_true, y_score)),
            "ensemble_auc": float(roc_auc_score(y_true, y_score)),
            "naive_baseline_ap": float(average_precision_score(nb_true, nb_score)),
            "naive_baseline_auc": float(roc_auc_score(nb_true, nb_score)),
        }
        print(f"  {horizon}: ensemble AP={out[horizon]['ensemble_ap']:.3f} vs "
              f"naive-frequency-only AP={out[horizon]['naive_baseline_ap']:.3f}", flush=True)

    with open("../results/visualization_data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved ../results/visualization_data.json")
