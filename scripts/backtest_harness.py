"""
Real rolling-origin backtest, matching the reference document's tab-01
methodology and tab-03 harness design: strict temporal cutoffs, no
random k-fold, results pooled across folds (per-fold slices are too
small here for stable precision/recall on their own), scored against
naive persistence and historical base rate every single fold, and
stratified by lead time (1-week vs 2-week horizon, standing in for
the RFP's 7/14/30-day stratification at weekly granularity).
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_curve, average_precision_score

from models import MODEL_REGISTRY

MIN_TRAIN_WEEKS = 6


def rolling_origin_folds(panel, label_col):
    weeks = sorted(panel["week"].unique())
    folds = []
    for i in range(MIN_TRAIN_WEEKS, len(weeks)):
        cutoff = weeks[i]
        train = panel[panel["week"] < cutoff].dropna(subset=[label_col])
        test = panel[panel["week"] == cutoff]
        if len(train) == 0 or len(test) == 0 or test[label_col].isna().all():
            continue
        folds.append((cutoff, train, test))
    return folds


def run_backtest(panel, label_col, lead_time_label):
    folds = rolling_origin_folds(panel, label_col)
    print(f"\n=== lead time: {lead_time_label} ({len(folds)} rolling-origin folds) ===")

    all_preds = {name: [] for name in MODEL_REGISTRY}
    all_labels = []
    all_meta = []

    icl_reasoning_samples = []
    gbm_importances = []

    for cutoff, train, test in folds:
        test_valid = test.dropna(subset=[label_col])
        if len(test_valid) == 0:
            continue
        y_true = test_valid[label_col].astype(int).values
        all_labels.extend(y_true.tolist())
        all_meta.extend(test_valid[["country", "week"]].to_dict("records"))

        for name, cls in MODEL_REGISTRY.items():
            model = cls()
            model.fit(train, label_col)
            probs = model.predict_proba(test_valid, label_col)
            all_preds[name].extend(np.asarray(probs).tolist())
            if name == "icl_temporal_graph" and len(icl_reasoning_samples) < 12:
                icl_reasoning_samples.extend(model.reasoning_log[-len(test_valid):])
            if name == "gbm_graph_features" and cutoff == folds[-1][0]:
                # feature importances from the LAST (largest-training-window) fold --
                # the most data-rich, most representative fit in the whole backtest
                from models import FEATURES as _FEATURES
                gbm_importances = sorted(zip(_FEATURES, model.model.feature_importances_.tolist()),
                                          key=lambda t: -t[1])

    y = np.array(all_labels)
    results = {
        "lead_time": lead_time_label, "n_predictions": len(y), "n_positive": int(y.sum()),
        "naive_always_negative_accuracy": float(1 - y.mean()) if len(y) else float("nan"),
        "models": {},
    }

    print(f"  pooled out-of-sample predictions: {len(y)}, positives: {int(y.sum())} "
          f"({100*y.mean():.1f}% base rate)")

    for name in MODEL_REGISTRY:
        p = np.array(all_preds[name])
        brier = brier_score_loss(y, p)
        ap = average_precision_score(y, p) if y.sum() > 0 else float("nan")
        # precision/recall at a fixed 0.5 threshold AND at the "top-N" operating
        # point (N = number of true positives) since severe imbalance makes a
        # 0.5 cutoff uninformative for several of these models
        pred_at_05 = (p >= 0.5).astype(int)
        tp = int(((pred_at_05 == 1) & (y == 1)).sum())
        fp = int(((pred_at_05 == 1) & (y == 0)).sum())
        fn = int(((pred_at_05 == 0) & (y == 1)).sum())
        tn = int(((pred_at_05 == 0) & (y == 0)).sum())
        precision_05 = tp / max(1, tp + fp)
        recall_05 = tp / max(1, tp + fn)
        specificity_05 = tn / max(1, tn + fp)
        accuracy_05 = (tp + tn) / max(1, tp + tn + fp + fn)
        f1_05 = 2 * precision_05 * recall_05 / max(1e-9, precision_05 + recall_05)

        n_top = max(1, int(y.sum()))
        top_idx = np.argsort(-p)[:n_top]
        precision_topN = y[top_idx].mean() if n_top > 0 else float("nan")

        results["models"][name] = {
            "brier": float(brier), "average_precision": float(ap),
            "precision_at_0.5": float(precision_05), "recall_at_0.5": float(recall_05),
            "specificity_at_0.5": float(specificity_05), "accuracy_at_0.5": float(accuracy_05),
            "f1_at_0.5": float(f1_05),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision_at_topN": float(precision_topN), "n_flagged_at_0.5": int(tp + fp),
            "delta_brier_vs_base_rate": None,
        }
        print(f"    {name:22s} Brier={brier:.4f}  AP={ap:.4f}  Acc={accuracy_05:.3f}  "
              f"P@0.5={precision_05:.3f}  R@0.5={recall_05:.3f}  Spec@0.5={specificity_05:.3f}  "
              f"P@top{n_top}={precision_topN:.3f}  [TP={tp} FP={fp} TN={tn} FN={fn}]")

    base_brier = results["models"]["historical_base_rate"]["brier"]
    for name in MODEL_REGISTRY:
        results["models"][name]["delta_brier_vs_base_rate"] = base_brier - results["models"][name]["brier"]

    return results, icl_reasoning_samples, gbm_importances


def main():
    panel = pd.read_csv("data/country_week_panel_v2.csv", parse_dates=["week"])

    all_results = []
    r1, samples1, imp1 = run_backtest(panel, "label_1w_ahead", "1-week horizon")
    all_results.append(r1)
    r2, samples2, imp2 = run_backtest(panel, "label_2w_ahead", "2-week horizon")
    all_results.append(r2)

    with open("data/backtest_results_v2.json", "w") as f:
        json.dump({
            "results": all_results,
            "icl_reasoning_samples": samples1[:8],
            "gbm_feature_importance_1w": imp1,
            "gbm_feature_importance_2w": imp2,
        }, f, indent=2, default=str)
    print("\nwrote data/backtest_results_v2.json")


if __name__ == "__main__":
    main()
