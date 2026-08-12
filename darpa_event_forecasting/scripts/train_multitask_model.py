"""
Multi-task joint occurrence+severity model: this round's two-stage fix
(two separately trained models, chained) turned a broken severity
result into a working one. This tests the natural next step -- one
shared model trained on a single 3-class target (0 = no event,
1 = event, not severe, 2 = event, severe) via XGBoost's native
multi:softprob objective, so the SAME trees that learn what predicts
occurrence also see severity labels during training, rather than
severity only ever being modeled downstream of occurrence's decisions.
P(any event) = P(class 1) + P(class 2); P(severe) = P(class 2) directly
from one model's output, evaluated the same frozen-threshold way as
every other result in this project.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier

DATA_PATH = "../data/discrete_event_candidates_v3.csv"
HORIZONS = ["10day", "14day"]
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
FP_THRESHOLD = 0.5

COMBINED_FEATURES = [
    "cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
    "days_since_last_event", "neighbor_count_30d",
    "acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth",
    "cell_count_30d_delta", "cell_count_90d_delta", "momentum_ratio_30_90",
    "neighbor_count_30d_ring2",
]


def make_3class_label(df, occ_col, severe_col):
    return np.where(df[severe_col] == 1, 2, np.where(df[occ_col] == 1, 1, 0))


def fit_predict_multitask(train_X, train_y3, test_X):
    """train_y3 in {0,1,2}. Returns (p_occurrence, p_severe)."""
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                         objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", random_state=0)
    clf.fit(train_X, train_y3)
    proba = clf.predict_proba(test_X)  # columns: [P(0), P(1), P(2)]
    p_occ = proba[:, 1] + proba[:, 2]
    p_severe = proba[:, 2]
    return p_occ, p_severe


def compute_metrics(y_true, y_score, threshold=FP_THRESHOLD):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)), "n_positives": int(y_true.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
    }


def rolling_folds(issue_dates_sorted, min_train, n_folds):
    remaining = len(issue_dates_sorted) - min_train
    if remaining < n_folds:
        n_folds = remaining
    step = max(1, remaining // n_folds)
    folds = []
    for k in range(n_folds):
        cutoff_idx = min_train + k * step
        if cutoff_idx >= len(issue_dates_sorted) - 1:
            break
        test_idx_end = min(cutoff_idx + step, len(issue_dates_sorted))
        folds.append((issue_dates_sorted[cutoff_idx], issue_dates_sorted[cutoff_idx:test_idx_end]))
    return folds


def rolling_predictions_multitask(df, feature_cols, occ_col, severe_col):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true_occ, all_true_sev, all_p_occ, all_p_sev = [], [], [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[occ_col].sum() == 0:
            continue
        train_y3 = make_3class_label(train, occ_col, severe_col)
        p_occ, p_sev = fit_predict_multitask(train[feature_cols].fillna(0).values, train_y3,
                                              test[feature_cols].fillna(0).values)
        all_true_occ.append(test[occ_col].values)
        all_true_sev.append(test[severe_col].values)
        all_p_occ.append(p_occ)
        all_p_sev.append(p_sev)
    return (np.concatenate(all_true_occ), np.concatenate(all_true_sev),
            np.concatenate(all_p_occ), np.concatenate(all_p_sev))


def run_frozen_threshold_multitask(df, feature_cols, occ_col, severe_col, target_col, target_precision=0.80):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]
    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]

    sel_true_occ, sel_true_sev, sel_p_occ, sel_p_sev = rolling_predictions_multitask(
        select_df, feature_cols, occ_col, severe_col)
    sel_true = sel_true_occ if target_col == "occurrence" else sel_true_sev
    sel_score = sel_p_occ if target_col == "occurrence" else sel_p_sev

    order = np.argsort(-sel_score)
    sorted_true, sorted_score = sel_true[order], sel_score[order]
    chosen_threshold = None
    for k in range(10, len(sorted_true)):
        if sorted_true[:k].sum() / k < target_precision:
            chosen_threshold = float(sorted_score[k - 1]); break
    if chosen_threshold is None:
        chosen_threshold = float(sorted_score[-1]) if len(sorted_score) else 0.5

    train_y3 = make_3class_label(select_df, occ_col, severe_col)
    holdout_p_occ, holdout_p_sev = fit_predict_multitask(
        select_df[feature_cols].fillna(0).values, train_y3, holdout_df[feature_cols].fillna(0).values)
    holdout_score = holdout_p_occ if target_col == "occurrence" else holdout_p_sev
    holdout_true = holdout_df[occ_col].values if target_col == "occurrence" else holdout_df[severe_col].values

    select_metrics = compute_metrics(sel_true, sel_score, threshold=chosen_threshold)
    holdout_metrics = compute_metrics(holdout_true, holdout_score, threshold=chosen_threshold)
    return {"split_date": str(split_date.date()), "chosen_threshold": chosen_threshold,
            "target_precision_on_select": target_precision,
            "select_window": select_metrics, "holdout_window": holdout_metrics}


if __name__ == "__main__":
    print("Loading round-3 dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows", flush=True)

    out = {}
    print("\n########## Pooled rolling-CV: multi-task joint model ##########", flush=True)
    pooled = {}
    for horizon in HORIZONS:
        occ_col, severe_col = f"label_{horizon}", f"label_{horizon}_severe"
        print(f"\n--- {horizon} ---", flush=True)
        true_occ, true_sev, p_occ, p_sev = rolling_predictions_multitask(df, COMBINED_FEATURES, occ_col, severe_col)
        m_occ = compute_metrics(true_occ, p_occ)
        m_sev = compute_metrics(true_sev, p_sev)
        print(f"  occurrence: precision={m_occ['precision']:.3f} recall={m_occ['recall']:.3f} AUC={m_occ['roc_auc']:.3f}", flush=True)
        print(f"  severity:   precision={m_sev['precision']:.3f} recall={m_sev['recall']:.3f} AUC={m_sev['roc_auc']:.3f}", flush=True)
        pooled[horizon] = {"occurrence": m_occ, "severity": m_sev}
    out["pooled_rolling_cv"] = pooled

    print("\n\n########## Frozen-threshold @ 80% target: multi-task joint model ##########", flush=True)
    frozen = {}
    for horizon in HORIZONS:
        occ_col, severe_col = f"label_{horizon}", f"label_{horizon}_severe"
        frozen[horizon] = {}
        for target_col in ["occurrence", "severity"]:
            print(f"\n--- {horizon} / {target_col} ---", flush=True)
            r = run_frozen_threshold_multitask(df, COMBINED_FEATURES, occ_col, severe_col, target_col, target_precision=0.80)
            print(f"  SELECT:  precision={r['select_window']['precision']:.3f} recall={r['select_window']['recall']:.3f}", flush=True)
            print(f"  HOLDOUT: precision={r['holdout_window']['precision']:.3f} recall={r['holdout_window']['recall']:.3f}", flush=True)
            frozen[horizon][target_col] = r
    out["frozen_80"] = frozen

    with open("../results/multitask_model_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n\nSaved ../results/multitask_model_results.json")
