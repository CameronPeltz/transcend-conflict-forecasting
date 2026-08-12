"""Evaluates the UNHCR asylum-application feature against the round-3 baseline."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier

HORIZONS = ["10day", "14day"]
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
FP_THRESHOLD = 0.5

ROUND3_COMBINED = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
                    "days_since_last_event", "neighbor_count_30d",
                    "acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth",
                    "cell_count_30d_delta", "cell_count_90d_delta", "momentum_ratio_30_90",
                    "neighbor_count_30d_ring2"]
FEATURE_SETS = {
    "round3_combined": ROUND3_COMBINED,
    "plus_unhcr": ROUND3_COMBINED + ["unhcr_asylum_applications"],
}


def fit_predict_ensemble(train_X, train_y, test_X):
    pos = max(1, train_y.sum()); neg = max(1, len(train_y) - pos)
    gbm = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                 random_state=0, class_weight="balanced", n_jobs=-1)
    scaler = StandardScaler()
    train_Xs, test_Xs = scaler.fit_transform(train_X), scaler.transform(test_X)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)
    gbm.fit(train_X, train_y); rf.fit(train_X, train_y); logreg.fit(train_Xs, train_y)
    return (gbm.predict_proba(test_X)[:, 1] + rf.predict_proba(test_X)[:, 1] + logreg.predict_proba(test_Xs)[:, 1]) / 3.0


def rolling_folds(issue_dates_sorted, min_train, n_folds):
    remaining = len(issue_dates_sorted) - min_train
    if remaining < n_folds: n_folds = remaining
    step = max(1, remaining // n_folds)
    folds = []
    for k in range(n_folds):
        cutoff_idx = min_train + k * step
        if cutoff_idx >= len(issue_dates_sorted) - 1: break
        test_idx_end = min(cutoff_idx + step, len(issue_dates_sorted))
        folds.append((issue_dates_sorted[cutoff_idx], issue_dates_sorted[cutoff_idx:test_idx_end]))
    return folds


def rolling_predictions(df, feature_cols, label_col):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true, all_score = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0: continue
        s = fit_predict_ensemble(train[feature_cols].fillna(0).values, train[label_col].values,
                                  test[feature_cols].fillna(0).values)
        all_true.append(test[label_col].values); all_score.append(s)
    return np.concatenate(all_true), np.concatenate(all_score)


def compute_metrics(y_true, y_score, threshold=FP_THRESHOLD):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"n": int(len(y_true)), "n_positives": int(y_true.sum()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "false_positive_rate": float(fp / max(fp + tn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None}


def run_frozen_threshold(df, feature_cols, label_col, target_precision=0.80):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]
    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]
    sel_true, sel_score = rolling_predictions(select_df, feature_cols, label_col)
    order = np.argsort(-sel_score)
    sorted_true, sorted_score = sel_true[order], sel_score[order]
    chosen_threshold = None
    for k in range(10, len(sorted_true)):
        if sorted_true[:k].sum() / k < target_precision:
            chosen_threshold = float(sorted_score[k - 1]); break
    if chosen_threshold is None:
        chosen_threshold = float(sorted_score[-1]) if len(sorted_score) else 0.5
    train_X = select_df[feature_cols].fillna(0).values
    train_y = select_df[label_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    holdout_score = fit_predict_ensemble(train_X, train_y, holdout_X)
    holdout_y = holdout_df[label_col].values
    return {"select_window": compute_metrics(sel_true, sel_score, threshold=chosen_threshold),
            "holdout_window": compute_metrics(holdout_y, holdout_score, threshold=chosen_threshold)}


if __name__ == "__main__":
    df = pd.read_csv("../data/discrete_event_candidates_v8_unhcr.csv", parse_dates=["issue_date"])
    out = {"pooled": {}, "frozen_80": {}}
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        for fs_name, cols in FEATURE_SETS.items():
            y_true, y_score = rolling_predictions(df, cols, label_col)
            m = compute_metrics(y_true, y_score)
            print(f"[pooled] {horizon}/{fs_name}: precision={m['precision']:.3f} recall={m['recall']:.3f} AUC={m['roc_auc']:.3f}", flush=True)
            out["pooled"][f"{horizon}_{fs_name}"] = m
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        out["frozen_80"][horizon] = {}
        for fs_name, cols in FEATURE_SETS.items():
            r = run_frozen_threshold(df, cols, label_col)
            print(f"[frozen80] {horizon}/{fs_name}: SELECT p={r['select_window']['precision']:.3f} -> "
                  f"HOLDOUT p={r['holdout_window']['precision']:.3f} r={r['holdout_window']['recall']:.3f}", flush=True)
            out["frozen_80"][horizon][fs_name] = r
    with open("../results/unhcr_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("Saved ../results/unhcr_results.json")
