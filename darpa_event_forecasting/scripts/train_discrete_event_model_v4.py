"""
Round 3, lever 7 (completed): does the real, historically-matched GDELT
country-week signal -- a genuinely different, faster-reacting, news-
derived source than UCDP's own event log -- move the ~72%/69% honest
precision ceiling round 3 established?

Uses data/discrete_event_candidates_v4.csv (round 3's combined
momentum+spatial feature set, plus 10 new gdelt_* columns joined in
never-look-ahead from a real 2015-2025, 19-country GDELT pull).
Same frozen-threshold, 80%-precision-target discipline as every other
headline number in this project.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier

DATA_PATH = "../data/discrete_event_candidates_v4.csv"
HORIZONS = ["10day", "14day"]
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
FP_THRESHOLD = 0.5
MIN_SUPPORT = 30

ROUND3_COMBINED = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
                    "days_since_last_event", "neighbor_count_30d",
                    "acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth",
                    "cell_count_30d_delta", "cell_count_90d_delta", "momentum_ratio_30_90",
                    "neighbor_count_30d_ring2"]
GDELT_FEATURES = ["gdelt_n_events", "gdelt_material_conflict_share", "gdelt_mean_goldstein",
                   "gdelt_mean_tone", "gdelt_distinct_actors", "gdelt_n_events_delta",
                   "gdelt_material_conflict_share_delta", "gdelt_mean_goldstein_delta",
                   "gdelt_mean_tone_delta", "gdelt_distinct_actors_delta"]

FEATURE_SETS = {
    "round3_combined": ROUND3_COMBINED,
    "plus_gdelt": ROUND3_COMBINED + GDELT_FEATURES,
}


def fit_predict_ensemble(train_X, train_y, test_X):
    pos = max(1, train_y.sum())
    neg = max(1, len(train_y) - pos)
    gbm = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                 random_state=0, class_weight="balanced", n_jobs=-1)
    scaler = StandardScaler()
    train_Xs, test_Xs = scaler.fit_transform(train_X), scaler.transform(test_X)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)
    gbm.fit(train_X, train_y); rf.fit(train_X, train_y); logreg.fit(train_Xs, train_y)
    p = (gbm.predict_proba(test_X)[:, 1] + rf.predict_proba(test_X)[:, 1] + logreg.predict_proba(test_Xs)[:, 1]) / 3.0
    return p


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


def rolling_predictions(df, feature_cols, label_col):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true, all_score = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0:
            continue
        s = fit_predict_ensemble(train[feature_cols].fillna(0).values, train[label_col].values,
                                  test[feature_cols].fillna(0).values)
        all_true.append(test[label_col].values); all_score.append(s)
    return np.concatenate(all_true), np.concatenate(all_score)


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


def run_frozen_threshold(df, feature_cols, label_col, target_precision, support_col=None):
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

    select_metrics = compute_metrics(sel_true, sel_score, threshold=chosen_threshold)
    holdout_metrics = compute_metrics(holdout_y, holdout_score, threshold=chosen_threshold)
    out = {"split_date": str(split_date.date()), "chosen_threshold": chosen_threshold,
           "target_precision_on_select": target_precision,
           "select_window": select_metrics, "holdout_window": holdout_metrics}

    if support_col is not None:
        support_vals = holdout_df[support_col].fillna(0).values
        keep = support_vals >= MIN_SUPPORT
        if keep.sum() > 0 and holdout_y[keep].sum() > 0:
            out["holdout_window_supportfiltered"] = compute_metrics(
                holdout_y[keep], holdout_score[keep], threshold=chosen_threshold)
            out["support_filter_rows_kept"] = int(keep.sum())
            out["support_filter_rows_total"] = int(len(keep))
    return out


if __name__ == "__main__":
    print("Loading round-4 (GDELT-joined) dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows", flush=True)

    out = {}

    print("\n########## Pooled rolling-CV: round-3 combined vs plus-GDELT ##########", flush=True)
    pooled = []
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        for fs_name, cols in FEATURE_SETS.items():
            print(f"\n--- {horizon} / {fs_name} ---", flush=True)
            y_true, y_score = rolling_predictions(df, cols, label_col)
            m = compute_metrics(y_true, y_score)
            m["horizon"], m["feature_set"] = horizon, fs_name
            pooled.append(m)
            print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"AP={m['average_precision']:.3f} AUC={m['roc_auc']:.3f}", flush=True)
    out["pooled_rolling_cv"] = pooled

    print("\n\n########## Frozen-threshold @ 80% target: round-3 combined (reproducibility) vs plus-GDELT ##########", flush=True)
    frozen = {}
    for fs_name, cols in FEATURE_SETS.items():
        frozen[fs_name] = {}
        for horizon in HORIZONS:
            label_col = f"label_{horizon}"
            print(f"\n--- {horizon} / {fs_name} ---", flush=True)
            r = run_frozen_threshold(df, cols, label_col, target_precision=0.80, support_col="n_hist_events_total")
            print(f"  SELECT:  precision={r['select_window']['precision']:.3f} recall={r['select_window']['recall']:.3f}", flush=True)
            print(f"  HOLDOUT: precision={r['holdout_window']['precision']:.3f} recall={r['holdout_window']['recall']:.3f} "
                  f"n={r['holdout_window']['n']}", flush=True)
            if "holdout_window_supportfiltered" in r:
                sf = r["holdout_window_supportfiltered"]
                print(f"  HOLDOUT (support-filtered): precision={sf['precision']:.3f} recall={sf['recall']:.3f}", flush=True)
            frozen[fs_name][horizon] = r
    out["frozen_80"] = frozen

    with open("../results/round4_gdelt_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n\nSaved ../results/round4_gdelt_results.json")
