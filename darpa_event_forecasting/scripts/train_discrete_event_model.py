"""
A few real iterations and optimizations on the discrete, ACLED-style,
grid-cell-level event-forecasting task built by
build_discrete_event_dataset.py -- testing the actual real DARPA
program specification (10-day and two-week lead-time horizons,
discrete geolocated events) rather than the country-week binary
classifier used throughout the rest of this project.

Same discipline as everywhere else in this project: rolling-origin
(expanding-window, never-look-ahead) backtesting, the established
"current best approach" ensemble (XGBoost + Random Forest + Logistic
Regression, simple probability averaging), and the same full metrics
suite (precision, recall, F1, Brier, average precision, ROC-AUC,
log-loss, MCC), plus explicit false-positive RATE -- because "reduce
within-country false-positive rates by roughly half" is one of the
program's own named Month 6 targets, not an incidental metric.

Iterations: 4 feature sets x 2 horizons = 8 real configurations.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              log_loss, matthews_corrcoef, precision_score, recall_score,
                              f1_score, confusion_matrix)
from xgboost import XGBClassifier

DATA_PATH = "../data/discrete_event_candidates.csv"

FEATURE_SETS = {
    "cell_only": ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
                  "days_since_last_event"],
    "cell_plus_spatial": ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
                           "days_since_last_event", "neighbor_count_30d"],
    "cell_plus_country": ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
                           "days_since_last_event", "neighbor_count_30d", "country_count_30d"],
    "full_combined": ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
                       "days_since_last_event", "neighbor_count_30d", "country_count_30d",
                       "actor_diversity", "type_share_state_based", "type_share_one_sided"],
}
HORIZONS = ["10day", "14day"]
MIN_TRAIN_ISSUE_DATES = 52  # >=1 year of weekly issue dates before the first test fold
N_FOLDS = 10
FP_THRESHOLD = 0.5  # a fixed, disclosed operating threshold for the false-positive-rate metric


def fit_predict_ensemble(train_X, train_y, test_X):
    pos = max(1, train_y.sum())
    neg = max(1, len(train_y) - pos)
    gbm = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                 random_state=0, class_weight="balanced", n_jobs=-1)
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    test_Xs = scaler.transform(test_X)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)

    gbm.fit(train_X, train_y)
    rf.fit(train_X, train_y)
    logreg.fit(train_Xs, train_y)

    p_gbm = gbm.predict_proba(test_X)[:, 1]
    p_rf = rf.predict_proba(test_X)[:, 1]
    p_logreg = logreg.predict_proba(test_Xs)[:, 1]
    return (p_gbm + p_rf + p_logreg) / 3.0


def rolling_folds(issue_dates_sorted, min_train, n_folds):
    """Expanding-window rolling-origin folds, evenly spaced across the
    remaining issue dates -- same discipline as large_panel.py
    elsewhere in this project."""
    remaining = len(issue_dates_sorted) - min_train
    if remaining < n_folds:
        n_folds = remaining
    step = max(1, remaining // n_folds)
    folds = []
    for k in range(n_folds):
        cutoff_idx = min_train + k * step
        if cutoff_idx >= len(issue_dates_sorted) - 1:
            break
        test_idx_start = cutoff_idx
        test_idx_end = min(cutoff_idx + step, len(issue_dates_sorted))
        train_cutoff_date = issue_dates_sorted[cutoff_idx]
        test_dates = issue_dates_sorted[test_idx_start:test_idx_end]
        folds.append((train_cutoff_date, test_dates))
    return folds


def evaluate_config(df, feature_cols, label_col):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)

    all_y_true, all_y_score = [], []
    fold_metrics = []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0 or test[label_col].nunique() < 2:
            continue
        train_X = train[feature_cols].fillna(0).values
        train_y = train[label_col].values
        test_X = test[feature_cols].fillna(0).values
        test_y = test[label_col].values

        scores = fit_predict_ensemble(train_X, train_y, test_X)
        all_y_true.append(test_y)
        all_y_score.append(scores)

    if not all_y_true:
        return None
    y_true = np.concatenate(all_y_true)
    y_score = np.concatenate(all_y_score)
    y_pred = (y_score >= FP_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fp_rate = fp / max(fp + tn, 1)  # false positives as a share of all real negatives

    return {
        "n_predictions": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp_rate),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_folds_used": len(all_y_true),
    }


if __name__ == "__main__":
    print("Loading discrete-event candidate dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows, {df['issue_date'].nunique()} weekly issue dates, "
          f"{df['priogrid_gid'].nunique()} active cells", flush=True)

    import os
    RESULTS_PATH = "../results/discrete_event_forecasting_results.json"
    results = []
    done = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done = {(r["horizon"], r["feature_set"]) for r in results}
        print(f"Resuming: {len(done)} configs already done.", flush=True)

    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        for fs_name, cols in FEATURE_SETS.items():
            if (horizon, fs_name) in done:
                continue
            print(f"\n=== {horizon} / {fs_name} ===", flush=True)
            m = evaluate_config(df, cols, label_col)
            if m is None:
                print("  skipped (insufficient data)", flush=True)
                continue
            m["horizon"] = horizon
            m["feature_set"] = fs_name
            m["features"] = cols
            results.append(m)
            print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"AP={m['average_precision']:.3f} AUC={m['roc_auc']:.3f} "
                  f"FP_rate={m['false_positive_rate']:.3f} (n={m['n_predictions']})", flush=True)
            # checkpoint after every config -- cheap insurance given this session's
            # history of unexplained background-process kills on long-running jobs
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} configuration results.")
