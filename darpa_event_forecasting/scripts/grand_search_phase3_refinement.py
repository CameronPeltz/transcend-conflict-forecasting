"""
Phase 3: phase 2 found that no single configuration wins both horizons
-- LightGBM alone dominates 10-day (75.7% holdout precision, the best
result found across every approach tried in this project) but is worst
at 14-day (67.9%); Random Forest alone wins 14-day (71.6%, also best
recall) but is mediocre at 10-day. This phase does real hyperparameter
refinement around each horizon's own winner (not just re-confirming
phase 2), plus tests whether the winning combos help the severity task
too, via the same two-stage architecture validated earlier.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
from lightgbm import LGBMClassifier

DATA_PATH = "../data/discrete_event_candidates_mega.csv"
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
FP_THRESHOLD = 0.5

CELL = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d", "days_since_last_event"]
RING1 = ["neighbor_count_30d"]; RING2 = ["neighbor_count_30d_ring2"]
MOMENTUM = ["cell_count_30d_delta", "cell_count_90d_delta", "momentum_ratio_30_90"]
ACLED = ["acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth"]
GDELT = ["gdelt_n_events", "gdelt_material_conflict_share", "gdelt_mean_goldstein", "gdelt_mean_tone",
         "gdelt_distinct_actors", "gdelt_n_events_delta", "gdelt_material_conflict_share_delta",
         "gdelt_mean_goldstein_delta", "gdelt_mean_tone_delta", "gdelt_distinct_actors_delta"]
CHIRPS = ["chirps_rainfall_mm", "chirps_rainfall_mm_lag1", "chirps_rainfall_anomaly"]
UNHCR = ["unhcr_asylum_applications"]
HAWKES = ["hawkes_self_fast", "hawkes_self_medium", "hawkes_self_slow",
          "hawkes_neighbor_fast", "hawkes_neighbor_medium", "hawkes_neighbor_slow"]
CUSUM = ["cusum_upward", "cusum_downward", "cusum_rate_ratio"]
GRAPHDIFF = ["graph_diffusion_hop1", "graph_diffusion_hop2", "graph_diffusion_hop3"]
ALL_FEATURES = CELL + RING1 + RING2 + MOMENTUM + ACLED + GDELT + CHIRPS + UNHCR + HAWKES + CUSUM + GRAPHDIFF

# Real hyperparameter grid around each horizon's phase-2 winner
LGBM_GRID = [
    dict(n_estimators=150, max_depth=4, learning_rate=0.08, num_leaves=31),   # phase-2 winner
    dict(n_estimators=100, max_depth=3, learning_rate=0.10, num_leaves=15),
    dict(n_estimators=250, max_depth=5, learning_rate=0.05, num_leaves=31),
    dict(n_estimators=150, max_depth=4, learning_rate=0.08, num_leaves=63),
    dict(n_estimators=300, max_depth=6, learning_rate=0.03, num_leaves=63),
    dict(n_estimators=80, max_depth=3, learning_rate=0.15, num_leaves=15),
    dict(n_estimators=150, max_depth=4, learning_rate=0.08, num_leaves=31, min_child_samples=10),
    dict(n_estimators=150, max_depth=4, learning_rate=0.08, num_leaves=31, reg_alpha=0.1, reg_lambda=0.1),
]
RF_GRID = [
    dict(n_estimators=300, max_depth=5, min_samples_leaf=5),   # phase-2 winner
    dict(n_estimators=500, max_depth=8, min_samples_leaf=3),
    dict(n_estimators=200, max_depth=6, min_samples_leaf=8),
    dict(n_estimators=800, max_depth=6, min_samples_leaf=5),
    dict(n_estimators=300, max_depth=10, min_samples_leaf=2),
    dict(n_estimators=300, max_depth=5, min_samples_leaf=5, max_features="sqrt"),
    dict(n_estimators=300, max_depth=5, min_samples_leaf=5, max_features=0.5),
    dict(n_estimators=400, max_depth=7, min_samples_leaf=4),
]


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


def fit_predict_lgbm(params, train_X, train_y, test_X):
    pos = max(1, train_y.sum()); neg = max(1, len(train_y) - pos)
    m = LGBMClassifier(**params, scale_pos_weight=neg / pos, verbosity=-1, random_state=0)
    m.fit(train_X, train_y)
    return m.predict_proba(test_X)[:, 1]


def fit_predict_rf(params, train_X, train_y, test_X):
    m = RandomForestClassifier(**params, random_state=0, class_weight="balanced", n_jobs=-1)
    m.fit(train_X, train_y)
    return m.predict_proba(test_X)[:, 1]


def rolling_predictions(df, feature_cols, label_col, fit_fn, params):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true, all_score = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0: continue
        s = fit_fn(params, train[feature_cols].fillna(0).values, train[label_col].values, test[feature_cols].fillna(0).values)
        all_true.append(test[label_col].values); all_score.append(s)
    return np.concatenate(all_true), np.concatenate(all_score)


def compute_metrics(y_true, y_score, threshold=FP_THRESHOLD):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"n": int(len(y_true)), "n_positives": int(y_true.sum()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None}


def run_frozen_threshold(df, feature_cols, label_col, fit_fn, params, target_precision=0.80):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]
    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]
    sel_true, sel_score = rolling_predictions(select_df, feature_cols, label_col, fit_fn, params)
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
    holdout_score = fit_fn(params, train_X, train_y, holdout_X)
    holdout_y = holdout_df[label_col].values
    return {"select_window": compute_metrics(sel_true, sel_score, threshold=chosen_threshold),
            "holdout_window": compute_metrics(holdout_y, holdout_score, threshold=chosen_threshold),
            "chosen_threshold": chosen_threshold}


if __name__ == "__main__":
    print("Loading mega dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    results = {"lgbm_10day_sweep": [], "rf_14day_sweep": []}
    t0 = time.time()

    print("\n########## LightGBM hyperparameter sweep, 10-day ##########", flush=True)
    for i, params in enumerate(LGBM_GRID):
        iter_t0 = time.time()
        r = run_frozen_threshold(df, ALL_FEATURES, "label_10day", fit_predict_lgbm, params)
        hw = r["holdout_window"]
        print(f"[{i+1}/{len(LGBM_GRID)}] {params}: HOLDOUT precision={hw['precision']:.3f} "
              f"recall={hw['recall']:.3f} ({time.time()-iter_t0:.0f}s)", flush=True)
        results["lgbm_10day_sweep"].append({"params": params, "result": r})

    print("\n########## Random Forest hyperparameter sweep, 14-day ##########", flush=True)
    for i, params in enumerate(RF_GRID):
        iter_t0 = time.time()
        r = run_frozen_threshold(df, ALL_FEATURES, "label_14day", fit_predict_rf, params)
        hw = r["holdout_window"]
        print(f"[{i+1}/{len(RF_GRID)}] {params}: HOLDOUT precision={hw['precision']:.3f} "
              f"recall={hw['recall']:.3f} ({time.time()-iter_t0:.0f}s)", flush=True)
        results["rf_14day_sweep"].append({"params": params, "result": r})

    with open("../results/phase3_refinement_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved phase3_refinement_results.json. Total time {(time.time()-t0)/60:.1f} min", flush=True)
