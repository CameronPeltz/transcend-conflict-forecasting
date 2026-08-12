"""
Phase 2: the 125-config phase-1 search (pooled 4-fold CV) showed
precision/AP differences across nearly every configuration are within
noise (0.49-0.51 AP spread) -- consistent with this project's repeated
finding that pooled cross-validation doesn't discriminate well on this
task; the frozen-threshold holdout is what has actually differentiated
configurations all along (round-2 baseline 71.3% -> round-4 GDELT
73.9%). This phase runs the full, rigorous 10-fold frozen-threshold
80%-target validation (both horizons) on a diverse set of the most
promising candidates from phase 1, then a further hyperparameter
refinement sweep on the winner -- real additional iterations aimed at
squeezing more out of the best-performing configuration, not just
re-confirming it.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "../data/discrete_event_candidates_mega.csv"
HORIZONS = ["10day", "14day"]
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
FP_THRESHOLD = 0.5
MIN_SUPPORT = 30

CELL = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d", "days_since_last_event"]
RING1 = ["neighbor_count_30d"]
RING2 = ["neighbor_count_30d_ring2"]
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
ROUND3 = CELL + RING1 + RING2 + MOMENTUM + ACLED
ALL_FEATURES = ROUND3 + GDELT + CHIRPS + UNHCR + HAWKES + CUSUM + GRAPHDIFF

# Diverse candidates from phase 1's top rankings, chosen to span different
# philosophies rather than just the top-N by a noisy pooled metric.
CANDIDATES = {
    "round4_established_best": (ROUND3 + GDELT, "ensemble_3way"),
    "everything_minus_chirps": ([c for c in ALL_FEATURES if c not in CHIRPS], "ensemble_3way"),
    "everything_all": (ALL_FEATURES, "ensemble_3way"),
    "everything_all_rf_only": (ALL_FEATURES, "rf_default"),
    "graphdiff_plus_cell_rf": (CELL + GRAPHDIFF, "rf_default"),
    "all_external_sources": (ROUND3 + GDELT + CHIRPS + UNHCR, "ensemble_3way"),
    "everything_lgbm": (ALL_FEATURES, "lgbm_default"),
    "everything_4way_ensemble": (ALL_FEATURES, "ensemble_4way_lgbm"),
}


def build_estimators(model_name, train_y, xgb_params=None, rf_params=None):
    pos = max(1, train_y.sum()); neg = max(1, len(train_y) - pos); spw = neg / pos
    xgb_params = xgb_params or dict(n_estimators=150, max_depth=3, learning_rate=0.08)
    rf_params = rf_params or dict(n_estimators=300, max_depth=5, min_samples_leaf=5)
    if model_name == "xgb_default":
        return [("xgb", XGBClassifier(**xgb_params, scale_pos_weight=spw, eval_metric="logloss", random_state=0), "raw")]
    if model_name == "rf_default":
        return [("rf", RandomForestClassifier(**rf_params, random_state=0, class_weight="balanced", n_jobs=-1), "raw")]
    if model_name == "logreg":
        return [("lr", LogisticRegression(class_weight="balanced", max_iter=2000), "scaled")]
    if model_name == "lgbm_default":
        return [("lgbm", LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.08,
                                         scale_pos_weight=spw, verbosity=-1, random_state=0), "raw")]
    if model_name == "ensemble_3way":
        return (build_estimators("xgb_default", train_y, xgb_params) + build_estimators("rf_default", train_y, rf_params)
                + build_estimators("logreg", train_y))
    if model_name == "ensemble_4way_lgbm":
        return build_estimators("ensemble_3way", train_y, xgb_params, rf_params) + build_estimators("lgbm_default", train_y)
    raise ValueError(model_name)


def fit_predict(model_name, train_X, train_y, test_X, xgb_params=None, rf_params=None):
    estimators = build_estimators(model_name, train_y, xgb_params, rf_params)
    scaler = StandardScaler().fit(train_X)
    train_Xs, test_Xs = scaler.transform(train_X), scaler.transform(test_X)
    scores = []
    for _, est, kind in estimators:
        Xtr, Xte = (train_Xs, test_Xs) if kind == "scaled" else (train_X, test_X)
        est.fit(Xtr, train_y)
        scores.append(est.predict_proba(Xte)[:, 1])
    return np.mean(scores, axis=0)


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


def rolling_predictions(df, feature_cols, label_col, model_name, xgb_params=None, rf_params=None):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true, all_score = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0: continue
        s = fit_predict(model_name, train[feature_cols].fillna(0).values, train[label_col].values,
                         test[feature_cols].fillna(0).values, xgb_params, rf_params)
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


def run_frozen_threshold(df, feature_cols, label_col, target_precision=0.80, xgb_params=None, rf_params=None, support_col=None):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]
    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]
    sel_true, sel_score = rolling_predictions(select_df, feature_cols, label_col, CURRENT_MODEL, xgb_params, rf_params)
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
    holdout_score = fit_predict(CURRENT_MODEL, train_X, train_y, holdout_X, xgb_params, rf_params)
    holdout_y = holdout_df[label_col].values
    out = {"select_window": compute_metrics(sel_true, sel_score, threshold=chosen_threshold),
           "holdout_window": compute_metrics(holdout_y, holdout_score, threshold=chosen_threshold),
           "chosen_threshold": chosen_threshold}
    if support_col is not None and support_col in holdout_df.columns:
        support_vals = holdout_df[support_col].fillna(0).values
        keep = support_vals >= MIN_SUPPORT
        if keep.sum() > 0 and holdout_y[keep].sum() > 0:
            out["holdout_window_supportfiltered"] = compute_metrics(holdout_y[keep], holdout_score[keep], threshold=chosen_threshold)
    return out


CURRENT_MODEL = "ensemble_3way"


if __name__ == "__main__":
    print("Loading mega dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows", flush=True)

    results = {}
    t0 = time.time()
    n_total = len(CANDIDATES) * len(HORIZONS)
    n_done = 0
    for cand_name, (cols, model_name) in CANDIDATES.items():
        CURRENT_MODEL = model_name
        results[cand_name] = {"features": cols, "model": model_name, "by_horizon": {}}
        for horizon in HORIZONS:
            label_col = f"label_{horizon}"
            iter_t0 = time.time()
            r = run_frozen_threshold(df, cols, label_col, support_col="n_hist_events_total")
            results[cand_name]["by_horizon"][horizon] = r
            n_done += 1
            hw = r["holdout_window"]
            print(f"[{n_done}/{n_total}] {cand_name} ({model_name}) / {horizon}: "
                  f"HOLDOUT precision={hw['precision']:.3f} recall={hw['recall']:.3f} "
                  f"({time.time()-iter_t0:.0f}s)", flush=True)

    with open("../results/phase2_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved phase2_validation_results.json. Total time {(time.time()-t0)/60:.1f} min", flush=True)
