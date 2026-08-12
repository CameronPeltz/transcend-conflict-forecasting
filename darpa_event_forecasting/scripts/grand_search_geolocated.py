"""
Large-scale iteration search for the geo-located (grid-cell, 10-14 day)
event forecasting task specifically -- not the country-week escalation
question, the actual DARPA-specified discrete/geolocated unit. Mirrors
the parent project's own grand_search_v2.py discipline (many real
configurations, every one backtested under rolling-origin, never-
look-ahead validation, logged to JSONL, nothing hand-picked after the
fact) applied to this task's own mega feature set (base UCDP-derived
counts + momentum + spatial + ACLED + GDELT + CHIRPS + UNHCR + Hawkes-
kernel + CUSUM changepoint + graph diffusion -- everything built across
rounds 1-5).

Search phase uses a reduced 4-fold rolling-CV (vs. the 10-fold used for
final validation) specifically to make >100 real configurations
tractable in one run; the winning configuration(s) are re-validated
under the full 10-fold frozen-threshold discipline in a separate
follow-up script before being trusted.
"""
import itertools
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "../data/discrete_event_candidates_mega.csv"
LOG_PATH = "../results/grand_search_geolocated_log.jsonl"
SEARCH_HORIZON = "10day"
MIN_TRAIN_ISSUE_DATES = 52
SEARCH_N_FOLDS = 4
FP_THRESHOLD = 0.5

# ============================================================ feature families
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

FEATURE_SETS = {
    "cell_only": CELL,
    "plus_ring1": CELL + RING1,
    "plus_ring2": CELL + RING1 + RING2,
    "plus_momentum": CELL + RING1 + RING2 + MOMENTUM,
    "plus_acled": CELL + RING1 + RING2 + MOMENTUM + ACLED,
    "round3_baseline": ROUND3,
    "plus_gdelt_only": CELL + GDELT,
    "plus_chirps_only": CELL + CHIRPS,
    "plus_unhcr_only": CELL + UNHCR,
    "plus_hawkes_only": CELL + HAWKES,
    "plus_cusum_only": CELL + CUSUM,
    "plus_graphdiff_only": CELL + GRAPHDIFF,
    "round3_plus_gdelt": ROUND3 + GDELT,
    "round3_plus_gdelt_chirps": ROUND3 + GDELT + CHIRPS,
    "round3_plus_gdelt_chirps_unhcr": ROUND3 + GDELT + CHIRPS + UNHCR,
    "round3_plus_all_external": ROUND3 + GDELT + CHIRPS + UNHCR,
    "round3_plus_hawkes_cusum": ROUND3 + HAWKES + CUSUM,
    "round3_plus_graphdiff": ROUND3 + GRAPHDIFF,
    "everything": ALL_FEATURES,
    "everything_minus_gdelt": [c for c in ALL_FEATURES if c not in GDELT],
    "everything_minus_chirps": [c for c in ALL_FEATURES if c not in CHIRPS],
    "everything_minus_unhcr": [c for c in ALL_FEATURES if c not in UNHCR],
    "everything_minus_hawkes_cusum": [c for c in ALL_FEATURES if c not in HAWKES + CUSUM],
    "everything_minus_graphdiff": [c for c in ALL_FEATURES if c not in GRAPHDIFF],
    "external_only_no_ucdp_counts": GDELT + CHIRPS + UNHCR + ACLED,
}

# ============================================================ model configs
def build_estimators(model_name, train_y):
    pos = max(1, train_y.sum())
    neg = max(1, len(train_y) - pos)
    spw = neg / pos
    if model_name == "xgb_default":
        return [("xgb", XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                                       scale_pos_weight=spw, eval_metric="logloss", random_state=0), "raw")]
    if model_name == "xgb_deep":
        return [("xgb", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       scale_pos_weight=spw, eval_metric="logloss", random_state=0), "raw")]
    if model_name == "xgb_shallow_fast":
        return [("xgb", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.12,
                                       scale_pos_weight=spw, eval_metric="logloss", random_state=0), "raw")]
    if model_name == "xgb_many_trees":
        return [("xgb", XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                       scale_pos_weight=spw, eval_metric="logloss", random_state=0), "raw")]
    if model_name == "rf_default":
        return [("rf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                                random_state=0, class_weight="balanced", n_jobs=-1), "raw")]
    if model_name == "rf_deep":
        return [("rf", RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3,
                                                random_state=0, class_weight="balanced", n_jobs=-1), "raw")]
    if model_name == "extratrees":
        return [("et", ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                             random_state=0, class_weight="balanced", n_jobs=-1), "raw")]
    if model_name == "logreg":
        return [("lr", LogisticRegression(class_weight="balanced", max_iter=2000), "scaled")]
    if model_name == "lgbm_default":
        return [("lgbm", LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.08,
                                         scale_pos_weight=spw, verbosity=-1, random_state=0), "raw")]
    if model_name == "mlp":
        return [("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=0), "scaled")]
    if model_name == "ensemble_gbm_rf":
        return build_estimators("xgb_default", train_y) + build_estimators("rf_default", train_y)
    if model_name == "ensemble_3way":  # the established baseline ensemble
        return (build_estimators("xgb_default", train_y) + build_estimators("rf_default", train_y)
                + build_estimators("logreg", train_y))
    if model_name == "ensemble_4way_lgbm":
        return (build_estimators("xgb_default", train_y) + build_estimators("rf_default", train_y)
                + build_estimators("logreg", train_y) + build_estimators("lgbm_default", train_y))
    if model_name == "ensemble_gbm_lgbm":
        return build_estimators("xgb_default", train_y) + build_estimators("lgbm_default", train_y)
    if model_name == "ensemble_all6":
        return (build_estimators("xgb_default", train_y) + build_estimators("rf_default", train_y)
                + build_estimators("logreg", train_y) + build_estimators("lgbm_default", train_y)
                + build_estimators("extratrees", train_y) + build_estimators("mlp", train_y))
    raise ValueError(model_name)


MODEL_NAMES = ["xgb_default", "rf_default", "logreg", "lgbm_default", "ensemble_3way"]


def fit_predict(model_name, train_X, train_y, test_X):
    estimators = build_estimators(model_name, train_y)
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


def rolling_predictions(df, feature_cols, label_col, model_name, n_folds):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, n_folds)
    all_true, all_score = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0:
            continue
        s = fit_predict(model_name, train[feature_cols].fillna(0).values, train[label_col].values,
                         test[feature_cols].fillna(0).values)
        all_true.append(test[label_col].values); all_score.append(s)
    if not all_true:
        return None, None
    return np.concatenate(all_true), np.concatenate(all_score)


def compute_metrics(y_true, y_score, threshold=FP_THRESHOLD):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)), "n_positives": int(y_true.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
    }


def main():
    print("Loading mega dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows, {len(FEATURE_SETS)} feature sets x {len(MODEL_NAMES)} models = "
          f"{len(FEATURE_SETS) * len(MODEL_NAMES)} configurations planned", flush=True)

    label_col = f"label_{SEARCH_HORIZON}"
    t0 = time.time()
    n_done = 0
    total = len(FEATURE_SETS) * len(MODEL_NAMES)

    with open(LOG_PATH, "w", encoding="utf-8") as logf:
        for fs_name, cols in FEATURE_SETS.items():
            for model_name in MODEL_NAMES:
                iter_t0 = time.time()
                try:
                    y_true, y_score = rolling_predictions(df, cols, label_col, model_name, SEARCH_N_FOLDS)
                    if y_true is None:
                        raise ValueError("no valid folds")
                    m = compute_metrics(y_true, y_score)
                    m.update({"feature_set": fs_name, "model": model_name, "horizon": SEARCH_HORIZON,
                               "n_features": len(cols), "elapsed_sec": round(time.time() - iter_t0, 1)})
                    status = "ok"
                except Exception as e:
                    m = {"feature_set": fs_name, "model": model_name, "horizon": SEARCH_HORIZON,
                         "n_features": len(cols), "error": str(e), "elapsed_sec": round(time.time() - iter_t0, 1)}
                    status = "error"
                logf.write(json.dumps(m) + "\n")
                logf.flush()
                n_done += 1
                elapsed_total = time.time() - t0
                eta_min = (elapsed_total / n_done) * (total - n_done) / 60
                if status == "ok":
                    print(f"[{n_done}/{total}] {fs_name} / {model_name}: precision={m['precision']:.3f} "
                          f"recall={m['recall']:.3f} AP={m['average_precision']:.3f} AUC={m['roc_auc']:.3f} "
                          f"({m['elapsed_sec']:.1f}s, ETA {eta_min:.1f}m)", flush=True)
                else:
                    print(f"[{n_done}/{total}] {fs_name} / {model_name}: ERROR ({m['error']}) "
                          f"({m['elapsed_sec']:.1f}s, ETA {eta_min:.1f}m)", flush=True)

    print(f"\nDone. {n_done} configurations logged to {LOG_PATH}. Total time {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
