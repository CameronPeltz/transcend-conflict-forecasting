"""
Round 3: tests whether the six levers identified after round 2 actually
move precision toward 80%, one at a time and combined, all under the
same rolling-origin discipline used throughout, PLUS (new this round)
every meaningful configuration is checked under frozen-threshold,
never-look-ahead holdout validation targeting 80% precision specifically
-- not just the 50% target round 2 used. That's the real number: round 2
already showed a 50%-target threshold degrades to ~40% out of sample, so
an 80%-target number is only trustworthy if it survives the same test.

Levers tested this round:
  1. Momentum/trend features (cell_count_30d_delta, 90d_delta, momentum_ratio)
  2. Wider spatial radius (neighbor_count_30d_ring2, Chebyshev distance 2)
  3. Combined (1)+(2) on top of the round-2 best feature set
  4. Severity-tightened label (>=5 fatalities in the horizon window,
     instead of any UCDP-coded event) using the combined feature set
  5. Minimum-support floor (suppressing cells with < MIN_SUPPORT real
     historical events), applied post-hoc to the frozen-threshold holdout
  6. PRIO-GRID covariates and a historically-matched GDELT re-pull were
     both investigated and are NOT included -- disclosed honestly in the
     write-up, not silently dropped.

Levers 7 (ICL/LLM hybrid on ambiguous cases) is a separate scoped script,
scripts/icl_hybrid_test.py, since it calls a live local LLM per case
rather than fitting into this ensemble harness.
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

DATA_PATH = "../data/discrete_event_candidates_v3.csv"
HORIZONS = ["10day", "14day"]
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
FP_THRESHOLD = 0.5
MIN_SUPPORT = 30  # minimum real historical events a cell must have to be scored

ROUND2_BEST = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
               "days_since_last_event", "neighbor_count_30d",
               "acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth"]
MOMENTUM = ["cell_count_30d_delta", "cell_count_90d_delta", "momentum_ratio_30_90"]
RING2 = ["neighbor_count_30d_ring2"]

FEATURE_SETS = {
    "round2_best": ROUND2_BEST,
    "plus_momentum": ROUND2_BEST + MOMENTUM,
    "plus_ring2_spatial": ROUND2_BEST + RING2,
    "plus_momentum_plus_ring2": ROUND2_BEST + MOMENTUM + RING2,
}


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
    all_idx, all_true, all_score = [], [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0:
            continue
        scores = fit_predict_ensemble(train[feature_cols].fillna(0).values, train[label_col].values,
                                       test[feature_cols].fillna(0).values)
        all_idx.append(test.index.values)
        all_true.append(test[label_col].values)
        all_score.append(scores)
    return np.concatenate(all_idx), np.concatenate(all_true), np.concatenate(all_score)


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


def run_feature_sweep(df):
    results = []
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        for fs_name, cols in FEATURE_SETS.items():
            print(f"\n=== [sweep] {horizon} / {fs_name} ===", flush=True)
            idx, y_true, y_score = rolling_predictions(df, cols, label_col)
            m = compute_metrics(y_true, y_score)
            m["horizon"], m["feature_set"], m["features"] = horizon, fs_name, cols
            results.append(m)
            print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"AP={m['average_precision']:.3f} AUC={m['roc_auc']:.3f} "
                  f"FP_rate={m['false_positive_rate']:.3f}", flush=True)
    return results


def run_frozen_threshold_validation(df, feature_cols, label_col, target_precision, support_col=None):
    """Same discipline as round 2, but parameterized on target_precision so
    we can test the actual 80% target this time, not just 50%. If
    support_col is given, the holdout is ALSO evaluated after suppressing
    rows below MIN_SUPPORT on that column, reported as holdout_window_supportfiltered."""
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]

    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]

    sel_idx, sel_true, sel_score = rolling_predictions(select_df, feature_cols, label_col)

    order = np.argsort(-sel_score)
    sorted_true = sel_true[order]
    sorted_score = sel_score[order]
    chosen_threshold = None
    for k in range(10, len(sorted_true)):
        prec_at_k = sorted_true[:k].sum() / k
        if prec_at_k < target_precision:
            chosen_threshold = float(sorted_score[k - 1])
            break
    if chosen_threshold is None:
        chosen_threshold = float(sorted_score[-1]) if len(sorted_score) else 0.5

    train_X = select_df[feature_cols].fillna(0).values
    train_y = select_df[label_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    holdout_y = holdout_df[label_col].values
    holdout_score = fit_predict_ensemble(train_X, train_y, holdout_X)

    select_metrics = compute_metrics(sel_true, sel_score, threshold=chosen_threshold)
    holdout_metrics = compute_metrics(holdout_y, holdout_score, threshold=chosen_threshold)

    out = {
        "split_date": str(split_date.date()), "chosen_threshold": chosen_threshold,
        "target_precision_on_select": target_precision,
        "select_window": select_metrics, "holdout_window": holdout_metrics,
    }

    if support_col is not None:
        support_vals = holdout_df[support_col].fillna(0).values
        keep = support_vals >= MIN_SUPPORT
        if keep.sum() > 0 and holdout_y[keep].sum() > 0:
            out["holdout_window_supportfiltered"] = compute_metrics(
                holdout_y[keep], holdout_score[keep], threshold=chosen_threshold)
            out["support_filter_min_events"] = MIN_SUPPORT
            out["support_filter_rows_kept"] = int(keep.sum())
            out["support_filter_rows_total"] = int(len(keep))

    return out


if __name__ == "__main__":
    print("Loading round-3 candidate dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows, {df['issue_date'].nunique()} weekly issue dates, "
          f"{df['priogrid_gid'].nunique()} active cells", flush=True)

    out = {}

    print("\n\n########## 1. Feature-set sweep: momentum, wider spatial radius, combined ##########", flush=True)
    out["feature_sweep"] = run_feature_sweep(df)

    best_combined = FEATURE_SETS["plus_momentum_plus_ring2"]

    print("\n\n########## 2. Frozen-threshold @ 80% target: round-2 baseline (sanity check) ##########", flush=True)
    frozen_80 = {}
    for horizon in HORIZONS:
        print(f"\n--- {horizon} / round2_best / target=0.80 ---", flush=True)
        r = run_frozen_threshold_validation(df, ROUND2_BEST, f"label_{horizon}", target_precision=0.80)
        print(f"  SELECT:  precision={r['select_window']['precision']:.3f} recall={r['select_window']['recall']:.3f}", flush=True)
        print(f"  HOLDOUT: precision={r['holdout_window']['precision']:.3f} recall={r['holdout_window']['recall']:.3f} "
              f"n={r['holdout_window']['n']}", flush=True)
        frozen_80[horizon] = r
    out["frozen_80_baseline"] = frozen_80

    print("\n\n########## 3. Frozen-threshold @ 80% target: combined new features ##########", flush=True)
    frozen_80_combined = {}
    for horizon in HORIZONS:
        print(f"\n--- {horizon} / plus_momentum_plus_ring2 / target=0.80 ---", flush=True)
        r = run_frozen_threshold_validation(df, best_combined, f"label_{horizon}", target_precision=0.80,
                                             support_col="n_hist_events_total")
        print(f"  SELECT:  precision={r['select_window']['precision']:.3f} recall={r['select_window']['recall']:.3f}", flush=True)
        print(f"  HOLDOUT: precision={r['holdout_window']['precision']:.3f} recall={r['holdout_window']['recall']:.3f} "
              f"n={r['holdout_window']['n']}", flush=True)
        if "holdout_window_supportfiltered" in r:
            sf = r["holdout_window_supportfiltered"]
            print(f"  HOLDOUT (support>=~{MIN_SUPPORT} filtered, n={r['support_filter_rows_kept']}/{r['support_filter_rows_total']}): "
                  f"precision={sf['precision']:.3f} recall={sf['recall']:.3f}", flush=True)
        frozen_80_combined[horizon] = r
    out["frozen_80_combined_features"] = frozen_80_combined

    print("\n\n########## 4. Frozen-threshold @ 80% target: combined features + severity-tightened label ##########", flush=True)
    frozen_80_severe = {}
    for horizon in HORIZONS:
        label_col = f"label_{horizon}_severe"
        print(f"\n--- {horizon} / plus_momentum_plus_ring2 / {label_col} / target=0.80 ---", flush=True)
        r = run_frozen_threshold_validation(df, best_combined, label_col, target_precision=0.80,
                                             support_col="n_hist_events_total")
        print(f"  SELECT:  precision={r['select_window']['precision']:.3f} recall={r['select_window']['recall']:.3f}", flush=True)
        print(f"  HOLDOUT: precision={r['holdout_window']['precision']:.3f} recall={r['holdout_window']['recall']:.3f} "
              f"n={r['holdout_window']['n']}", flush=True)
        if "holdout_window_supportfiltered" in r:
            sf = r["holdout_window_supportfiltered"]
            print(f"  HOLDOUT (support-filtered, n={r['support_filter_rows_kept']}/{r['support_filter_rows_total']}): "
                  f"precision={sf['precision']:.3f} recall={sf['recall']:.3f}", flush=True)
        frozen_80_severe[horizon] = r
    out["frozen_80_severe_label"] = frozen_80_severe

    print("\n\n########## 5. Frozen-threshold @ 50% target (round-2 comparability check) ##########", flush=True)
    frozen_50_combined = {}
    for horizon in HORIZONS:
        r = run_frozen_threshold_validation(df, best_combined, f"label_{horizon}", target_precision=0.50,
                                             support_col="n_hist_events_total")
        frozen_50_combined[horizon] = r
        print(f"  {horizon}: SELECT precision={r['select_window']['precision']:.3f} -> "
              f"HOLDOUT precision={r['holdout_window']['precision']:.3f}", flush=True)
    out["frozen_50_combined_features"] = frozen_50_combined

    with open("../results/round3_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n\nSaved ../results/round3_results.json")
