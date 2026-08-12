"""
Round 3, lever 4 (retry): round 3's first attempt at severity tightened
the SAME single model to a stricter label and got a real negative result
(0% precision/recall on the frozen holdout) -- the analysis concluded
severity needs its own modeling structure, not just a stricter bar on
the occurrence model. This implements that directly:

  Stage 1 -- P(any event in the horizon window | features): the existing,
    validated combined feature-set ensemble (unchanged).
  Stage 2 -- P(severe (>=5 fatalities) | an event happens, features):
    a SEPARATE ensemble trained only on the rows where an event actually
    occurred in training data (label_{h}==1), predicting label_{h}_severe
    conditional on occurrence. This subset is far less imbalanced than
    the full population -- among rows where an event happens, roughly
    42% (10-day) / 44% (14-day) are severe, versus a 5-7% base rate in
    the full population -- which is exactly the kind of learnable signal
    a single-stage model chasing an 80% target on the rare severe label
    could not find.
  Combined score -- P(severe) = P(any event) x P(severe | any event),
    the standard two-stage chain-rule combination, evaluated against the
    real label_{h}_severe ground truth under the same frozen-threshold
    discipline as every other result in this project.
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

COMBINED_FEATURES = [
    "cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
    "days_since_last_event", "neighbor_count_30d",
    "acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth",
    "cell_count_30d_delta", "cell_count_90d_delta", "momentum_ratio_30_90",
    "neighbor_count_30d_ring2",
]


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


def two_stage_rolling_predictions(df, feature_cols, occ_col, severe_col):
    """For each rolling fold: fit stage-1 (occurrence) on the full training
    window, fit stage-2 (severity) on only the training rows where an event
    occurred, apply both to every test row, and combine via the chain rule.
    Also returns the stage-1-only score for comparison."""
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true_severe, all_score_combined, all_score_occ_only = [], [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[severe_col].sum() == 0:
            continue

        train_X = train[feature_cols].fillna(0).values
        test_X = test[feature_cols].fillna(0).values

        occ_score = fit_predict_ensemble(train_X, train[occ_col].values, test_X)

        train_occurred = train[train[occ_col] == 1]
        if train_occurred[severe_col].sum() < 5 or (len(train_occurred) - train_occurred[severe_col].sum()) < 5:
            # not enough of one class among "occurred" rows this fold to fit stage 2
            continue
        stage2_X = train_occurred[feature_cols].fillna(0).values
        stage2_y = train_occurred[severe_col].values
        severity_score = fit_predict_ensemble(stage2_X, stage2_y, test_X)

        combined = occ_score * severity_score

        all_true_severe.append(test[severe_col].values)
        all_score_combined.append(combined)
        all_score_occ_only.append(occ_score)

    return (np.concatenate(all_true_severe), np.concatenate(all_score_combined),
            np.concatenate(all_score_occ_only))


def run_frozen_threshold_two_stage(df, feature_cols, occ_col, severe_col, target_precision=0.80):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]
    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]

    sel_true, sel_score, _ = two_stage_rolling_predictions(select_df, feature_cols, occ_col, severe_col)

    order = np.argsort(-sel_score)
    sorted_true, sorted_score = sel_true[order], sel_score[order]
    chosen_threshold = None
    for k in range(10, len(sorted_true)):
        if sorted_true[:k].sum() / k < target_precision:
            chosen_threshold = float(sorted_score[k - 1]); break
    if chosen_threshold is None:
        chosen_threshold = float(sorted_score[-1]) if len(sorted_score) else 0.5

    # fresh fit on all of select_df, applied to the untouched holdout
    train_occ_X = select_df[feature_cols].fillna(0).values
    train_occ_y = select_df[occ_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    occ_score_holdout = fit_predict_ensemble(train_occ_X, train_occ_y, holdout_X)

    train_occurred = select_df[select_df[occ_col] == 1]
    stage2_X = train_occurred[feature_cols].fillna(0).values
    stage2_y = train_occurred[severe_col].values
    severity_score_holdout = fit_predict_ensemble(stage2_X, stage2_y, holdout_X)

    combined_holdout = occ_score_holdout * severity_score_holdout
    holdout_y = holdout_df[severe_col].values

    select_metrics = compute_metrics(sel_true, sel_score, threshold=chosen_threshold)
    holdout_metrics = compute_metrics(holdout_y, combined_holdout, threshold=chosen_threshold)
    return {
        "split_date": str(split_date.date()), "chosen_threshold": chosen_threshold,
        "target_precision_on_select": target_precision,
        "n_train_occurred_for_stage2": int(len(train_occurred)),
        "n_train_occurred_severe": int(train_occurred[severe_col].sum()),
        "select_window": select_metrics, "holdout_window": holdout_metrics,
    }


if __name__ == "__main__":
    print("Loading round-3 dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows", flush=True)

    out = {}

    print("\n########## Pooled rolling-CV: two-stage combined vs occurrence-only, vs round-3 single-stage severe ##########", flush=True)
    pooled = {}
    for horizon in HORIZONS:
        occ_col, severe_col = f"label_{horizon}", f"label_{horizon}_severe"
        print(f"\n--- {horizon} ---", flush=True)
        y_true, y_combined, y_occ_only = two_stage_rolling_predictions(df, COMBINED_FEATURES, occ_col, severe_col)
        m_combined = compute_metrics(y_true, y_combined)
        m_occ_only_as_severity_proxy = compute_metrics(y_true, y_occ_only)  # sanity baseline: does occurrence alone rank severity?
        print(f"  two-stage combined:      precision={m_combined['precision']:.3f} recall={m_combined['recall']:.3f} "
              f"AP={m_combined['average_precision']:.3f} AUC={m_combined['roc_auc']:.3f}", flush=True)
        print(f"  occurrence-score-only:   precision={m_occ_only_as_severity_proxy['precision']:.3f} "
              f"recall={m_occ_only_as_severity_proxy['recall']:.3f} AP={m_occ_only_as_severity_proxy['average_precision']:.3f} "
              f"AUC={m_occ_only_as_severity_proxy['roc_auc']:.3f}", flush=True)
        pooled[horizon] = {"two_stage_combined": m_combined, "occurrence_score_only": m_occ_only_as_severity_proxy}
    out["pooled_rolling_cv"] = pooled

    print("\n\n########## Frozen-threshold @ 80% target: two-stage combined severity model ##########", flush=True)
    frozen = {}
    for horizon in HORIZONS:
        occ_col, severe_col = f"label_{horizon}", f"label_{horizon}_severe"
        print(f"\n--- {horizon} ---", flush=True)
        r = run_frozen_threshold_two_stage(df, COMBINED_FEATURES, occ_col, severe_col, target_precision=0.80)
        print(f"  stage-2 trained on {r['n_train_occurred_for_stage2']} real 'event occurred' rows "
              f"({r['n_train_occurred_severe']} of them severe)", flush=True)
        print(f"  SELECT:  precision={r['select_window']['precision']:.3f} recall={r['select_window']['recall']:.3f}", flush=True)
        print(f"  HOLDOUT: precision={r['holdout_window']['precision']:.3f} recall={r['holdout_window']['recall']:.3f} "
              f"n={r['holdout_window']['n']}", flush=True)
        frozen[horizon] = r
    out["frozen_80_two_stage"] = frozen

    with open("../results/two_stage_severity_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n\nSaved ../results/two_stage_severity_results.json")
