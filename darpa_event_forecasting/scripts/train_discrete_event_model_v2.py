"""
Round 2: implements the remaining recommendations from
event_forecasting_writeup.html tab 05 that weren't yet done:

  1. A 5th feature set adding the real ACLED cross-source signal
     (acled_civ_targeting_events_prevmonth / _fatalities_prevmonth)
     on top of round 1's best config (cell + spatial).
  2. Frozen-threshold temporal holdout validation, mirroring Criterion
     2's own discipline exactly: an early threshold-selection window
     and a strictly later, untouched holdout window; the threshold is
     chosen only on the early window (lowest threshold reaching a
     target precision there), then frozen and applied unchanged to the
     holdout, with only the holdout result reported as the finding.
  3. Per-country breakdown of the pooled metrics, given Afghanistan's
     disclosed disproportionate share of the data.
  4. Generalization to unseen EVENT TYPES (not just unseen regions,
     which Criterion 4's leave-one-country-out already covers): the
     model is trained on candidates whose future outcome (if positive)
     is state-based or non-state conflict, with one-sided violence
     against civilians held out entirely from training labels, then
     tested on whether it still ranks real one-sided-violence
     candidates above real negatives.
  5. Real, checked case studies pulled directly from the holdout
     predictions.

Uses the round-2 dataset (UCDP v26.1 + ACLED cross-source feature,
extended through late 2025) built by build_discrete_event_dataset_v2.py.
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

DATA_PATH = "../data/discrete_event_candidates_v2.csv"

BEST_ROUND1_FEATURES = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
                         "days_since_last_event", "neighbor_count_30d"]
FEATURE_SETS = {
    "cell_plus_spatial": BEST_ROUND1_FEATURES,
    "cell_plus_spatial_plus_acled": BEST_ROUND1_FEATURES + [
        "acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth"],
}
HORIZONS = ["10day", "14day"]
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
FP_THRESHOLD = 0.5


def fit_predict_ensemble(train_X, train_y, test_X, return_models=False):
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
    scores = (p_gbm + p_rf + p_logreg) / 3.0
    if return_models:
        return scores, (gbm, rf, logreg, scaler)
    return scores


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


# ---------------------------------------------------------------- 1: feature-set sweep
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


# ---------------------------------------------------------------- 2: frozen-threshold holdout
def run_frozen_threshold_validation(df, feature_cols, label_col, target_precision=0.5):
    """Same discipline as scripts/precision_threshold_validation.py used for
    Criterion 2: split issue dates into an early selection window and a
    strictly later holdout window; choose the lowest threshold reaching
    target_precision on the SELECT window only; freeze it; apply unchanged
    to the untouched holdout window; report only the holdout result."""
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]

    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]

    # train on everything before the select/holdout split's own internal rolling
    # folds within select_df, to get real out-of-fold scores on select_df itself
    sel_idx, sel_true, sel_score = rolling_predictions(select_df, feature_cols, label_col)

    # threshold selection: lowest threshold reaching target precision on select_df
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

    # now train fresh on ALL of select_df, apply frozen threshold to holdout_df
    train_X = select_df[feature_cols].fillna(0).values
    train_y = select_df[label_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    holdout_y = holdout_df[label_col].values
    holdout_score = fit_predict_ensemble(train_X, train_y, holdout_X)

    select_metrics = compute_metrics(sel_true, sel_score, threshold=chosen_threshold)
    holdout_metrics = compute_metrics(holdout_y, holdout_score, threshold=chosen_threshold)
    return {
        "split_date": str(split_date.date()), "chosen_threshold": chosen_threshold,
        "target_precision_on_select": target_precision,
        "select_window": select_metrics, "holdout_window": holdout_metrics,
        "holdout_df_index": holdout_df.index.values, "holdout_score": holdout_score, "holdout_y": holdout_y,
    }


# ---------------------------------------------------------------- 3: per-country breakdown
def run_country_breakdown(df, feature_cols, label_col):
    idx, y_true, y_score = rolling_predictions(df, feature_cols, label_col)
    countries = df.loc[idx, "country"].values
    out = []
    for c in sorted(set(countries)):
        mask = countries == c
        if mask.sum() < 30 or y_true[mask].sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_score[mask])
        m["country"] = c
        out.append(m)
    return sorted(out, key=lambda r: -r["n"])


# ---------------------------------------------------------------- 4: event-type generalization
def run_event_type_generalization(df, feature_cols, horizon):
    """Train using only candidates whose real future positive outcome (if
    any) is state-based (type 1) or non-state (type 2) conflict -- one-
    sided violence against civilians (type 3) positives are relabeled to
    0 in TRAINING ONLY, so the model never sees "one-sided violence" as a
    positive example. Test: on the real holdout weeks, does the model
    still rank real one-sided-violence candidates above real negatives,
    despite never being trained to recognize that category?"""
    type_col = f"label_{horizon}_type"
    label_col = f"label_{horizon}"
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.7)
    split_date = issue_dates_sorted[split_idx]

    train = df[df["issue_date"] < split_date].copy()
    test = df[df["issue_date"] >= split_date].copy()

    train["label_no_onesided"] = train[label_col].where(train[type_col] != 3, 0)

    train_X = train[feature_cols].fillna(0).values
    train_y = train["label_no_onesided"].values
    test_X = test[feature_cols].fillna(0).values

    scores = fit_predict_ensemble(train_X, train_y, test_X)

    is_onesided_positive = (test[type_col] == 3).values
    is_negative = (test[label_col] == 0).values
    onesided_scores = scores[is_onesided_positive]
    negative_scores = scores[is_negative]

    if len(onesided_scores) == 0 or len(negative_scores) == 0:
        return None

    # AUC-style: probability a real held-out-type positive scores above a random real negative
    combined_y = np.concatenate([np.ones(len(onesided_scores)), np.zeros(len(negative_scores))])
    combined_s = np.concatenate([onesided_scores, negative_scores])
    auc_unseen_type = float(roc_auc_score(combined_y, combined_s))

    return {
        "horizon": horizon, "split_date": str(split_date.date()),
        "n_train": int(len(train)), "n_onesided_positives_hidden_from_training": int(train[type_col].eq(3).sum()),
        "n_test_onesided_positives": int(is_onesided_positive.sum()),
        "n_test_negatives": int(is_negative.sum()),
        "mean_score_onesided_positive": float(onesided_scores.mean()),
        "mean_score_negative": float(negative_scores.mean()),
        "auc_ranking_unseen_type_above_negatives": auc_unseen_type,
    }


# ---------------------------------------------------------------- 5: real case studies
def build_case_studies(df, holdout_index, holdout_score, holdout_y, name_lookup, n=8):
    sub = df.loc[holdout_index].copy()
    sub["score"] = holdout_score
    sub["actual"] = holdout_y
    hits = sub[(sub["actual"] == 1) & (sub["score"] >= 0.5)].sort_values("score", ascending=False)
    misses = sub[(sub["actual"] == 1) & (sub["score"] < 0.5)].sort_values("score")
    false_alarms = sub[(sub["actual"] == 0) & (sub["score"] >= 0.7)].sort_values("score", ascending=False)

    def to_records(frame, k):
        out = []
        for _, r in frame.head(k).iterrows():
            out.append({
                "issue_date": str(r["issue_date"].date()) if hasattr(r["issue_date"], "date") else str(r["issue_date"]),
                "priogrid_gid": int(r["priogrid_gid"]), "country": r["country"],
                "score": round(float(r["score"]), 3),
                "cell_count_90d": int(r["cell_count_90d"]), "neighbor_count_30d": int(r["neighbor_count_30d"]),
                "days_since_last_event": int(r["days_since_last_event"]),
            })
        return out

    return {"real_hits": to_records(hits, n), "real_misses": to_records(misses, n),
            "real_false_alarms": to_records(false_alarms, n)}


if __name__ == "__main__":
    print("Loading round-2 discrete-event candidate dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows, {df['issue_date'].nunique()} weekly issue dates, "
          f"{df['priogrid_gid'].nunique()} active cells", flush=True)

    out = {}

    print("\n\n########## 1. Feature-set sweep (adds real ACLED cross-source signal) ##########", flush=True)
    out["feature_sweep"] = run_feature_sweep(df)

    print("\n\n########## 2. Frozen-threshold holdout validation (mirrors Criterion 2) ##########", flush=True)
    frozen = {}
    case_studies = {}
    for horizon in HORIZONS:
        print(f"\n--- {horizon} ---", flush=True)
        r = run_frozen_threshold_validation(df, FEATURE_SETS["cell_plus_spatial_plus_acled"], f"label_{horizon}")
        print(f"  split={r['split_date']} threshold={r['chosen_threshold']:.3f}", flush=True)
        print(f"  SELECT:  precision={r['select_window']['precision']:.3f} recall={r['select_window']['recall']:.3f}", flush=True)
        print(f"  HOLDOUT: precision={r['holdout_window']['precision']:.3f} recall={r['holdout_window']['recall']:.3f} "
              f"FP_rate={r['holdout_window']['false_positive_rate']:.3f} n={r['holdout_window']['n']}", flush=True)
        cs = build_case_studies(df, r["holdout_df_index"], r["holdout_score"], r["holdout_y"], None)
        case_studies[horizon] = cs
        frozen[horizon] = {k: v for k, v in r.items() if k not in ("holdout_df_index", "holdout_score", "holdout_y")}
    out["frozen_threshold_validation"] = frozen
    out["case_studies"] = case_studies

    print("\n\n########## 3. Per-country breakdown ##########", flush=True)
    country_breakdown = {}
    for horizon in HORIZONS:
        cb = run_country_breakdown(df, FEATURE_SETS["cell_plus_spatial_plus_acled"], f"label_{horizon}")
        country_breakdown[horizon] = cb
        print(f"\n--- {horizon} ---", flush=True)
        for r in cb:
            print(f"  {r['country']:15s} n={r['n']:6d} precision={r['precision']:.3f} "
                  f"recall={r['recall']:.3f} AP={r['average_precision']:.3f}", flush=True)
    out["country_breakdown"] = country_breakdown

    print("\n\n########## 4. Event-type generalization (unseen: one-sided violence) ##########", flush=True)
    event_type_gen = {}
    for horizon in HORIZONS:
        r = run_event_type_generalization(df, FEATURE_SETS["cell_plus_spatial_plus_acled"], horizon)
        event_type_gen[horizon] = r
        print(f"\n--- {horizon} ---", flush=True)
        print(json.dumps(r, indent=2))
    out["event_type_generalization"] = event_type_gen

    with open("../results/round2_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n\nSaved ../results/round2_results.json")
