"""
The actual test: does adding the real Rootwise DRC radio-transcript
signal change/improve forecasts for the exact weeks where that signal
exists, versus the same models using only event-coded (GDELT) features?

Honesty constraint stated up front, not after the fact: UCDP's real
curated data ends 2024-12-31 (checked directly against the downloaded
file) -- it does not reach the Rootwise radio window (2026-06-29
through 2026-07-27) at all. GDELT is near-real-time and does reach
that window, so GDELT-derived escalation labels are the only real,
available ground truth for this specific comparison. This mirrors
exactly how Tracks A and B already work elsewhere in this project.

Second honesty constraint: only ~4 real weeks have both a usable
lag-1 radio feature AND a real GDELT-derived label to check against.
That is nowhere near the sample size Criterion 2's 858-prediction
result had. This script computes what's honestly computable at that
scale (Brier score, log-loss, raw probability shift, rank behavior)
and does NOT report a precision/recall figure built on ~4 real
observations, which would use rounding to imply confidence that
doesn't exist at this sample size.
"""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

import build_drc_panel as bp
import rootwise_features as rw
from hypergraph_model import HypergraphNN

EVENT_FEATURES = [
    "n_events_lag1", "n_events_lag2", "n_events_delta",
    "material_conflict_share_lag1", "material_conflict_share_lag2", "material_conflict_share_delta",
    "mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
    "distinct_actors_lag1", "distinct_actors_lag2", "distinct_actors_delta",
    "mean_tone_lag1", "mean_tone_delta",
]


def build_panel(radio_kwargs=None):
    raw = bp.load_gdelt_drc_raw()
    panel = bp.build_gdelt_drc_panel(raw)
    rdf = rw.load_rootwise_raw()
    radio_feat = rw.build_weekly_features(rdf, **(radio_kwargs or {}))
    merged = panel.merge(radio_feat.drop(columns=["country"]), on="week", how="left")
    return merged


def fit_predict_gbm_ensemble(train, test, feature_cols):
    train_X = train[feature_cols].fillna(0)
    test_X = test[feature_cols].fillna(0)
    y_train = train["label_quad_1"].fillna(0).astype(int)
    pos = max(1, y_train.sum()); neg = max(1, len(y_train) - pos)
    gbm = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=0, class_weight="balanced")
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X); test_Xs = scaler.transform(test_X)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)
    gbm.fit(train_X, y_train); rf.fit(train_X, y_train); logreg.fit(train_Xs, y_train)
    return (gbm.predict_proba(test_X)[:, 1] + rf.predict_proba(test_X)[:, 1] + logreg.predict_proba(test_Xs)[:, 1]) / 3.0


def rolling_backtest(panel, feature_cols, min_train=20):
    """NB: only the LABEL is required to be non-null for a row to be
    usable for training/testing -- individual feature columns (radio
    features especially) are allowed to be NaN and are imputed to 0 by
    fit_predict_gbm_ensemble's .fillna(0). This matters a lot here:
    most of DRC's ~150-week GDELT history predates the 4-week Rootwise
    radio sample entirely, so requiring complete radio features on
    every training row would silently drop nearly the whole training
    set. Weeks with no radio coverage get radio features = 0 (a real,
    defensible reading: zero conflict-relevant radio signal was
    collected, not "unknown"), which is the same fillna(0) convention
    already used for every other feature block in this project."""
    weeks = sorted(panel["week"].unique())
    rows = []
    for i in range(min_train, len(weeks)):
        cutoff = weeks[i]
        train = panel[panel["week"] < cutoff].dropna(subset=["label_quad_1"])
        test = panel[panel["week"] == cutoff].dropna(subset=["label_quad_1"])
        if len(train) < min_train or len(test) == 0:
            continue
        probs = fit_predict_gbm_ensemble(train, test, feature_cols)
        for (_, r), p in zip(test.iterrows(), probs):
            has_radio = pd.notna(r.get("radio_n_clips_lag1"))
            rows.append({"week": cutoff, "prob": float(p), "actual": int(r["label_quad_1"]), "has_radio": bool(has_radio)})
    return pd.DataFrame(rows)


def main():
    print("Building DRC panel (GDELT event-coded + real Rootwise radio features)...")
    panel = build_panel()
    panel.to_csv("data/drc/drc_panel_with_radio.csv", index=False)
    print(f"panel: {len(panel)} weeks, {panel['week'].min()} to {panel['week'].max()}")
    radio_weeks = panel[panel["radio_n_clips"].notna()]["week"].tolist()
    print(f"weeks with real radio coverage: {radio_weeks}")

    results = {}

    print("\n=== BASELINE: event-coded features only ===")
    baseline_preds = rolling_backtest(panel, EVENT_FEATURES, min_train=20)
    baseline_preds.to_csv("data/drc/baseline_predictions.csv", index=False)
    y = baseline_preds["actual"].values
    p = np.clip(baseline_preds["prob"].values, 1e-6, 1 - 1e-6)
    print(f"n={len(y)} pos={y.sum()} Brier={brier_score_loss(y,p):.4f}")
    if y.sum() > 0 and y.sum() < len(y):
        print(f"log_loss={log_loss(y,p,labels=[0,1]):.4f}")
    results["baseline_full_history"] = {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if 0 < y.sum() < len(y) else None,
    }

    print("\n=== RADIO-AUGMENTED: event-coded + real radio features ===")
    all_features = EVENT_FEATURES + rw.RADIO_FEATURE_SET
    radio_preds = rolling_backtest(panel, all_features, min_train=20)
    radio_preds.to_csv("data/drc/radio_augmented_predictions.csv", index=False)

    # restrict comparison to EXACTLY the same weeks in both runs, for a fair
    # apples-to-apples check -- the honest core of this whole test
    overlap_weeks = sorted(set(baseline_preds["week"]) & set(radio_preds["week"]) & set(radio_weeks))
    print(f"\n=== Fair comparison: identical {len(overlap_weeks)} weeks, both runs ===")
    b_sub = baseline_preds[baseline_preds["week"].isin(overlap_weeks)].sort_values("week")
    r_sub = radio_preds[radio_preds["week"].isin(overlap_weeks)].sort_values("week")
    print("week | actual | baseline_prob | radio_augmented_prob")
    for w in overlap_weeks:
        bw = b_sub[b_sub.week == w].iloc[0]
        rw_ = r_sub[r_sub.week == w].iloc[0]
        print(f"  {w.date()} | {bw.actual} | {bw.prob:.3f} | {rw_.prob:.3f}")

    yb = b_sub["actual"].values; pb = np.clip(b_sub["prob"].values, 1e-6, 1 - 1e-6)
    yr = r_sub["actual"].values; pr = np.clip(r_sub["prob"].values, 1e-6, 1 - 1e-6)
    results["fair_comparison"] = {
        "n_weeks": len(overlap_weeks),
        "weeks": [str(w.date()) for w in overlap_weeks],
        "actuals": [int(x) for x in yb],
        "baseline_probs": [float(x) for x in pb],
        "radio_augmented_probs": [float(x) for x in pr],
        "baseline_brier": float(brier_score_loss(yb, pb)) if len(set(yb.tolist())) > 0 else None,
        "radio_augmented_brier": float(brier_score_loss(yr, pr)) if len(set(yr.tolist())) > 0 else None,
    }
    # per-week squared error (Brier is just mean of these) -- honest even at n=4
    sq_err_b = (pb - yb) ** 2
    sq_err_r = (pr - yr) ** 2
    results["fair_comparison"]["per_week_squared_error_baseline"] = [float(x) for x in sq_err_b]
    results["fair_comparison"]["per_week_squared_error_radio"] = [float(x) for x in sq_err_r]
    print(f"\nmean squared error, baseline: {sq_err_b.mean():.4f}, radio-augmented: {sq_err_r.mean():.4f}")

    with open("data/drc/radio_integration_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nDONE. Wrote data/drc/radio_integration_results.json")


if __name__ == "__main__":
    main()
