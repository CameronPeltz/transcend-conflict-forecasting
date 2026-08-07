"""
DARPA proposal Criterion 2: "Forecast precision >=80% using independently
validated metrics." This script exists to test that claim honestly rather
than assert it.

The methodological trap: if you pick the probability threshold that
maximizes precision on the SAME pooled test predictions you then report
precision on, you are optimizing on your own scoreboard -- a form of
data snooping. The number would be real in the sense that it's computed
from real predictions, but it would not be a trustworthy estimate of
what precision to expect on the NEXT unseen week.

The honest fix used here, consistent with this project's bitemporal
discipline throughout: split the rolling-origin folds themselves in
time into an EARLY threshold-selection window and a LATER, strictly
held-out reporting window. The threshold is chosen using only the early
window (find the lowest threshold that hits the target precision there),
then frozen and applied, unchanged, to the later window. Only the later
window's resulting precision/recall/etc. are reported as the real
result. This is genuinely out-of-sample with respect to the threshold
choice, not just with respect to model fitting -- the strongest claim
this project can honestly make without an actual external/third-party
audit (which it does not have, and does not claim to have).

Runs on Track C (UCDP, the pure fatality-coded track) and Track B (large
scraped GDELT), the two tracks with real, non-trivial pooled sample
sizes. Track A is excluded here -- too few folds to split further.
"""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import build_ucdp_panel as up
import large_panel as lp


def fit_predict_ensemble(train, test, feature_cols, label_col, cat_cols=("country",)):
    train_X = train[feature_cols].fillna(0).copy()
    test_X = test[feature_cols].fillna(0).copy()
    cat_cols = [c for c in cat_cols if c in train.columns]
    if cat_cols:
        tr_cat = pd.get_dummies(train[cat_cols].astype(str))
        te_cat = pd.get_dummies(test[cat_cols].astype(str)).reindex(columns=tr_cat.columns, fill_value=0)
        train_X = pd.concat([train_X.reset_index(drop=True), tr_cat.reset_index(drop=True)], axis=1)
        test_X = pd.concat([test_X.reset_index(drop=True), te_cat.reset_index(drop=True)], axis=1)

    y_train = train[label_col].fillna(0).astype(int)
    pos = max(1, y_train.sum()); neg = max(1, len(y_train) - pos)

    gbm = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                 random_state=0, class_weight="balanced")
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    test_Xs = scaler.transform(test_X)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)

    gbm.fit(train_X, y_train)
    rf.fit(train_X, y_train)
    logreg.fit(train_Xs, y_train)

    p_gbm = gbm.predict_proba(test_X)[:, 1]
    p_rf = rf.predict_proba(test_X)[:, 1]
    p_logreg = logreg.predict_proba(test_Xs)[:, 1]
    return (p_gbm + p_rf + p_logreg) / 3.0


def collect_predictions_track_c(min_train=8):
    panel, _ = up.build_panel()
    feature_cols = up.UCDP_FEATURE_SET
    label_col = "label_1"
    weeks = sorted(panel["week"].unique())
    weeks = [w for w in weeks if w >= pd.Timestamp("2021-08-01")]

    rows = []
    for cutoff in weeks:
        train = panel[(panel["week"] < cutoff)].dropna(subset=[label_col])
        test = panel[panel["week"] == cutoff].dropna(subset=[label_col])
        if len(train) < 50 or len(test) == 0:
            continue
        probs = fit_predict_ensemble(train, test, feature_cols, label_col)
        for (_, r), p in zip(test.iterrows(), probs):
            rows.append({"week": cutoff, "country": r["country"], "prob": float(p), "actual": int(r[label_col])})
    return pd.DataFrame(rows)


def collect_predictions_track_b(min_train=8):
    raw = lp.load_raw()
    panel = lp.build_panel(raw_df=raw)
    feature_cols = lp.FEATURE_SETS["core"]
    label_col = "label_quad_1"
    folds = lp.rolling_origin_folds(panel, label_col, min_train)

    rows = []
    for cutoff, train, test in folds:
        test_v = test.dropna(subset=[label_col])
        if len(test_v) == 0:
            continue
        probs = fit_predict_ensemble(train, test_v, feature_cols, label_col, cat_cols=("country",))
        for (_, r), p in zip(test_v.iterrows(), probs):
            rows.append({"week": cutoff, "country": r["country"], "prob": float(p), "actual": int(r[label_col])})
    return pd.DataFrame(rows)


def metrics_at_threshold(df, t):
    pred = (df["prob"] >= t).astype(int)
    y = df["actual"].values
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, len(y))
    coverage = (tp + fp) / max(1, len(y))  # fraction of all weeks the system actually alerts on
    return dict(threshold=float(t), n=len(y), n_pos=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, specificity=specificity, accuracy=accuracy,
                coverage=coverage)


def find_threshold_for_target_precision(df_select, target=0.80, min_flags=5):
    """Scan candidate thresholds (every real probability value produced),
    return the LOWEST threshold that reaches the target precision on the
    selection set with at least min_flags positive flags (so we're not
    just picking the one super-confident lucky prediction)."""
    candidates = sorted(df_select["prob"].unique())
    best = None
    for t in candidates:
        m = metrics_at_threshold(df_select, t)
        if m["tp"] + m["fp"] < min_flags:
            continue
        if m["precision"] >= target:
            if best is None or t < best["threshold"]:
                best = m
    return best


def run_track(name, df, split_frac=0.6, target=0.80):
    weeks = sorted(df["week"].unique())
    split_idx = int(len(weeks) * split_frac)
    split_week = weeks[split_idx]
    select_df = df[df["week"] < split_week]
    holdout_df = df[df["week"] >= split_week]

    result = {
        "track": name,
        "n_weeks_total": len(weeks),
        "split_week": str(split_week),
        "select_n": len(select_df), "select_n_pos": int(select_df["actual"].sum()),
        "holdout_n": len(holdout_df), "holdout_n_pos": int(holdout_df["actual"].sum()),
    }

    chosen = find_threshold_for_target_precision(select_df, target=target)
    result["threshold_chosen_on_select_set"] = chosen

    if chosen is not None:
        held_out_result = metrics_at_threshold(holdout_df, chosen["threshold"])
        result["held_out_result_at_that_threshold"] = held_out_result
    else:
        result["held_out_result_at_that_threshold"] = None

    # also report the full precision/coverage curve on the HOLD-OUT set only,
    # for transparency (what precision is achievable at various coverage
    # levels, computed honestly out-of-sample)
    curve = []
    for t in np.linspace(0.05, 0.95, 19):
        curve.append(metrics_at_threshold(holdout_df, t))
    result["holdout_full_curve"] = curve

    # default-threshold (0.5) result on the SAME held-out set, for comparison
    result["holdout_at_0.5"] = metrics_at_threshold(holdout_df, 0.5)

    return result


def main():
    print("Collecting real out-of-sample predictions, Track C (UCDP)...")
    df_c = collect_predictions_track_c()
    df_c.to_csv("results_v2/precision_validation_track_c_predictions.csv", index=False)
    print(f"  {len(df_c)} real predictions, {df_c['actual'].sum()} positive")

    print("Collecting real out-of-sample predictions, Track B (large GDELT)...")
    df_b = collect_predictions_track_b()
    df_b.to_csv("results_v2/precision_validation_track_b_predictions.csv", index=False)
    print(f"  {len(df_b)} real predictions, {df_b['actual'].sum()} positive")

    out = {}
    out["C_pure_ucdp"] = run_track("C_pure_ucdp", df_c, split_frac=0.6, target=0.80)
    out["B_large_scraped_gdelt"] = run_track("B_large_scraped_gdelt", df_b, split_frac=0.6, target=0.80)

    with open("results_v2/precision_threshold_validation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    for track, r in out.items():
        print(f"\n=== {track} ===")
        print(f"select set: n={r['select_n']} pos={r['select_n_pos']} | holdout set: n={r['holdout_n']} pos={r['holdout_n_pos']}")
        if r["threshold_chosen_on_select_set"]:
            t = r["threshold_chosen_on_select_set"]["threshold"]
            print(f"threshold chosen on SELECT set to hit 80% precision: {t:.3f}")
            h = r["held_out_result_at_that_threshold"]
            print(f"  applied unchanged to HELD-OUT set: precision={h['precision']:.3f} recall={h['recall']:.3f} "
                  f"coverage={h['coverage']:.3f} n_flagged={h['tp']+h['fp']} (tp={h['tp']} fp={h['fp']})")
        else:
            print("  NO threshold on the select set reached 80% precision with >=5 flags.")
        d5 = r["holdout_at_0.5"]
        print(f"  for comparison, default 0.5 threshold on holdout: precision={d5['precision']:.3f} recall={d5['recall']:.3f}")

    print("\nDONE. Wrote results_v2/precision_threshold_validation.json")


if __name__ == "__main__":
    main()
