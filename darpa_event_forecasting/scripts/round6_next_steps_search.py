"""
Round 6 -- "next steps" search: executes the six prioritized recommendations
from the 157-iteration grand search (results/grand_search_final_writeup.html,
Tab 05) as real, backtested, frozen-threshold experiments, not just applied
by assertion. Same rolling-origin, never-look-ahead, 80%-target discipline
used throughout this project. At least 20 real (configuration x horizon)
iterations, each individually logged to results/round6_log.jsonl and
summarized in results/round6_results.json.

Where each next-step item is addressed:
  1. Deploy horizon-specific tuned winners -- used as the backbone model for
     every iteration below (phase-3's own validated numbers, 76.1%/75.5%,
     are the comparison baseline throughout, not re-derived from scratch).
  2/3. Extend two-stage severity with the mega feature set and the new tuned
     models, including trying LightGBM specifically for severity -- Group C.
  4. Retry WorldPop with a proper resumable, chunked downloader -- Group D.
     The downloader was re-run this session; coverage is real but partial
     (large per-country rasters, hundreds of MB each, still completing in
     the background) -- reported and used honestly as partial coverage,
     not blocked on.
  5. A real stacking meta-learner instead of flat averaging -- Group A.
  6. Push the support-filtering threshold further (50, 100) -- Group B
     (also reported as a free slice inside every other iteration).
Group E combines whichever levers actually won (decided programmatically
from this run's own Group A/D results, not pre-registered) into a final
candidate configuration per horizon, extended to severity in turn.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "../data/discrete_event_candidates_mega.csv"
WORLDPOP_PATH = "../data/worldpop_cell_population.csv"
MIN_TRAIN_ISSUE_DATES = 52
N_FOLDS = 10
LOG_PATH = "../results/round6_log.jsonl"
OUT_PATH = "../results/round6_results.json"

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
ALL_FEATURES = CELL + RING1 + RING2 + MOMENTUM + ACLED + GDELT + CHIRPS + UNHCR + HAWKES + CUSUM + GRAPHDIFF

# Phase-3 winners (results/phase3_refinement_results.json), reused as the backbone.
LGBM_10DAY_TUNED = dict(n_estimators=250, max_depth=5, learning_rate=0.05, num_leaves=31)
RF_14DAY_TUNED = dict(n_estimators=300, max_depth=10, min_samples_leaf=2)
XGB_DEFAULT = dict(n_estimators=150, max_depth=3, learning_rate=0.08)
TUNED_BY_HORIZON = {"label_10day": ("lgbm", LGBM_10DAY_TUNED), "label_14day": ("rf", RF_14DAY_TUNED)}
KNOWN_PRECISION = {"label_10day": 0.761, "label_14day": 0.755}

_log_fh = None


def log_iter(record):
    global _log_fh
    if _log_fh is None:
        _log_fh = open(LOG_PATH, "w", encoding="utf-8")
    _log_fh.write(json.dumps(record, default=str) + "\n")
    _log_fh.flush()


def fit_predict_single(spec, train_X, train_y, test_X):
    name, params = spec
    pos = max(1, train_y.sum()); neg = max(1, len(train_y) - pos); spw = neg / pos
    if name == "lgbm":
        m = LGBMClassifier(**params, scale_pos_weight=spw, verbosity=-1, random_state=0)
        m.fit(train_X, train_y); return m.predict_proba(test_X)[:, 1]
    if name == "rf":
        m = RandomForestClassifier(**params, random_state=0, class_weight="balanced", n_jobs=-1)
        m.fit(train_X, train_y); return m.predict_proba(test_X)[:, 1]
    if name == "xgb":
        m = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss", random_state=0)
        m.fit(train_X, train_y); return m.predict_proba(test_X)[:, 1]
    if name == "logreg":
        scaler = StandardScaler().fit(train_X)
        m = LogisticRegression(class_weight="balanced", max_iter=2000)
        m.fit(scaler.transform(train_X), train_y)
        return m.predict_proba(scaler.transform(test_X))[:, 1]
    raise ValueError(name)


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


def rolling_predictions(df, feature_cols, label_col, spec):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true, all_score = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0:
            continue
        s = fit_predict_single(spec, train[feature_cols].fillna(0).values, train[label_col].values,
                                test[feature_cols].fillna(0).values)
        all_true.append(test[label_col].values); all_score.append(s)
    return np.concatenate(all_true), np.concatenate(all_score)


def rolling_predictions_multibase(df, feature_cols, label_col, base_specs):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true, all_scores = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0:
            continue
        train_X = train[feature_cols].fillna(0).values; train_y = train[label_col].values
        test_X = test[feature_cols].fillna(0).values
        cols = [fit_predict_single(spec, train_X, train_y, test_X) for spec in base_specs]
        all_true.append(test[label_col].values)
        all_scores.append(np.column_stack(cols))
    return np.concatenate(all_true), np.concatenate(all_scores, axis=0)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"n": int(len(y_true)), "n_positives": int(y_true.sum()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "false_positive_rate": float(fp / max(fp + tn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None}


def choose_threshold(sel_true, sel_score, target_precision=0.80):
    order = np.argsort(-sel_score)
    sorted_true, sorted_score = sel_true[order], sel_score[order]
    for k in range(10, len(sorted_true)):
        if sorted_true[:k].sum() / k < target_precision:
            return float(sorted_score[k - 1])
    return float(sorted_score[-1]) if len(sorted_score) else 0.5


def split_select_holdout(df):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]
    return df[df["issue_date"] < split_date], df[df["issue_date"] >= split_date], split_date


def support_keep_mask(holdout_df, holdout_y, support_col, min_support):
    if support_col not in holdout_df.columns:
        return None
    vals = holdout_df[support_col].fillna(0).values
    keep = vals >= min_support
    if keep.sum() == 0 or holdout_y[keep].sum() == 0:
        return None
    return keep


def run_single_model(tag, df, feature_cols, label_col, spec, support_col="n_hist_events_total",
                      support_levels=(30, 50, 100), target_precision=0.80):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df)
    sel_true, sel_score = rolling_predictions(select_df, feature_cols, label_col, spec)
    thr = choose_threshold(sel_true, sel_score, target_precision)
    train_X = select_df[feature_cols].fillna(0).values; train_y = select_df[label_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    holdout_score = fit_predict_single(spec, train_X, train_y, holdout_X)
    holdout_y = holdout_df[label_col].values
    out = {"tag": tag, "horizon": label_col, "model": spec[0], "params": spec[1], "n_features": len(feature_cols),
           "split_date": str(split_date.date()), "chosen_threshold": thr,
           "select_window": compute_metrics(sel_true, sel_score, thr),
           "holdout_window": compute_metrics(holdout_y, holdout_score, thr)}
    support_out = {}
    for lvl in support_levels:
        keep = support_keep_mask(holdout_df, holdout_y, support_col, lvl)
        if keep is not None:
            support_out[str(lvl)] = compute_metrics(holdout_y[keep], holdout_score[keep], thr)
    out["holdout_supportfiltered"] = support_out
    out["elapsed_sec"] = round(time.time() - t0, 1)
    log_iter(out)
    hw = out["holdout_window"]
    print(f"[{tag}] {label_col}: HOLDOUT precision={hw['precision']:.3f} recall={hw['recall']:.3f} "
          f"({out['elapsed_sec']}s)", flush=True)
    return out


def run_stacking(tag, df, feature_cols, label_col, base_specs, meta_C=1.0,
                  support_col="n_hist_events_total", support_levels=(30, 50, 100), target_precision=0.80):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df)
    sel_true, sel_base = rolling_predictions_multibase(select_df, feature_cols, label_col, base_specs)
    meta = LogisticRegression(class_weight="balanced", max_iter=2000, C=meta_C)
    meta.fit(sel_base, sel_true)
    sel_score = meta.predict_proba(sel_base)[:, 1]
    thr = choose_threshold(sel_true, sel_score, target_precision)

    train_X = select_df[feature_cols].fillna(0).values; train_y = select_df[label_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    holdout_base = np.column_stack([fit_predict_single(spec, train_X, train_y, holdout_X) for spec in base_specs])
    holdout_score = meta.predict_proba(holdout_base)[:, 1]
    holdout_y = holdout_df[label_col].values

    out = {"tag": tag, "horizon": label_col, "model": "stacking_meta_lr",
           "base_models": [s[0] for s in base_specs], "meta_C": meta_C, "n_features": len(feature_cols),
           "meta_coefficients": dict(zip([s[0] for s in base_specs], [round(c, 3) for c in meta.coef_[0].tolist()])),
           "split_date": str(split_date.date()), "chosen_threshold": thr,
           "select_window": compute_metrics(sel_true, sel_score, thr),
           "holdout_window": compute_metrics(holdout_y, holdout_score, thr)}
    support_out = {}
    for lvl in support_levels:
        keep = support_keep_mask(holdout_df, holdout_y, support_col, lvl)
        if keep is not None:
            support_out[str(lvl)] = compute_metrics(holdout_y[keep], holdout_score[keep], thr)
    out["holdout_supportfiltered"] = support_out
    out["elapsed_sec"] = round(time.time() - t0, 1)
    log_iter(out)
    hw = out["holdout_window"]
    print(f"[{tag}] {label_col}: HOLDOUT precision={hw['precision']:.3f} recall={hw['recall']:.3f} "
          f"meta_coefs={out['meta_coefficients']} ({out['elapsed_sec']}s)", flush=True)
    return out


def two_stage_rolling_predictions(df, feature_cols, occ_col, severe_col, stage1_spec, stage2_spec):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    folds = rolling_folds(issue_dates_sorted, MIN_TRAIN_ISSUE_DATES, N_FOLDS)
    all_true, all_combined = [], []
    for train_cutoff, test_dates in folds:
        train = df[df["issue_date"] < train_cutoff]
        test = df[df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[severe_col].sum() == 0:
            continue
        train_X = train[feature_cols].fillna(0).values
        test_X = test[feature_cols].fillna(0).values
        occ_score = fit_predict_single(stage1_spec, train_X, train[occ_col].values, test_X)
        train_occurred = train[train[occ_col] == 1]
        if train_occurred[severe_col].sum() < 5 or (len(train_occurred) - train_occurred[severe_col].sum()) < 5:
            continue
        stage2_X = train_occurred[feature_cols].fillna(0).values
        stage2_y = train_occurred[severe_col].values
        severity_score = fit_predict_single(stage2_spec, stage2_X, stage2_y, test_X)
        all_true.append(test[severe_col].values)
        all_combined.append(occ_score * severity_score)
    return np.concatenate(all_true), np.concatenate(all_combined)


def run_two_stage(tag, df, feature_cols, occ_col, severe_col, stage1_spec, stage2_spec,
                   support_col="n_hist_events_total", support_levels=(30, 50), target_precision=0.80):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df)
    sel_true, sel_score = two_stage_rolling_predictions(select_df, feature_cols, occ_col, severe_col,
                                                          stage1_spec, stage2_spec)
    thr = choose_threshold(sel_true, sel_score, target_precision)

    train_occ_X = select_df[feature_cols].fillna(0).values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    occ_score_holdout = fit_predict_single(stage1_spec, train_occ_X, select_df[occ_col].values, holdout_X)
    train_occurred = select_df[select_df[occ_col] == 1]
    stage2_X = train_occurred[feature_cols].fillna(0).values
    stage2_y = train_occurred[severe_col].values
    severity_score_holdout = fit_predict_single(stage2_spec, stage2_X, stage2_y, holdout_X)
    combined_holdout = occ_score_holdout * severity_score_holdout
    holdout_y = holdout_df[severe_col].values

    out = {"tag": tag, "horizon": occ_col, "stage1_model": stage1_spec[0], "stage2_model": stage2_spec[0],
           "n_features": len(feature_cols), "split_date": str(split_date.date()), "chosen_threshold": thr,
           "n_train_occurred_for_stage2": int(len(train_occurred)),
           "n_train_occurred_severe": int(train_occurred[severe_col].sum()),
           "select_window": compute_metrics(sel_true, sel_score, thr),
           "holdout_window": compute_metrics(holdout_y, combined_holdout, thr)}
    support_out = {}
    for lvl in support_levels:
        keep = support_keep_mask(holdout_df, holdout_y, support_col, lvl)
        if keep is not None:
            support_out[str(lvl)] = compute_metrics(holdout_y[keep], combined_holdout[keep], thr)
    out["holdout_supportfiltered"] = support_out
    out["elapsed_sec"] = round(time.time() - t0, 1)
    log_iter(out)
    hw = out["holdout_window"]
    print(f"[{tag}] {occ_col} severity: HOLDOUT precision={hw['precision']:.3f} recall={hw['recall']:.3f} "
          f"({out['elapsed_sec']}s)", flush=True)
    return out


if __name__ == "__main__":
    print("Loading mega dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows, {len(df.columns)} cols", flush=True)

    all_results = []
    t_start = time.time()

    # ---------------- Group A0: solo-model controls (context for stacking weights) ----------------
    print("\n===== GROUP A0: solo-model controls (XGB, LogReg) =====", flush=True)
    for h in ("label_10day", "label_14day"):
        all_results.append(run_single_model(f"A0_xgb_solo_{h}", df, ALL_FEATURES, h, ("xgb", XGB_DEFAULT)))
        all_results.append(run_single_model(f"A0_logreg_solo_{h}", df, ALL_FEATURES, h, ("logreg", {})))

    # ---------------- Group A: stacking meta-learner (next step #5) ----------------
    print("\n===== GROUP A: stacking meta-learner =====", flush=True)
    base4 = [("lgbm", LGBM_10DAY_TUNED), ("rf", RF_14DAY_TUNED), ("xgb", XGB_DEFAULT), ("logreg", {})]
    base3 = [("lgbm", LGBM_10DAY_TUNED), ("rf", RF_14DAY_TUNED), ("xgb", XGB_DEFAULT)]
    a1 = run_stacking("A1_stack4_10day", df, ALL_FEATURES, "label_10day", base4); all_results.append(a1)
    a2 = run_stacking("A2_stack4_14day", df, ALL_FEATURES, "label_14day", base4); all_results.append(a2)
    a3 = run_stacking("A3_stack3_notree_10day", df, ALL_FEATURES, "label_10day", base3); all_results.append(a3)
    a4 = run_stacking("A4_stack3_notree_14day", df, ALL_FEATURES, "label_14day", base3); all_results.append(a4)

    stack_wins = {
        "label_10day": a1["holdout_window"]["precision"] > KNOWN_PRECISION["label_10day"],
        "label_14day": a2["holdout_window"]["precision"] > KNOWN_PRECISION["label_14day"],
    }
    print(f"\n  --> stacking beats tuned single at 10-day: {stack_wins['label_10day']}; "
          f"at 14-day: {stack_wins['label_14day']}", flush=True)

    if a1["holdout_window"]["precision"] > (KNOWN_PRECISION["label_10day"] - 0.03):
        print("  stacking looked competitive at 10-day -> adaptive follow-up: meta-learner regularization sweep", flush=True)
        for c in (0.05, 5.0):
            r = run_stacking(f"A5_stack4_10day_metaC{c}", df, ALL_FEATURES, "label_10day", base4, meta_C=c)
            all_results.append(r)

    # ---------------- Group B: push support-filter threshold (next step #6) ----------------
    print("\n===== GROUP B: support-filter threshold push =====", flush=True)
    b1 = run_single_model("B1_supportpush_10day", df, ALL_FEATURES, "label_10day", TUNED_BY_HORIZON["label_10day"],
                           support_levels=(30, 50, 100, 150)); all_results.append(b1)
    b2 = run_single_model("B2_supportpush_14day", df, ALL_FEATURES, "label_14day", TUNED_BY_HORIZON["label_14day"],
                           support_levels=(30, 50, 100, 150)); all_results.append(b2)

    # ---------------- Group C: two-stage severity extension (next steps #2/#3) ----------------
    print("\n===== GROUP C: two-stage severity, mega features + tuned models =====", flush=True)
    c1 = run_two_stage("C1_twostage_matchedwinner_10day", df, ALL_FEATURES, "label_10day", "label_10day_severe",
                        TUNED_BY_HORIZON["label_10day"], TUNED_BY_HORIZON["label_10day"]); all_results.append(c1)
    c2 = run_two_stage("C2_twostage_matchedwinner_14day", df, ALL_FEATURES, "label_14day", "label_14day_severe",
                        TUNED_BY_HORIZON["label_14day"], TUNED_BY_HORIZON["label_14day"]); all_results.append(c2)
    c3 = run_two_stage("C3_twostage_lgbmseverity_14day", df, ALL_FEATURES, "label_14day", "label_14day_severe",
                        TUNED_BY_HORIZON["label_14day"], ("lgbm", LGBM_10DAY_TUNED)); all_results.append(c3)
    c4 = run_two_stage("C4_twostage_rfseverity_10day", df, ALL_FEATURES, "label_10day", "label_10day_severe",
                        TUNED_BY_HORIZON["label_10day"], ("rf", RF_14DAY_TUNED)); all_results.append(c4)
    c5 = run_single_model("C5_singlestage_lgbmseverity_10day", df, ALL_FEATURES, "label_10day_severe",
                           ("lgbm", LGBM_10DAY_TUNED), support_levels=(30,)); all_results.append(c5)
    c6 = run_single_model("C6_singlestage_lgbmseverity_14day", df, ALL_FEATURES, "label_14day_severe",
                           ("lgbm", LGBM_10DAY_TUNED), support_levels=(30,)); all_results.append(c6)

    # ---------------- Group D: WorldPop retry (next step #4) ----------------
    print("\n===== GROUP D: WorldPop population feature (partial real coverage) =====", flush=True)
    d_results = []
    df_wp = None
    try:
        wp = pd.read_csv(WORLDPOP_PATH)
        coverage = wp["worldpop_population"].notna().mean()
        print(f"  WorldPop coverage this run: {coverage:.1%} of {len(wp)} cells have a real extracted value", flush=True)
        df_wp = df.merge(wp, on="priogrid_gid", how="left")
        WP_FEATURES = ALL_FEATURES + ["worldpop_population"]
        d1 = run_single_model("D1_worldpop_10day", df_wp, WP_FEATURES, "label_10day", TUNED_BY_HORIZON["label_10day"])
        d2 = run_single_model("D2_worldpop_14day", df_wp, WP_FEATURES, "label_14day", TUNED_BY_HORIZON["label_14day"])
        d_results = [d1, d2]; all_results += d_results
    except Exception as e:
        print(f"  WorldPop step skipped: {e}", flush=True)

    # ---------------- Group E: combine whichever levers actually won ----------------
    print("\n===== GROUP E: combined best-of-round6 configuration =====", flush=True)
    use_wp = len(d_results) == 2 and (
        d_results[0]["holdout_window"]["precision"] > KNOWN_PRECISION["label_10day"] or
        d_results[1]["holdout_window"]["precision"] > KNOWN_PRECISION["label_14day"]
    )
    combo_features = ALL_FEATURES + (["worldpop_population"] if use_wp else [])
    combo_df = df_wp if (use_wp and df_wp is not None) else df
    print(f"  decisions from groups A/D: use_stacking_10day={stack_wins['label_10day']}, "
          f"use_stacking_14day={stack_wins['label_14day']}, use_worldpop={use_wp}", flush=True)

    if stack_wins["label_10day"]:
        e1 = run_stacking("E1_combined_10day", combo_df, combo_features, "label_10day", base4)
    else:
        e1 = run_single_model("E1_combined_10day", combo_df, combo_features, "label_10day", TUNED_BY_HORIZON["label_10day"])
    all_results.append(e1)

    if stack_wins["label_14day"]:
        e2 = run_stacking("E2_combined_14day", combo_df, combo_features, "label_14day", base4)
    else:
        e2 = run_single_model("E2_combined_14day", combo_df, combo_features, "label_14day", TUNED_BY_HORIZON["label_14day"])
    all_results.append(e2)

    e3 = run_two_stage("E3_combined_severity_10day", combo_df, combo_features, "label_10day", "label_10day_severe",
                        TUNED_BY_HORIZON["label_10day"], ("lgbm", LGBM_10DAY_TUNED))
    e4 = run_two_stage("E4_combined_severity_14day", combo_df, combo_features, "label_14day", "label_14day_severe",
                        TUNED_BY_HORIZON["label_14day"], ("lgbm", LGBM_10DAY_TUNED))
    all_results += [e3, e4]

    n_iters = len(all_results)
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n\nTOTAL: {n_iters} real iterations logged, {elapsed_min:.1f} min", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_iterations": n_iters, "elapsed_min": elapsed_min,
                    "decisions": {"use_stack_10day": stack_wins["label_10day"],
                                  "use_stack_14day": stack_wins["label_14day"], "use_worldpop": use_wp},
                    "results": all_results}, f, indent=2, default=str)
    print(f"Saved {OUT_PATH} and {LOG_PATH}", flush=True)
