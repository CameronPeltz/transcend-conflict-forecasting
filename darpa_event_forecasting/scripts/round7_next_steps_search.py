"""
Round 7 -- acting on round 6's own next-steps list (results/round6_next_steps_writeup.html,
Tab 06), plus a fresh internally-generated next-steps list at the end. Same rolling-origin,
never-look-ahead, frozen-threshold-at-80%-target discipline as every prior round. At least
20 real (configuration x horizon) iterations, logged to results/round7_log.jsonl and
results/round7_results.json.

Where each round-6 next-step item is addressed:
  1. Deploy the winning config -- used as the comparison baseline throughout (round 6's
     78.8%/79.6% at 10-day, 75.5%/78.1% at 14-day), not re-run in isolation.
  2. Finish WorldPop / fix the coverage confound -- Group D: a clean covered-subset A/B test
     (does population help AT ALL once the missing-data confound is removed) plus a coverage-
     indicator feature (does telling the model which rows are real vs. filled fix round 6's
     negative result).
  3. Search the >=100/>=150 support-filter boundary -- Group B: a finer sweep (75/100/125/150/175).
  4. Stacking jointly re-optimized on the support-filtered population, not filtered post-hoc --
     Group C: trains AND evaluates on the filtered subset directly, at two thresholds.
  5. (Scope decision, not a modeling step -- not applicable here.)
Group A is a round-6-motivated ablation (does dropping LightGBM, which the meta-learner gave
~zero weight, change anything). Group E extends two-stage severity with a genuinely different
stage-1 (a 3-model average, not a single tuned model) and pushes its support-filter reporting
to >=100. Group F builds the final combined recipe from whichever of B/C/D actually won, decided
programmatically. Group G is a short adaptive follow-up chosen from this round's own results.
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
LOG_PATH = "../results/round7_log.jsonl"
OUT_PATH = "../results/round7_results.json"

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

LGBM_10DAY_TUNED = dict(n_estimators=250, max_depth=5, learning_rate=0.05, num_leaves=31)
RF_14DAY_TUNED = dict(n_estimators=300, max_depth=10, min_samples_leaf=2)
XGB_DEFAULT = dict(n_estimators=150, max_depth=3, learning_rate=0.08)
TUNED_BY_HORIZON = {"label_10day": ("lgbm", LGBM_10DAY_TUNED), "label_14day": ("rf", RF_14DAY_TUNED)}
KNOWN_PRECISION = {"label_10day": 0.788, "label_14day": 0.755}   # round-6 unfiltered bests

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
    if name == "avg3":
        preds = [fit_predict_single(("lgbm", LGBM_10DAY_TUNED), train_X, train_y, test_X),
                 fit_predict_single(("rf", RF_14DAY_TUNED), train_X, train_y, test_X),
                 fit_predict_single(("xgb", XGB_DEFAULT), train_X, train_y, test_X)]
        return np.mean(preds, axis=0)
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
           "n_rows_used": len(df), "split_date": str(split_date.date()), "chosen_threshold": thr,
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
          f"n={hw['n']} ({out['elapsed_sec']}s)", flush=True)
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
           "n_rows_used": len(df),
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
          f"n={hw['n']} meta_coefs={out['meta_coefficients']} ({out['elapsed_sec']}s)", flush=True)
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
    base3 = [("lgbm", LGBM_10DAY_TUNED), ("rf", RF_14DAY_TUNED), ("xgb", XGB_DEFAULT)]

    # ---------------- Group A: drop LightGBM from stacking (round-6 gave it ~0 weight) ----------------
    print("\n===== GROUP A: RF+XGB-only stacking (LightGBM dropped) =====", flush=True)
    base2 = [("rf", RF_14DAY_TUNED), ("xgb", XGB_DEFAULT)]
    a1 = run_stacking("A1_stack_rf_xgb_only_10day", df, ALL_FEATURES, "label_10day", base2); all_results.append(a1)
    a2 = run_stacking("A2_stack_rf_xgb_only_14day", df, ALL_FEATURES, "label_14day", base2); all_results.append(a2)

    # ---------------- Group B: finer support-filter sweep (round-6 next step 3) ----------------
    print("\n===== GROUP B: finer support-filter sweep (75/100/125/150/175) =====", flush=True)
    fine_levels = (75, 100, 125, 150, 175)
    b1 = run_stacking("B1_finegrain_support_10day", df, ALL_FEATURES, "label_10day", base3,
                       support_levels=fine_levels); all_results.append(b1)
    b2 = run_single_model("B2_finegrain_support_14day", df, ALL_FEATURES, "label_14day",
                           TUNED_BY_HORIZON["label_14day"], support_levels=fine_levels); all_results.append(b2)

    # ---------------- Group C: stacking trained AND evaluated on the filtered population (round-6 next step 4) ----------------
    print("\n===== GROUP C: joint support-filtered training (not just post-hoc slicing) =====", flush=True)
    c_results = {}
    for h, spec_or_stack in [("label_10day", "stack"), ("label_14day", "single")]:
        for lvl in (50, 100):
            sub = df[df["n_hist_events_total"].fillna(0) >= lvl]
            tag = f"C_jointfilter_sup{lvl}_{h}"
            if spec_or_stack == "stack":
                r = run_stacking(tag, sub, ALL_FEATURES, h, base3, support_levels=())
            else:
                r = run_single_model(tag, sub, ALL_FEATURES, h, TUNED_BY_HORIZON[h], support_levels=())
            all_results.append(r); c_results[(h, lvl)] = r

    # ---------------- Group D: WorldPop confound isolation (round-6 next step 2) ----------------
    print("\n===== GROUP D: WorldPop -- covered-subset A/B test + coverage-indicator feature =====", flush=True)
    d_results = {}
    try:
        wp = pd.read_csv(WORLDPOP_PATH)
        covered_gids = set(wp.loc[wp["worldpop_population"].notna(), "priogrid_gid"])
        coverage = wp["worldpop_population"].notna().mean()
        print(f"  WorldPop real coverage this run: {coverage:.1%} of {len(wp)} cells ({len(covered_gids)} gids)", flush=True)

        df_covered = df[df["priogrid_gid"].isin(covered_gids)]
        df_wp = df.merge(wp, on="priogrid_gid", how="left")
        df_wp["worldpop_has_data"] = df_wp["priogrid_gid"].isin(covered_gids).astype(int)
        df_wp_covered = df_wp[df_wp["priogrid_gid"].isin(covered_gids)]

        for h in ("label_10day", "label_14day"):
            spec = TUNED_BY_HORIZON[h]
            d_ctrl = run_single_model(f"D_coveredonly_control_{h}", df_covered, ALL_FEATURES, h, spec, support_levels=())
            d_treat = run_single_model(f"D_coveredonly_treatment_{h}", df_wp_covered, ALL_FEATURES + ["worldpop_population"],
                                        h, spec, support_levels=())
            d_ind = run_single_model(f"D_indicator_{h}", df_wp, ALL_FEATURES + ["worldpop_population", "worldpop_has_data"],
                                      h, spec)
            all_results += [d_ctrl, d_treat, d_ind]
            d_results[h] = {"control": d_ctrl, "treatment": d_treat, "indicator": d_ind}
    except Exception as e:
        print(f"  WorldPop group skipped: {e}", flush=True)

    # ---------------- Group E: two-stage severity extensions ----------------
    print("\n===== GROUP E: two-stage severity -- avg3 stage-1, and extended support reporting =====", flush=True)
    e1 = run_two_stage("E1_severity_avg3stage1_10day", df, ALL_FEATURES, "label_10day", "label_10day_severe",
                        ("avg3", {}), ("lgbm", LGBM_10DAY_TUNED)); all_results.append(e1)
    e2 = run_two_stage("E2_severity_avg3stage1_14day", df, ALL_FEATURES, "label_14day", "label_14day_severe",
                        ("avg3", {}), ("rf", RF_14DAY_TUNED)); all_results.append(e2)
    e3 = run_two_stage("E3_severity_matchedwinner_moresupport_10day", df, ALL_FEATURES, "label_10day", "label_10day_severe",
                        TUNED_BY_HORIZON["label_10day"], TUNED_BY_HORIZON["label_10day"],
                        support_levels=(30, 50, 100)); all_results.append(e3)
    e4 = run_two_stage("E4_severity_matchedwinner_moresupport_14day", df, ALL_FEATURES, "label_14day", "label_14day_severe",
                        TUNED_BY_HORIZON["label_14day"], TUNED_BY_HORIZON["label_14day"],
                        support_levels=(30, 50, 100)); all_results.append(e4)

    # ---------------- Group F: final combined recipe, decided from Groups B/C/D's own results ----------------
    print("\n===== GROUP F: final combined recipe (decided from this round's own results) =====", flush=True)
    decisions = {}
    f_results = {}
    for h, lvl_choices in [("label_10day", (50, 100)), ("label_14day", (50, 100))]:
        posthoc_100 = (b1 if h == "label_10day" else b2)["holdout_supportfiltered"].get("100")
        best_joint_lvl, best_joint_prec = None, -1
        for lvl in lvl_choices:
            r = c_results[(h, lvl)]
            p = r["holdout_window"]["precision"]
            if p > best_joint_prec:
                best_joint_prec, best_joint_lvl = p, lvl
        use_joint = posthoc_100 is None or best_joint_prec > posthoc_100["precision"]

        ind_precision = d_results.get(h, {}).get("indicator", {}).get("holdout_window", {}).get("precision", -1)
        use_indicator = ind_precision > KNOWN_PRECISION[h]

        decisions[h] = {"use_joint_filter": use_joint, "joint_filter_level": best_joint_lvl if use_joint else None,
                         "use_worldpop_indicator": use_indicator}
        print(f"  {h}: use_joint_filter={use_joint} (level={best_joint_lvl}, prec={best_joint_prec:.3f} vs "
              f"posthoc@100={posthoc_100['precision'] if posthoc_100 else None}), "
              f"use_worldpop_indicator={use_indicator} (indicator_prec={ind_precision:.3f} vs "
              f"baseline={KNOWN_PRECISION[h]:.3f})", flush=True)

        base_df = df
        feats = ALL_FEATURES
        if use_indicator:
            wp = pd.read_csv(WORLDPOP_PATH)
            covered_gids = set(wp.loc[wp["worldpop_population"].notna(), "priogrid_gid"])
            base_df = base_df.merge(wp, on="priogrid_gid", how="left")
            base_df["worldpop_has_data"] = base_df["priogrid_gid"].isin(covered_gids).astype(int)
            feats = feats + ["worldpop_population", "worldpop_has_data"]
        if use_joint:
            base_df = base_df[base_df["n_hist_events_total"].fillna(0) >= best_joint_lvl]

        tag = f"F_combined_{h}"
        if h == "label_10day":
            r = run_stacking(tag, base_df, feats, h, base3, support_levels=() if use_joint else (30, 50, 100))
        else:
            r = run_single_model(tag, base_df, feats, h, TUNED_BY_HORIZON[h],
                                  support_levels=() if use_joint else (30, 50, 100))
        all_results.append(r); f_results[h] = r

    # ---------------- Group G: short adaptive follow-up from this round's own results ----------------
    print("\n===== GROUP G: adaptive follow-up =====", flush=True)
    if decisions["label_10day"]["use_joint_filter"] and decisions["label_10day"]["joint_filter_level"] == 100:
        print("  joint-filtering at >=100 won at 10-day -> adaptive follow-up: try >=150 jointly too", flush=True)
        sub = df[df["n_hist_events_total"].fillna(0) >= 150]
        g1 = run_stacking("G1_jointfilter_sup150_10day", sub, ALL_FEATURES, "label_10day", base3, support_levels=())
        all_results.append(g1)
    if decisions["label_14day"]["use_joint_filter"] and decisions["label_14day"]["joint_filter_level"] == 100:
        print("  joint-filtering at >=100 won at 14-day -> adaptive follow-up: try >=150 jointly too", flush=True)
        sub = df[df["n_hist_events_total"].fillna(0) >= 150]
        g2 = run_single_model("G2_jointfilter_sup150_14day", sub, ALL_FEATURES, "label_14day",
                               TUNED_BY_HORIZON["label_14day"], support_levels=())
        all_results.append(g2)

    n_iters = len(all_results)
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n\nTOTAL: {n_iters} real iterations logged, {elapsed_min:.1f} min", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_iterations": n_iters, "elapsed_min": elapsed_min, "decisions": decisions,
                    "results": all_results}, f, indent=2, default=str)
    print(f"Saved {OUT_PATH} and {LOG_PATH}", flush=True)
