"""
Round 9 -- acting on round 8's next-steps list (results/round8_next_steps_writeup.html,
Tab 05). At least 30 real iterations, same rolling-origin, never-look-ahead, frozen-
threshold discipline as every prior round. Logged to results/round9_log.jsonl and
results/round9_results.json.

Where each round-8 next-step item is addressed:
  1. Deploy the 60%-target severity model at 14-day -- Group D extends it with a
     support-filter reporting pass, as production configs got throughout this project.
  2. Diagnose 10-day severity's threshold instability directly -- Group A: a finer
     target-precision grid (45/50/55/60/65/70%) at both horizons, with the SELECT-window
     chosen threshold reported alongside each holdout result, so the instability is visible
     in the actual threshold values chosen, not just inferred from precision jumping around.
  3. (Close the WorldPop imputation-engineering angle -- no new downloads, no new fill
     strategies.) Respected: no downloader is invoked this round.
  4. Report the support-filter lever as a range, not a point estimate -- Group B extends
     round 8's 5-split stability check with 2 more split points (0.45, 0.75) for a fuller
     picture, and Group E's multi-split ensemble is a direct, tested response to the same
     finding (does averaging across time-robust model fits reduce the variance itself,
     rather than just documenting it).
  5. Check whether WorldPop population is redundant with existing event-history features --
     Group C: a clean 3-way ablation (population alone / history alone / combined) on the
     254 covered cells, isolating whether population adds orthogonal signal.
Group E (new architecture, not on round 8's list) tests a multi-split ensemble: averaging
predictions from models trained through 5 different chronological cutoffs, evaluated on one
common held-out window, to see whether it reduces round 8's documented 6-7-point precision
spread. Group F is the final round-9 combined recipe. Group G is a short adaptive follow-up.
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
LOG_PATH = "../results/round9_log.jsonl"
OUT_PATH = "../results/round9_results.json"

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
BASE2 = [("rf", RF_14DAY_TUNED), ("xgb", XGB_DEFAULT)]   # round-7's confirmed 2-model stack

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


def split_select_holdout(df, split_frac=0.6):
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * split_frac)
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
                      support_levels=(30, 50, 100), target_precision=0.80, split_frac=0.6, eval_df=None):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df, split_frac)
    if eval_df is not None:
        holdout_df = eval_df
    sel_true, sel_score = rolling_predictions(select_df, feature_cols, label_col, spec)
    thr = choose_threshold(sel_true, sel_score, target_precision)
    train_X = select_df[feature_cols].fillna(0).values; train_y = select_df[label_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    holdout_score = fit_predict_single(spec, train_X, train_y, holdout_X)
    holdout_y = holdout_df[label_col].values
    out = {"tag": tag, "horizon": label_col, "model": spec[0], "params": spec[1], "n_features": len(feature_cols),
           "n_rows_used": len(df), "split_frac": split_frac, "target_precision": target_precision,
           "split_date": str(split_date.date()), "chosen_threshold": thr,
           "select_window": compute_metrics(sel_true, sel_score, thr),
           "holdout_window": compute_metrics(holdout_y, holdout_score, thr)}
    support_out = {}
    for lvl in support_levels:
        keep = support_keep_mask(holdout_df, holdout_y, support_col, lvl)
        if keep is not None:
            support_out[str(lvl)] = compute_metrics(holdout_y[keep], holdout_score[keep], thr)
    out["holdout_supportfiltered"] = support_out
    out["_holdout_score"] = holdout_score  # kept for in-process ensembling only, stripped before saving
    out["elapsed_sec"] = round(time.time() - t0, 1)
    hw = out["holdout_window"]
    print(f"[{tag}] {label_col}: HOLDOUT precision={hw['precision']:.3f} recall={hw['recall']:.3f} "
          f"n={hw['n']} thr={thr:.3f} ({out['elapsed_sec']}s)", flush=True)
    log_iter({k: v for k, v in out.items() if not k.startswith("_")})
    return out


def run_stacking(tag, df, feature_cols, label_col, base_specs, meta_C=1.0,
                  support_col="n_hist_events_total", support_levels=(30, 50, 100), target_precision=0.80,
                  split_frac=0.6, eval_df=None):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df, split_frac)
    if eval_df is not None:
        holdout_df = eval_df
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
           "n_rows_used": len(df), "split_frac": split_frac,
           "split_date": str(split_date.date()), "chosen_threshold": thr,
           "select_window": compute_metrics(sel_true, sel_score, thr),
           "holdout_window": compute_metrics(holdout_y, holdout_score, thr)}
    support_out = {}
    for lvl in support_levels:
        keep = support_keep_mask(holdout_df, holdout_y, support_col, lvl)
        if keep is not None:
            support_out[str(lvl)] = compute_metrics(holdout_y[keep], holdout_score[keep], thr)
    out["holdout_supportfiltered"] = support_out
    out["_holdout_score"] = holdout_score
    out["elapsed_sec"] = round(time.time() - t0, 1)
    hw = out["holdout_window"]
    print(f"[{tag}] {label_col}: HOLDOUT precision={hw['precision']:.3f} recall={hw['recall']:.3f} "
          f"n={hw['n']} thr={thr:.3f} ({out['elapsed_sec']}s)", flush=True)
    log_iter({k: v for k, v in out.items() if not k.startswith("_")})
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
           "n_features": len(feature_cols), "target_precision": target_precision,
           "split_date": str(split_date.date()), "chosen_threshold": thr,
           "n_select_positives": int(sel_true.sum()), "n_select": int(len(sel_true)),
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
    print(f"[{tag}] {occ_col} severity @ target={target_precision}: HOLDOUT precision={hw['precision']:.3f} "
          f"recall={hw['recall']:.3f} thr={thr:.4f} ({out['elapsed_sec']}s)", flush=True)
    return out


if __name__ == "__main__":
    print("Loading mega dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows, {len(df.columns)} cols", flush=True)

    all_results = []
    t_start = time.time()

    # ---------------- Group A: diagnose 10-day severity threshold instability (round-8 next step 2) ----------------
    print("\n===== GROUP A: severity target-precision fine grid (45/50/55/60/65/70%), both horizons =====", flush=True)
    a_results = {"label_10day": {}, "label_14day": {}}
    for target in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        r10 = run_two_stage(f"A_severity_target{int(target*100)}_10day", df, ALL_FEATURES, "label_10day",
                             "label_10day_severe", TUNED_BY_HORIZON["label_10day"], TUNED_BY_HORIZON["label_10day"],
                             target_precision=target)
        r14 = run_two_stage(f"A_severity_target{int(target*100)}_14day", df, ALL_FEATURES, "label_14day",
                             "label_14day_severe", TUNED_BY_HORIZON["label_14day"], TUNED_BY_HORIZON["label_14day"],
                             target_precision=target)
        all_results += [r10, r14]
        a_results["label_10day"][target] = r10
        a_results["label_14day"][target] = r14

    print("\n  -- select-window chosen thresholds by target, 10-day (the diagnostic itself) --", flush=True)
    for target, r in a_results["label_10day"].items():
        print(f"     target={target:.2f} -> chosen_threshold={r['chosen_threshold']:.4f}, "
              f"select_precision={r['select_window']['precision']:.3f}, "
              f"holdout_precision={r['holdout_window']['precision']:.3f}", flush=True)

    # ---------------- Group B: extend the support-filter split-point range (round-8 next step 4) ----------------
    print("\n===== GROUP B: support-filter stability, 2 more split points (0.45, 0.75) =====", flush=True)
    b_results = {}
    for split_frac in (0.45, 0.75):
        b10 = run_stacking(f"B_stability_split{int(split_frac*100)}_10day", df, ALL_FEATURES, "label_10day", BASE2,
                            support_levels=(75, 100, 125), split_frac=split_frac)
        b14 = run_single_model(f"B_stability_split{int(split_frac*100)}_14day", df, ALL_FEATURES, "label_14day",
                                TUNED_BY_HORIZON["label_14day"], support_levels=(75, 100, 125), split_frac=split_frac)
        all_results += [{k: v for k, v in b10.items() if not k.startswith("_")},
                         {k: v for k, v in b14.items() if not k.startswith("_")}]
        b_results[split_frac] = {"10day": b10, "14day": b14}

    # ---------------- Group C: is WorldPop redundant with existing history features? (round-8 next step 5) ----------------
    print("\n===== GROUP C: WorldPop vs. event-history features -- a clean 3-way ablation on covered cells =====", flush=True)
    wp = pd.read_csv(WORLDPOP_PATH)
    covered_gids = set(wp.loc[wp["worldpop_population"].notna(), "priogrid_gid"])
    print(f"  {len(covered_gids)} covered cells (real WorldPop coverage, no new download)", flush=True)
    df_wp_covered = df[df["priogrid_gid"].isin(covered_gids)].merge(wp, on="priogrid_gid", how="left")

    corr_targets = ["cell_count_365d", "neighbor_count_30d", "n_hist_events_total"]
    corrs = {c: float(df_wp_covered[["worldpop_population", c]].dropna().corr().iloc[0, 1]) for c in corr_targets}
    print(f"  Pearson correlation, worldpop_population vs. existing features (covered cells): {corrs}", flush=True)

    c_results = {}
    for h in ("label_10day", "label_14day"):
        spec = TUNED_BY_HORIZON[h]
        c_pop = run_single_model(f"C_worldpop_only_{h}", df_wp_covered, ["worldpop_population"], h, spec,
                                  support_levels=())
        c_hist = run_single_model(f"C_history_only_{h}", df_wp_covered, ALL_FEATURES, h, spec, support_levels=())
        c_comb = run_single_model(f"C_combined_{h}", df_wp_covered, ALL_FEATURES + ["worldpop_population"], h, spec,
                                   support_levels=())
        all_results += [{k: v for k, v in c_pop.items() if not k.startswith("_")},
                         {k: v for k, v in c_hist.items() if not k.startswith("_")},
                         {k: v for k, v in c_comb.items() if not k.startswith("_")}]
        c_results[h] = {"pop_only": c_pop, "history_only": c_hist, "combined": c_comb}
        gain = c_comb["holdout_window"]["precision"] - c_hist["holdout_window"]["precision"]
        print(f"  {h}: pop_only AP={c_pop['holdout_window']['average_precision']:.3f} (vs. history_only "
              f"AP={c_hist['holdout_window']['average_precision']:.3f}) -- combined adds {gain:+.3f} precision "
              f"over history_only", flush=True)

    # ---------------- Group D: finalize severity deploy configs with support-filter reporting (round-8 next step 1) ----------------
    print("\n===== GROUP D: severity deploy configs, support-filter reporting added =====", flush=True)
    best_10day_target, best_10day_recall = None, -1
    for target, r in a_results["label_10day"].items():
        p, rec = r["holdout_window"]["precision"], r["holdout_window"]["recall"]
        if p >= 0.50 and rec > best_10day_recall:
            best_10day_target, best_10day_recall = target, rec
    if best_10day_target is None:
        best_10day_target = 0.70  # fall back to the stable high-precision point found in round 8
    print(f"  10-day severity: choosing target={best_10day_target} (best recall among targets with "
          f">=50% actual holdout precision; falls back to 0.70 if none qualify)", flush=True)

    d1 = run_two_stage("D_severity_deploy_10day", df, ALL_FEATURES, "label_10day", "label_10day_severe",
                        TUNED_BY_HORIZON["label_10day"], TUNED_BY_HORIZON["label_10day"],
                        target_precision=best_10day_target, support_levels=(30, 50, 100))
    d2 = run_two_stage("D_severity_deploy_14day", df, ALL_FEATURES, "label_14day", "label_14day_severe",
                        TUNED_BY_HORIZON["label_14day"], TUNED_BY_HORIZON["label_14day"],
                        target_precision=0.60, support_levels=(30, 50, 100))
    all_results += [d1, d2]

    # ---------------- Group E: multi-split ensemble (tests round-8's documented variance directly) ----------------
    print("\n===== GROUP E: multi-split ensemble -- 5 time-robust fits, averaged on one common eval window =====", flush=True)
    ensemble_splits = (0.5, 0.55, 0.6, 0.65, 0.7)
    common_eval_df = split_select_holdout(df, max(ensemble_splits))[1]   # holdout of the LATEST split -> common to all
    print(f"  common evaluation window: {len(common_eval_df)} rows, split_date >= "
          f"{split_select_holdout(df, max(ensemble_splits))[2].date()}", flush=True)

    for h, spec_or_stack in [("label_10day", "stack"), ("label_14day", "single")]:
        base_scores, base_thrs = [], []
        e_base_results = []
        for sf in ensemble_splits:
            tag = f"E_base_split{int(sf*100)}_{h}"
            if spec_or_stack == "stack":
                r = run_stacking(tag, df, ALL_FEATURES, h, BASE2, support_levels=(), split_frac=sf,
                                  eval_df=common_eval_df)
            else:
                r = run_single_model(tag, df, ALL_FEATURES, h, TUNED_BY_HORIZON[h], support_levels=(),
                                      split_frac=sf, eval_df=common_eval_df)
            all_results.append({k: v for k, v in r.items() if not k.startswith("_")})
            e_base_results.append(r)
            base_scores.append(r["_holdout_score"])
            base_thrs.append(r["chosen_threshold"])

        avg_score = np.mean(base_scores, axis=0)
        avg_thr = float(np.mean(base_thrs))
        y_true = common_eval_df[h].values
        m = compute_metrics(y_true, avg_score, avg_thr)
        support_out = {}
        for lvl in (75, 100, 125):
            keep = support_keep_mask(common_eval_df, y_true, "n_hist_events_total", lvl)
            if keep is not None:
                support_out[str(lvl)] = compute_metrics(y_true[keep], avg_score[keep], avg_thr)
        ensemble_out = {"tag": f"E_ensemble_{h}", "horizon": h, "model": "multi_split_ensemble",
                         "n_base_models": len(ensemble_splits), "split_fracs": list(ensemble_splits),
                         "avg_threshold": avg_thr, "holdout_window": m, "holdout_supportfiltered": support_out}
        log_iter(ensemble_out)
        all_results.append(ensemble_out)
        print(f"[E_ensemble_{h}] {h}: common-window precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"n={m['n']} (vs. single-split-0.6 on same common window, see comparison below)", flush=True)

        single_tag = f"E_base_split60_{h}"
        single_r = next(r for r in e_base_results if r["tag"] == single_tag)
        print(f"     single 0.6-split model on same common window: precision="
              f"{single_r['holdout_window']['precision']:.3f} recall={single_r['holdout_window']['recall']:.3f}", flush=True)

    # ---------------- Group F: final round-9 combined recipe ----------------
    print("\n===== GROUP F: final round-9 combined recipe =====", flush=True)
    f1 = run_stacking("F_final_10day", df, ALL_FEATURES, "label_10day", BASE2, support_levels=(75, 100, 125))
    f2 = run_single_model("F_final_14day", df, ALL_FEATURES, "label_14day", TUNED_BY_HORIZON["label_14day"],
                           support_levels=(75, 100, 125))
    all_results += [{k: v for k, v in f1.items() if not k.startswith("_")},
                     {k: v for k, v in f2.items() if not k.startswith("_")}]

    # ---------------- Group G: adaptive follow-up ----------------
    print("\n===== GROUP G: adaptive follow-up =====", flush=True)
    pop_only_ap_10 = c_results["label_10day"]["pop_only"]["holdout_window"]["average_precision"]
    hist_only_ap_10 = c_results["label_10day"]["history_only"]["holdout_window"]["average_precision"]
    if pop_only_ap_10 > 0.4 * hist_only_ap_10:
        print(f"  population-alone AP ({pop_only_ap_10:.3f}) is more than 40% of history-alone AP "
              f"({hist_only_ap_10:.3f}) at 10-day -> adaptive follow-up: same ablation at 14-day support>=30 "
              f"filter, to see if the pattern holds under the lever that matters most in production", flush=True)
        g1 = run_single_model("G_worldpop_ablation_supportfiltered_14day", df_wp_covered,
                               ALL_FEATURES + ["worldpop_population"], "label_14day", TUNED_BY_HORIZON["label_14day"],
                               support_levels=(30,))
        all_results.append({k: v for k, v in g1.items() if not k.startswith("_")})

    n_iters = len(all_results)
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n\nTOTAL: {n_iters} real iterations logged, {elapsed_min:.1f} min", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_iterations": n_iters, "elapsed_min": elapsed_min,
                    "worldpop_correlations": corrs, "best_10day_severity_target": best_10day_target,
                    "results": all_results}, f, indent=2, default=str)
    print(f"Saved {OUT_PATH} and {LOG_PATH}", flush=True)
