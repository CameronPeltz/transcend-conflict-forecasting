"""
Round 8 -- acting on round 7's next-steps list, explicitly EXCLUDING next step 1
(finishing the WorldPop download) per direct instruction: no new downloads this
round. Uses only whatever WorldPop coverage already exists on disk. Same
rolling-origin, never-look-ahead, frozen-threshold discipline as every prior
round. At least 20 real iterations, logged to results/round8_log.jsonl and
results/round8_results.json.

Where each round-7 next-step item is addressed:
  1. (Finish the WorldPop download) -- explicitly SKIPPED this round. The
     downloader had kept running in the background since round 7 and finished
     Sudan on its own (918.9 MB, verified) before this round started, which is
     used below as a free bonus (254/464 cells, 54.7% coverage, up from 46.1%)
     -- no new download was triggered to get it.
  2. Try a smarter imputation than zero-fill for uncovered WorldPop cells --
     Group A: fills missing population with the covered-cell population mean
     instead of zero (a coarser "population-mean" version of the suggested
     country/region-mean, since coverage is all-or-nothing per country here),
     alone and combined with the coverage-indicator flag.
  3. Deploy the 2-model (RF+XGB) stack at 10-day -- Group B: finalizes that
     recipe with the confirmed support-filter optimum, cleanly re-validated.
  4. (Joint-filtered training) -- closed in round 7, not re-tested.
  5. Re-calibrate severity to a realistic target precision instead of
     inheriting the occurrence task's 80% -- Group C: sweeps target precision
     (50%/60%/70%) for the two-stage matched-winner severity model.
  6. Test whether the >=100 support-filter optimum is stable across different
     holdout split points, not just the one 60/40 split used throughout --
     Group D: re-runs the winning configs at 50/60/70% split points.
Group E builds a final round-8 combined recipe from whichever levers actually
won across Groups A/B/D, decided programmatically. Group F is a short adaptive
follow-up chosen from this round's own results.
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
LOG_PATH = "../results/round8_log.jsonl"
OUT_PATH = "../results/round8_results.json"

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
KNOWN_PRECISION = {"label_10day": 0.788, "label_14day": 0.760}   # round-7 unfiltered bests
BASE2 = [("rf", RF_14DAY_TUNED), ("xgb", XGB_DEFAULT)]           # round-7's confirmed 2-model stack

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
                      support_levels=(30, 50, 100), target_precision=0.80, split_frac=0.6):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df, split_frac)
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
    out["elapsed_sec"] = round(time.time() - t0, 1)
    log_iter(out)
    hw = out["holdout_window"]
    print(f"[{tag}] {label_col}: HOLDOUT precision={hw['precision']:.3f} recall={hw['recall']:.3f} "
          f"n={hw['n']} ({out['elapsed_sec']}s)", flush=True)
    return out


def run_stacking(tag, df, feature_cols, label_col, base_specs, meta_C=1.0,
                  support_col="n_hist_events_total", support_levels=(30, 50, 100), target_precision=0.80,
                  split_frac=0.6):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df, split_frac)
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
          f"n={hw['n']} ({out['elapsed_sec']}s)", flush=True)
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
                   support_col="n_hist_events_total", support_levels=(30, 50), target_precision=0.80,
                   split_frac=0.6):
    t0 = time.time()
    select_df, holdout_df, split_date = split_select_holdout(df, split_frac)
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
          f"recall={hw['recall']:.3f} ({out['elapsed_sec']}s)", flush=True)
    return out


if __name__ == "__main__":
    print("Loading mega dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    print(f"{len(df)} rows, {len(df.columns)} cols", flush=True)

    all_results = []
    t_start = time.time()

    # ---------------- Group A: WorldPop imputation fix, no new downloads (round-7 next step 2) ----------------
    print("\n===== GROUP A: WorldPop population-mean imputation (uses existing downloads only) =====", flush=True)
    wp = pd.read_csv(WORLDPOP_PATH)
    covered_gids = set(wp.loc[wp["worldpop_population"].notna(), "priogrid_gid"])
    coverage = wp["worldpop_population"].notna().mean()
    pop_mean = float(wp["worldpop_population"].mean())
    print(f"  WorldPop real coverage now: {coverage:.1%} of {len(wp)} cells ({len(covered_gids)} gids) "
          f"-- Sudan finished in the background since round 7, no new download triggered this round; "
          f"covered-cell population mean = {pop_mean:,.0f}", flush=True)
    wp_meanfill = wp.copy()
    wp_meanfill["worldpop_population"] = wp_meanfill["worldpop_population"].fillna(pop_mean)
    df_meanfill = df.merge(wp_meanfill, on="priogrid_gid", how="left")
    df_meanfill["worldpop_population"] = df_meanfill["worldpop_population"].fillna(pop_mean)  # rows w/ no gid match at all
    df_meanfill_ind = df_meanfill.copy()
    df_meanfill_ind["worldpop_has_data"] = df_meanfill_ind["priogrid_gid"].isin(covered_gids).astype(int)

    a1 = run_single_model("A1_worldpop_meanfill_10day", df_meanfill, ALL_FEATURES + ["worldpop_population"],
                           "label_10day", TUNED_BY_HORIZON["label_10day"]); all_results.append(a1)
    a2 = run_single_model("A2_worldpop_meanfill_14day", df_meanfill, ALL_FEATURES + ["worldpop_population"],
                           "label_14day", TUNED_BY_HORIZON["label_14day"]); all_results.append(a2)
    a3 = run_single_model("A3_worldpop_meanfill_plus_indicator_10day", df_meanfill_ind,
                           ALL_FEATURES + ["worldpop_population", "worldpop_has_data"],
                           "label_10day", TUNED_BY_HORIZON["label_10day"]); all_results.append(a3)
    a4 = run_single_model("A4_worldpop_meanfill_plus_indicator_14day", df_meanfill_ind,
                           ALL_FEATURES + ["worldpop_population", "worldpop_has_data"],
                           "label_14day", TUNED_BY_HORIZON["label_14day"]); all_results.append(a4)

    worldpop_variants_10day = {"meanfill": a1, "meanfill_indicator": a3}
    worldpop_variants_14day = {"meanfill": a2, "meanfill_indicator": a4}
    best_wp_10day_tag = max(worldpop_variants_10day, key=lambda k: worldpop_variants_10day[k]["holdout_window"]["precision"])
    best_wp_14day_tag = max(worldpop_variants_14day, key=lambda k: worldpop_variants_14day[k]["holdout_window"]["precision"])
    best_wp_10day_prec = worldpop_variants_10day[best_wp_10day_tag]["holdout_window"]["precision"]
    best_wp_14day_prec = worldpop_variants_14day[best_wp_14day_tag]["holdout_window"]["precision"]
    print(f"  --> best WorldPop variant: 10-day={best_wp_10day_tag} ({best_wp_10day_prec:.3f} vs "
          f"baseline {KNOWN_PRECISION['label_10day']:.3f}), 14-day={best_wp_14day_tag} "
          f"({best_wp_14day_prec:.3f} vs baseline {KNOWN_PRECISION['label_14day']:.3f})", flush=True)

    # ---------------- Group B: finalize the 2-model-stack deploy recipe (round-7 next step 3) ----------------
    print("\n===== GROUP B: 2-model stack (RF+XGB), final deploy recipe with support sweep =====", flush=True)
    b1 = run_stacking("B1_deploy_2model_stack_10day", df, ALL_FEATURES, "label_10day", BASE2,
                       support_levels=(75, 100, 125)); all_results.append(b1)
    b2 = run_single_model("B2_deploy_rf_14day", df, ALL_FEATURES, "label_14day", TUNED_BY_HORIZON["label_14day"],
                           support_levels=(75, 100, 125)); all_results.append(b2)

    # ---------------- Group C: re-calibrate severity's target precision (round-7 next step 5) ----------------
    print("\n===== GROUP C: severity target-precision sweep (50%/60%/70%, vs. the 80% used through round 7) =====", flush=True)
    c_results = {}
    for target in (0.50, 0.60, 0.70):
        c10 = run_two_stage(f"C_severity_target{int(target*100)}_10day", df, ALL_FEATURES, "label_10day",
                             "label_10day_severe", TUNED_BY_HORIZON["label_10day"], TUNED_BY_HORIZON["label_10day"],
                             target_precision=target)
        c14 = run_two_stage(f"C_severity_target{int(target*100)}_14day", df, ALL_FEATURES, "label_14day",
                             "label_14day_severe", TUNED_BY_HORIZON["label_14day"], TUNED_BY_HORIZON["label_14day"],
                             target_precision=target)
        all_results += [c10, c14]
        c_results[target] = {"10day": c10, "14day": c14}

    # ---------------- Group D: is the >=100 support optimum stable across holdout windows? (round-7 next step 6) ----------------
    print("\n===== GROUP D: support-filter optimum stability across split points (50%/70%, vs. the 60% used throughout) =====", flush=True)
    d_results = {}
    for split_frac in (0.5, 0.65, 0.7):
        d10 = run_stacking(f"D_stability_split{int(split_frac*100)}_10day", df, ALL_FEATURES, "label_10day", BASE2,
                            support_levels=(75, 100, 125), split_frac=split_frac)
        d14 = run_single_model(f"D_stability_split{int(split_frac*100)}_14day", df, ALL_FEATURES, "label_14day",
                                TUNED_BY_HORIZON["label_14day"], support_levels=(75, 100, 125), split_frac=split_frac)
        all_results += [d10, d14]
        d_results[split_frac] = {"10day": d10, "14day": d14}

    # ---------------- Group E: final round-8 combined recipe, decided from Groups A/B/D ----------------
    print("\n===== GROUP E: final round-8 combined recipe =====", flush=True)
    decisions = {
        "10day": {"use_worldpop": best_wp_10day_prec > KNOWN_PRECISION["label_10day"], "worldpop_variant": best_wp_10day_tag},
        "14day": {"use_worldpop": best_wp_14day_prec > KNOWN_PRECISION["label_14day"], "worldpop_variant": best_wp_14day_tag},
    }
    print(f"  decisions: {decisions}", flush=True)

    feats_10 = ALL_FEATURES + (["worldpop_population"] if decisions["10day"]["use_worldpop"] else [])
    base_df_10 = df_meanfill_ind if "indicator" in decisions["10day"]["worldpop_variant"] else df_meanfill
    if decisions["10day"]["use_worldpop"] and "indicator" in decisions["10day"]["worldpop_variant"]:
        feats_10 = feats_10 + ["worldpop_has_data"]
    e1 = run_stacking("E1_combined_10day", base_df_10 if decisions["10day"]["use_worldpop"] else df,
                       feats_10, "label_10day", BASE2, support_levels=(75, 100, 125)); all_results.append(e1)

    feats_14 = ALL_FEATURES + (["worldpop_population"] if decisions["14day"]["use_worldpop"] else [])
    base_df_14 = df_meanfill_ind if "indicator" in decisions["14day"]["worldpop_variant"] else df_meanfill
    if decisions["14day"]["use_worldpop"] and "indicator" in decisions["14day"]["worldpop_variant"]:
        feats_14 = feats_14 + ["worldpop_has_data"]
    e2 = run_single_model("E2_combined_14day", base_df_14 if decisions["14day"]["use_worldpop"] else df,
                           feats_14, "label_14day", TUNED_BY_HORIZON["label_14day"],
                           support_levels=(75, 100, 125)); all_results.append(e2)

    e3 = run_two_stage("E3_combined_severity_target60_10day", df, ALL_FEATURES, "label_10day", "label_10day_severe",
                        TUNED_BY_HORIZON["label_10day"], TUNED_BY_HORIZON["label_10day"], target_precision=0.60)
    e4 = run_two_stage("E4_combined_severity_target60_14day", df, ALL_FEATURES, "label_14day", "label_14day_severe",
                        TUNED_BY_HORIZON["label_14day"], TUNED_BY_HORIZON["label_14day"], target_precision=0.60)
    all_results += [e3, e4]

    # ---------------- Group F: adaptive follow-up from this round's own results ----------------
    print("\n===== GROUP F: adaptive follow-up =====", flush=True)
    precisions_by_split_10 = {0.6: b1["holdout_supportfiltered"].get("100", {}).get("precision")}
    for sf, r in d_results.items():
        precisions_by_split_10[sf] = r["10day"]["holdout_supportfiltered"].get("100", {}).get("precision")
    spread_10 = (max(v for v in precisions_by_split_10.values() if v is not None) -
                 min(v for v in precisions_by_split_10.values() if v is not None))
    print(f"  10-day support>=100 precision across split points: {precisions_by_split_10} (spread={spread_10:.3f})", flush=True)
    if spread_10 > 0.05:
        print("  spread > 5 points -> optimum is NOT stable -> adaptive follow-up: test support>=100 at an "
              "intermediate split (0.55) to locate where the instability begins", flush=True)
        f1 = run_stacking("F1_stability_split55_10day", df, ALL_FEATURES, "label_10day", BASE2,
                           support_levels=(100,), split_frac=0.55)
        all_results.append(f1)
    else:
        print("  spread <= 5 points -> optimum looks reasonably stable, no follow-up needed", flush=True)

    n_iters = len(all_results)
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n\nTOTAL: {n_iters} real iterations logged, {elapsed_min:.1f} min", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_iterations": n_iters, "elapsed_min": elapsed_min, "decisions": decisions,
                    "worldpop_coverage": coverage, "results": all_results}, f, indent=2, default=str)
    print(f"Saved {OUT_PATH} and {LOG_PATH}", flush=True)
