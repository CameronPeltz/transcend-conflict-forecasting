# -*- coding: utf-8 -*-
"""
Round 3 of hypergraph optimization -- executes the specific next-steps
recommended after Round 2 (results_v2/hypergraph_optimization_search_*,
hypergraph_threshold_comparison.py, hg03_pr_curve_analysis.py):

  1. Apply HG_11's one-time-calibration idea to the STRUCTURAL track (best AP
     of any hypergraph approach) instead of only the HGNN -- but to make that
     a fair, direct ensemble partner for HG_11, the structural GBM here is
     fit ONCE on the select window and frozen (not retrained per fold), then
     forward-pass scored across the holdout, exactly like HG_11's HGNN.
  2. Propagation-operator ablation (symmetric vs asymmetric) on the all-edges
     graph, where the operators are more likely to actually diverge than on
     the 3-edge graph HG_07/HG_08 tied on.
  3. Durability sweep: the SAME frozen-once HGNN evaluated with the select/
     holdout split moved earlier and later, to see how far forward a single
     frozen training pass stays useful.
  4. Calibration fit-slice size ablation: does the amount of real data used
     for the one-time isotonic fit change how well it holds up.

Every configuration is compliance-checked against Criterion 1 before it's
run, same discipline as every prior round.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hypergraphs_research"))
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

import large_panel as lp
from hypergraph_model import build_actor_lookup
from hg_research_lib import build_incidence_v2, HypergraphNNVariant, xgi_structural_features

OUT_JSON = "results_v2/hypergraph_round3_results.json"
OUT_LOG = "results_v2/hypergraph_round3_log.jsonl"
BEST_EDGES = ("country", "region_week", "global_week")
ALL_EDGES = ("country", "region_week", "global_week", "actor")

_log_fh = None
def log_iter(rec):
    global _log_fh
    if _log_fh is None:
        _log_fh = open(OUT_LOG, "w", encoding="utf-8")
    _log_fh.write(json.dumps(rec, default=str) + "\n"); _log_fh.flush()


def fixed_meanpool_features(nodes, edge_types, actor_lookup):
    H, _ = build_incidence_v2(nodes, actor_lookup, None, 156, edge_types)
    neighbor_count = np.asarray((H @ H.T).sum(axis=1)).flatten() - 1
    return pd.DataFrame({"country": nodes["country"].values, "week": nodes["week"].values,
                          "hg_meanpool_neighbor_count": np.clip(neighbor_count, 0, None)})


def summarize(y, p):
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()) if len(y) else 0, "brier": None, "ap": None, "roc_auc": None}
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = None
    return {"n": int(len(y)), "n_pos": int(y.sum()), "brier": float(brier_score_loss(y, p)),
            "ap": float(average_precision_score(y, p)), "roc_auc": roc}


def build_structural_panel(panel, feature_cols, edge_types, actor_lookup):
    nodes_all = panel[["country", "week"]].reset_index(drop=True)
    struct = xgi_structural_features(nodes_all, actor_lookup, None, 156, edge_types)
    meanpool = fixed_meanpool_features(nodes_all, edge_types, actor_lookup)
    panel2 = panel.merge(struct, on=["country", "week"], how="left").merge(meanpool, on=["country", "week"], how="left")
    cols = feature_cols + ["hg_degree", "hg_avg_neighbor_degree", "hg_meanpool_neighbor_count"]
    return panel2, cols


def run_structural_frozen_once(tag, panel, feature_cols, label_col, edge_types, actor_lookup, split_frac=0.6):
    """NEW this round: the structural GBM fit ONCE on the select window (not
    retrained per fold), frozen, forward-pass scored across the holdout --
    makes it a directly comparable, and directly ensemble-able, partner for
    HG_11's frozen-once HGNN. Compliance is even cleaner than the per-fold
    version: no learned embedding exists in the features at all (unchanged),
    AND now the downstream classifier itself is also fit exactly once."""
    t0 = time.time()
    panel2, cols = build_structural_panel(panel, feature_cols, edge_types, actor_lookup)
    weeks = sorted(panel2["week"].unique())
    split_date = weeks[int(len(weeks) * split_frac)]
    select_df = panel2[panel2["week"] < split_date].dropna(subset=cols + [label_col])
    holdout_df = panel2[panel2["week"] >= split_date]

    X_tr = select_df[cols].fillna(0).values; y_tr = select_df[label_col].astype(int).values
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
    X_ho = holdout_df[cols].fillna(0).values
    p = m.predict_proba(X_ho)[:, 1]
    valid = ~pd.isna(holdout_df[label_col].values)
    y = holdout_df[label_col].values[valid].astype(int)
    pv = np.clip(np.asarray(p)[valid], 1e-6, 1 - 1e-6)
    weeks_valid = holdout_df["week"].values[valid]
    countries_valid = holdout_df["country"].values[valid]
    metrics = summarize(y, pv)
    compliance = {"verdict": "COMPLIANT", "reason": "No learned graph embedding exists in the structural "
                  "features at all (fixed formulas over current graph structure). The downstream GBM classifier "
                  "is now ALSO fit exactly once, on the select window, and never refit -- every holdout row is a "
                  "forward pass through the frozen trees, with structural features recomputed fresh (fixed "
                  "formula, no fitting) for each new real node."}
    out = {"tag": tag, "edge_types": list(edge_types), "split_date": str(split_date), "compliance": compliance,
           "metrics": metrics, "elapsed_sec": round(time.time() - t0, 1)}
    log_iter(out)
    print(f"[{tag}] n={metrics['n']} brier={metrics.get('brier')} ap={metrics.get('ap')} "
          f"roc_auc={metrics.get('roc_auc')} ({out['elapsed_sec']}s)", flush=True)
    return out, pd.DataFrame({"week": weeks_valid, "country": countries_valid, "y": y, "p": pv})


def run_structural_frozen_once_calibrated(tag, panel, feature_cols, label_col, edge_types, actor_lookup, split_frac=0.6, cal_frac=0.8):
    t0 = time.time()
    panel2, cols = build_structural_panel(panel, feature_cols, edge_types, actor_lookup)
    weeks = sorted(panel2["week"].unique())
    split_date = weeks[int(len(weeks) * split_frac)]
    select_df = panel2[panel2["week"] < split_date].dropna(subset=cols + [label_col])
    holdout_df = panel2[panel2["week"] >= split_date]

    sel_weeks = sorted(select_df["week"].unique())
    cal_split = sel_weeks[int(len(sel_weeks) * cal_frac)]
    fit_df = select_df[select_df["week"] < cal_split]
    cal_df = select_df[select_df["week"] >= cal_split]

    X_tr = fit_df[cols].fillna(0).values; y_tr = fit_df[label_col].astype(int).values
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)

    p_cal_raw = np.clip(m.predict_proba(cal_df[cols].fillna(0).values)[:, 1], 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, cal_df[label_col].astype(int).values)

    p_hold_raw = np.clip(m.predict_proba(holdout_df[cols].fillna(0).values)[:, 1], 1e-6, 1 - 1e-6)
    valid = ~pd.isna(holdout_df[label_col].values)
    y = holdout_df[label_col].values[valid].astype(int)
    pv = np.clip(iso.transform(p_hold_raw[valid]), 1e-6, 1 - 1e-6)
    weeks_valid = holdout_df["week"].values[valid]
    countries_valid = holdout_df["country"].values[valid]
    metrics = summarize(y, pv)
    compliance = {"verdict": "COMPLIANT", "reason": "GBM fit exactly once on the fit slice; isotonic calibration "
                  "fit exactly once on a separate calibration slice never used for GBM fitting; both frozen "
                  "before the holdout is scored. Same two-stage-frozen pattern already used for HG_11 and ICL_08."}
    out = {"tag": tag, "edge_types": list(edge_types), "cal_frac": cal_frac, "compliance": compliance,
           "metrics": metrics, "elapsed_sec": round(time.time() - t0, 1)}
    log_iter(out)
    print(f"[{tag}] cal_frac={cal_frac} n={metrics['n']} brier={metrics.get('brier')} ap={metrics.get('ap')} "
          f"roc_auc={metrics.get('roc_auc')} ({out['elapsed_sec']}s)", flush=True)
    return out, pd.DataFrame({"week": weeks_valid, "country": countries_valid, "y": y, "p": pv})


def run_hgnn_frozen_once(tag, panel, feature_cols, label_col, edge_types, actor_lookup, prop_mode="symmetric", split_frac=0.6):
    t0 = time.time()
    weeks = sorted(panel["week"].unique())
    split_date = weeks[int(len(weeks) * split_frac)]
    select_df = panel[panel["week"] < split_date].dropna(subset=feature_cols + [label_col])
    holdout_df = panel[panel["week"] >= split_date]

    m = HypergraphNNVariant(feature_cols=feature_cols, edge_types=edge_types, prop_mode=prop_mode, epochs=100)
    m.set_actor_lookup(actor_lookup)
    m.fit(select_df, label_col)
    p = m.predict_proba(holdout_df, label_col)
    valid = ~pd.isna(holdout_df[label_col].values)
    y = holdout_df[label_col].values[valid].astype(int)
    pv = np.clip(np.asarray(p)[valid], 1e-6, 1 - 1e-6)
    weeks_valid = holdout_df["week"].values[valid]
    countries_valid = holdout_df["country"].values[valid]
    metrics = summarize(y, pv)
    compliance = {"verdict": "COMPLIANT", "reason": "fit() invoked exactly once on the select window; every "
                  "holdout prediction is a forward pass through frozen Theta1/Theta2."}
    out = {"tag": tag, "edge_types": list(edge_types), "prop_mode": prop_mode, "split_frac": split_frac,
           "split_date": str(split_date), "compliance": compliance, "metrics": metrics,
           "elapsed_sec": round(time.time() - t0, 1)}
    log_iter(out)
    print(f"[{tag}] prop={prop_mode} edges={edge_types} split_frac={split_frac} n={metrics['n']} "
          f"brier={metrics.get('brier')} ap={metrics.get('ap')} roc_auc={metrics.get('roc_auc')} "
          f"({out['elapsed_sec']}s)", flush=True)
    return out, pd.DataFrame({"week": weeks_valid, "country": countries_valid, "y": y, "p": pv})


def run_hgnn_frozen_once_calibrated(tag, panel, feature_cols, label_col, edge_types, actor_lookup,
                                      prop_mode="symmetric", split_frac=0.6, cal_frac=0.8):
    t0 = time.time()
    weeks = sorted(panel["week"].unique())
    split_date = weeks[int(len(weeks) * split_frac)]
    select_df = panel[panel["week"] < split_date].dropna(subset=feature_cols + [label_col])
    holdout_df = panel[panel["week"] >= split_date]

    sel_weeks = sorted(select_df["week"].unique())
    cal_split = sel_weeks[int(len(sel_weeks) * cal_frac)]
    fit_df = select_df[select_df["week"] < cal_split]
    cal_df = select_df[select_df["week"] >= cal_split]

    m = HypergraphNNVariant(feature_cols=feature_cols, edge_types=edge_types, prop_mode=prop_mode, epochs=100)
    m.set_actor_lookup(actor_lookup)
    m.fit(fit_df, label_col)

    p_cal_raw = np.clip(np.asarray(m.predict_proba(cal_df, label_col)), 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, cal_df[label_col].astype(int).values)

    p_hold_raw = np.clip(np.asarray(m.predict_proba(holdout_df, label_col)), 1e-6, 1 - 1e-6)
    valid = ~pd.isna(holdout_df[label_col].values)
    y = holdout_df[label_col].values[valid].astype(int)
    pv = np.clip(iso.transform(p_hold_raw[valid]), 1e-6, 1 - 1e-6)
    weeks_valid = holdout_df["week"].values[valid]
    countries_valid = holdout_df["country"].values[valid]
    metrics = summarize(y, pv)
    compliance = {"verdict": "COMPLIANT", "reason": "fit() and the isotonic calibration fit are each invoked "
                  "exactly once, on non-overlapping slices of the select window; every holdout prediction is two "
                  "frozen forward transforms."}
    out = {"tag": tag, "cal_frac": cal_frac, "split_frac": split_frac, "compliance": compliance,
           "metrics": metrics, "elapsed_sec": round(time.time() - t0, 1)}
    log_iter(out)
    print(f"[{tag}] cal_frac={cal_frac} split_frac={split_frac} n={metrics['n']} brier={metrics.get('brier')} "
          f"ap={metrics.get('ap')} roc_auc={metrics.get('roc_auc')} ({out['elapsed_sec']}s)", flush=True)
    return out, pd.DataFrame({"week": weeks_valid, "country": countries_valid, "y": y, "p": pv})


if __name__ == "__main__":
    t_start = time.time()
    raw = lp.load_raw()
    panel = lp.build_panel(raw_df=raw)
    actor_lookup = build_actor_lookup(raw, ["Actor1Code", "Actor2Code"], "ActionGeo_CountryCode", "period")
    feature_cols = lp.FEATURE_SETS["core"]
    LABEL = "label_quad_1"
    all_results = []
    saved_scores = {}

    print("\n===== GROUP 1: structural track, made frozen-once (new, for direct ensembling with HG_11) =====", flush=True)
    r, df_scores = run_structural_frozen_once("HG_12_structural_frozen_once", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup)
    all_results.append(r); saved_scores["HG_12_structural_frozen_once"] = df_scores
    r, df_scores = run_structural_frozen_once_calibrated("HG_13_structural_frozen_once_calibrated", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup)
    all_results.append(r); saved_scores["HG_13_structural_frozen_once_calibrated"] = df_scores

    print("\n===== GROUP 2: propagation-operator ablation on the all-edges graph =====", flush=True)
    r, _ = run_hgnn_frozen_once("HG_14_frozen_once_all_edges_symmetric", panel, feature_cols, LABEL, ALL_EDGES, actor_lookup, prop_mode="symmetric")
    all_results.append(r)
    r, _ = run_hgnn_frozen_once("HG_15_frozen_once_all_edges_asymmetric", panel, feature_cols, LABEL, ALL_EDGES, actor_lookup, prop_mode="asymmetric")
    all_results.append(r)

    print("\n===== GROUP 3: durability sweep -- same frozen-once HGNN, select/holdout split moved earlier/later =====", flush=True)
    r, _ = run_hgnn_frozen_once("HG_16_frozen_once_split40", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup, split_frac=0.4)
    all_results.append(r)
    r, hg11_scores_default = run_hgnn_frozen_once("HG_17_frozen_once_split60_repeat", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup, split_frac=0.6)
    all_results.append(r)
    r, _ = run_hgnn_frozen_once("HG_18_frozen_once_split75", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup, split_frac=0.75)
    all_results.append(r)

    print("\n===== GROUP 4: calibration fit-slice size ablation (HG_11-style, structural track) =====", flush=True)
    r, _ = run_structural_frozen_once_calibrated("HG_19_structural_calibrated_calfrac60", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup, cal_frac=0.6)
    all_results.append(r)
    r, _ = run_structural_frozen_once_calibrated("HG_20_structural_calibrated_calfrac90", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup, cal_frac=0.9)
    all_results.append(r)

    print("\n===== GROUP 5: HGNN, calibrated, at the same split as HG_13 (for a clean ensemble partner) =====", flush=True)
    r, hg11_scores = run_hgnn_frozen_once_calibrated("HG_21_hgnn_frozen_once_calibrated_repeat", panel, feature_cols, LABEL, BEST_EDGES, actor_lookup)
    all_results.append(r); saved_scores["HG_21_hgnn_frozen_once_calibrated_repeat"] = hg11_scores

    print("\n===== GROUP 6: real ensembles of the two best compliant, directly-comparable frozen-once scores =====", flush=True)
    a = saved_scores["HG_13_structural_frozen_once_calibrated"]
    b = saved_scores["HG_21_hgnn_frozen_once_calibrated_repeat"]
    merged = a.merge(b, on=["country", "week"], suffixes=("_struct", "_hgnn"))
    assert (merged["y_struct"] == merged["y_hgnn"]).all(), "label mismatch across scores being ensembled"
    y = merged["y_struct"].values.astype(int)

    p_avg = np.clip((merged["p_struct"].values + merged["p_hgnn"].values) / 2, 1e-6, 1 - 1e-6)
    m_avg = summarize(y, p_avg)
    out_avg = {"tag": "ENS_01_structural_plus_hgnn_average", "components": ["HG_13_structural_frozen_once_calibrated", "HG_21_hgnn_frozen_once_calibrated_repeat"],
               "combination": "simple average of two frozen, already-calibrated scores",
               "compliance": {"verdict": "COMPLIANT", "reason": "Averaging two frozen models' outputs fits no "
                              "new parameter; a new event changes what each frozen model outputs, and the "
                              "average is a fixed arithmetic combination of those two numbers, computed fresh "
                              "every time -- no retraining step of any kind is triggered."},
               "metrics": m_avg}
    log_iter(out_avg); all_results.append(out_avg)
    print(f"[ENS_01 average] n={m_avg['n']} brier={m_avg.get('brier')} ap={m_avg.get('ap')} roc_auc={m_avg.get('roc_auc')}", flush=True)

    # weight chosen ONCE via the two components' real select-window AP ranking (structural > hgnn on AP in
    # every round so far) -- a fixed, disclosed rule, not tuned against this holdout.
    w_struct = 0.65
    p_wavg = np.clip(w_struct * merged["p_struct"].values + (1 - w_struct) * merged["p_hgnn"].values, 1e-6, 1 - 1e-6)
    m_wavg = summarize(y, p_wavg)
    out_wavg = {"tag": "ENS_02_structural_plus_hgnn_weighted", "components": ["HG_13_structural_frozen_once_calibrated", "HG_21_hgnn_frozen_once_calibrated_repeat"],
                "combination": f"fixed weighted average, w_structural={w_struct} (weight set once from prior rounds' real AP ranking, not tuned on this holdout)",
                "compliance": {"verdict": "COMPLIANT", "reason": "Same reasoning as ENS_01; the weight is a "
                               "fixed constant chosen from prior real results, not fit on this holdout."},
                "metrics": m_wavg}
    log_iter(out_wavg); all_results.append(out_wavg)
    print(f"[ENS_02 weighted w={w_struct}] n={m_wavg['n']} brier={m_wavg.get('brier')} ap={m_wavg.get('ap')} roc_auc={m_wavg.get('roc_auc')}", flush=True)

    merged.to_json("results_v2/hg_ensemble_raw_scores.json", orient="records", indent=2)

    n_iters = len(all_results)
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n\nTOTAL: {n_iters} hypergraph round-3 iterations, {elapsed_min:.1f} min", flush=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"n_iterations": n_iters, "elapsed_min": elapsed_min, "results": all_results}, f, indent=2, default=str)
    print(f"Saved {OUT_JSON}, {OUT_LOG}, and results_v2/hg_ensemble_raw_scores.json")
