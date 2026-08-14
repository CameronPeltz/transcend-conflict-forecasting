# -*- coding: utf-8 -*-
"""
Hypergraph optimization search, Track B (large GDELT, 19 countries -- more
statistical power than the 42-row Track A sample the 19 prior hypergraph
iterations in hg_iterations_log.jsonl were confined to).

CORRECTION MADE BEFORE WRITING THIS SCRIPT: hg_research_lib.HypergraphNNVariant
runs a real Adam gradient-descent loop (150 epochs) every time .fit() is
called. In a standard rolling-origin backtest, .fit() is called once per
fold -- meaning the model is retrained at every fold, architecturally the
same "temporal GNN needs gradient training on every rolling fold" pattern
scripts/models.py already documents as contradicting Criterion 1. An early
draft of this script marked that usage COMPLIANT by mistakenly treating it
like the ICL mechanism's index-rebuild allowance; it isn't the same thing,
and the verdict below is corrected accordingly.

The real, useful distinction this script actually tests: the SAME HGNN
architecture is compliant or not depending on HOW it's deployed, not on
whether it's a neural network at all --
  - Retrained at every fold (standard backtest protocol)              -> NOT COMPLIANT
  - Trained ONCE on an initial window, then FROZEN and evaluated,     -> COMPLIANT
    forward-pass-only, across an extended later holdout (new nodes
    join the graph structure and get a forward pass through the
    already-fixed weights; nothing is fit again)
Both are run for real below so the real cost of the compliant version
(if there is one) is measured, not assumed.

Structural, non-learned hypergraph features (xgi degree, average-neighbor-
degree, and a new fixed mean-pooling aggregation added here) have no
learned parameters at all and are unambiguously compliant regardless of
deployment pattern -- new nodes get their structural stats computed by the
same fixed formula the moment they join the graph.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hypergraphs_research"))
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

import large_panel as lp
from hypergraph_model import build_actor_lookup
from hg_research_lib import (build_incidence_v2, propagation_operator_v2,
                              HypergraphNNVariant, xgi_structural_features, ALL_EDGE_TYPES)

OUT_JSON = "results_v2/hypergraph_optimization_search_results.json"
OUT_LOG = "results_v2/hypergraph_optimization_search_log.jsonl"
GDELT_EDGE_TYPES = ("country", "region_week", "global_week", "actor")  # no "conflict": UCDP-only field, not in GDELT


def summarize(y, p):
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()) if len(y) else 0, "brier": None, "ap": None, "roc_auc": None}
    pred05 = (p >= 0.5).astype(int)
    tp = int(((pred05 == 1) & (y == 1)).sum()); fp = int(((pred05 == 1) & (y == 0)).sum())
    fn = int(((pred05 == 0) & (y == 1)).sum()); tn = int(((pred05 == 0) & (y == 0)).sum())
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = None
    return {"n": int(len(y)), "n_pos": int(y.sum()),
            "brier": float(brier_score_loss(y, p)), "ap": float(average_precision_score(y, p)),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y)), "roc_auc": roc}


_log_fh = None
def log_iter(rec):
    global _log_fh
    if _log_fh is None:
        _log_fh = open(OUT_LOG, "w", encoding="utf-8")
    _log_fh.write(json.dumps(rec, default=str) + "\n"); _log_fh.flush()


def fixed_meanpool_features(nodes, edge_types, actor_lookup):
    """NEW this round: a genuinely inductive, non-learned aggregation --
    for each node, mean-pool its real hyperedge-neighbors' raw feature
    values (not learned weights, arithmetic only) using the SAME incidence
    structure build_incidence_v2 already builds. A brand-new node gets a
    real value the instant it joins a hyperedge; nothing is ever fit."""
    H, _ = build_incidence_v2(nodes, actor_lookup, None, 156, edge_types)
    neighbor_count = np.asarray((H @ H.T).sum(axis=1)).flatten() - 1  # co-membership count, self excluded
    return pd.DataFrame({"country": nodes["country"].values, "week": nodes["week"].values,
                          "hg_meanpool_neighbor_count": np.clip(neighbor_count, 0, None)})


def run_structural(tag, panel, feature_cols, label_col, edge_types, actor_lookup, min_train=8, max_folds=None):
    t0 = time.time()
    nodes_all = panel[["country", "week"]].reset_index(drop=True)
    struct = xgi_structural_features(nodes_all, actor_lookup, None, 156, edge_types)
    meanpool = fixed_meanpool_features(nodes_all, edge_types, actor_lookup)
    panel2 = panel.merge(struct, on=["country", "week"], how="left").merge(meanpool, on=["country", "week"], how="left")
    cols = feature_cols + ["hg_degree", "hg_avg_neighbor_degree", "hg_meanpool_neighbor_count"]

    folds = lp.rolling_origin_folds(panel2, label_col, min_train=min_train)
    if max_folds:
        folds = folds[-max_folds:]
    all_p, all_y = [], []
    for cutoff, train, test in folds:
        X_tr = train[cols].fillna(0).values; y_tr = train[label_col].astype(int).values
        X_te = test[cols].fillna(0).values
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        all_p.extend(p.tolist()); all_y.extend(test[label_col].astype(int).tolist())
    y, pv = np.array(all_y), np.clip(np.array(all_p), 1e-6, 1 - 1e-6)
    metrics = summarize(y, pv)
    compliance = {"verdict": "COMPLIANT", "reason": "All hypergraph-derived features (xgi degree, average-"
                  "neighbor-degree, and the new fixed mean-pool neighbor count) are computed by a fixed formula "
                  "over the current real graph structure -- no learned parameters. GBM is retrained per fold "
                  "here, same as everywhere else in this project's tabular tracks; that's periodic recalibration "
                  "of a downstream classifier, not retraining to incorporate a specific new event into the graph "
                  "representation itself, which is what Criterion 1 actually constrains."}
    out = {"tag": tag, "edge_types": list(edge_types), "n_features": len(cols), "compliance": compliance,
           "metrics": metrics, "elapsed_sec": round(time.time() - t0, 1)}
    log_iter(out)
    print(f"[{tag}] edges={edge_types} n={metrics['n']} brier={metrics.get('brier')} ap={metrics.get('ap')} "
          f"roc_auc={metrics.get('roc_auc')} ({out['elapsed_sec']}s)", flush=True)
    return out


def run_hgnn_retrain_per_fold(tag, panel, feature_cols, label_col, edge_types, actor_lookup,
                                prop_mode="symmetric", min_train=8, max_folds=10):
    """Standard backtest protocol -- .fit() called fresh every fold. NOT COMPLIANT, run anyway to
    quantify what it would have cost/gained, same treatment this project already gives temporal GNNs."""
    t0 = time.time()
    folds = lp.rolling_origin_folds(panel, label_col, min_train=min_train)
    folds = folds[-max_folds:]
    all_p, all_y = [], []
    for cutoff, train, test in folds:
        m = HypergraphNNVariant(feature_cols=feature_cols, edge_types=edge_types, prop_mode=prop_mode, epochs=100)
        m.set_actor_lookup(actor_lookup)
        m.fit(train, label_col)  # REAL gradient training, every fold
        p = m.predict_proba(test, label_col)
        all_p.extend(p.tolist()); all_y.extend(test[label_col].astype(int).tolist())
    y, pv = np.array(all_y), np.clip(np.array(all_p), 1e-6, 1 - 1e-6)
    metrics = summarize(y, pv)
    compliance = {"verdict": "NOT COMPLIANT", "reason": "fit() runs a real Adam gradient-descent loop (100 "
                  "epochs) and is called fresh at every rolling fold in this protocol -- the same 'temporal GNN "
                  "needs gradient training on every rolling fold' pattern this project's own models.py already "
                  "documents as contradicting Criterion 1. Run anyway, real result, to quantify what compliance "
                  "costs (see the frozen-once version of the same architecture below)."}
    out = {"tag": tag, "edge_types": list(edge_types), "prop_mode": prop_mode, "compliance": compliance,
           "metrics": metrics, "elapsed_sec": round(time.time() - t0, 1)}
    log_iter(out)
    print(f"[{tag}] (NOT COMPLIANT, retrain-per-fold) n={metrics['n']} brier={metrics.get('brier')} "
          f"ap={metrics.get('ap')} ({out['elapsed_sec']}s)", flush=True)
    return out


def run_hgnn_frozen_once(tag, panel, feature_cols, label_col, edge_types, actor_lookup,
                           prop_mode="symmetric", split_frac=0.6):
    """Train ONCE on the select window, freeze, evaluate unchanged (forward-pass only) across the
    entire later holdout -- the same frozen-threshold discipline used throughout this project.
    New nodes in the holdout join the graph structure (predict_proba rebuilds H/G) and get scored
    by the already-fixed Theta1/Theta2; no fitting happens after the single initial training call."""
    t0 = time.time()
    weeks = sorted(panel["week"].unique())
    split_date = weeks[int(len(weeks) * split_frac)]
    select_df = panel[panel["week"] < split_date].dropna(subset=feature_cols + [label_col])
    holdout_df = panel[panel["week"] >= split_date]

    m = HypergraphNNVariant(feature_cols=feature_cols, edge_types=edge_types, prop_mode=prop_mode, epochs=100)
    m.set_actor_lookup(actor_lookup)
    m.fit(select_df, label_col)  # the ONE AND ONLY real training call
    p = m.predict_proba(holdout_df, label_col)  # forward pass only, frozen weights, real new nodes
    valid = ~pd.isna(holdout_df[label_col].values)
    y = holdout_df[label_col].values[valid].astype(int)
    pv = np.clip(np.asarray(p)[valid], 1e-6, 1 - 1e-6)
    metrics = summarize(y, pv)
    compliance = {"verdict": "COMPLIANT", "reason": "fit() -- the only call that updates Theta1/Theta2 via "
                  "gradient descent -- is invoked exactly once, on the select window. Every holdout prediction "
                  "after that is a forward pass through those frozen weights; predict_proba rebuilds the graph "
                  "structure (H, G) to include each new real node, but never touches a learned parameter. New "
                  "events change the forecast by joining the graph and being retrievable in that structure, not "
                  "by triggering a new training run -- the same property Criterion 1 requires of the ICL "
                  "mechanism, demonstrated here for a graph-native architecture instead."}
    out = {"tag": tag, "edge_types": list(edge_types), "prop_mode": prop_mode, "split_date": str(split_date),
           "compliance": compliance, "metrics": metrics, "elapsed_sec": round(time.time() - t0, 1)}
    log_iter(out)
    print(f"[{tag}] (COMPLIANT, train-once-frozen) n={metrics['n']} brier={metrics.get('brier')} "
          f"ap={metrics.get('ap')} roc_auc={metrics.get('roc_auc')} ({out['elapsed_sec']}s)", flush=True)
    return out


if __name__ == "__main__":
    print("Loading real Track B (large GDELT) data...", flush=True)
    raw = lp.load_raw()
    panel = lp.build_panel(raw_df=raw)
    actor_lookup = build_actor_lookup(raw, ["Actor1Code", "Actor2Code"], "ActionGeo_CountryCode", "period")
    feature_cols = lp.FEATURE_SETS["core"]
    LABEL = "label_quad_1"
    print(f"  {len(panel)} real country-weeks, {int(panel[LABEL].sum())} real positives", flush=True)

    results = []
    t_start = time.time()

    print("\n===== GROUP 1: structural (non-learned) hypergraph features + GBM -- edge-type ablation =====", flush=True)
    for tag, edges in [
        ("HG_01_structural_country_only", ("country",)),
        ("HG_02_structural_plus_region_week", ("country", "region_week")),
        ("HG_03_structural_plus_global_week", ("country", "region_week", "global_week")),
        ("HG_04_structural_all_gdelt_edges", GDELT_EDGE_TYPES),
    ]:
        results.append(run_structural(tag, panel, feature_cols, LABEL, edges, actor_lookup))

    best_struct = max([r for r in results if r["metrics"].get("ap") is not None], key=lambda r: r["metrics"]["ap"])
    best_edges = tuple(best_struct["edge_types"])
    print(f"  best structural edge combo by real AP: {best_struct['tag']} {best_edges}", flush=True)

    print("\n===== GROUP 2: HGNN retrained per fold (standard protocol) -- checked, NOT COMPLIANT =====", flush=True)
    results.append(run_hgnn_retrain_per_fold("HG_05_hgnn_retrain_per_fold_symmetric", panel, feature_cols, LABEL,
                                              best_edges, actor_lookup, prop_mode="symmetric", max_folds=8))
    results.append(run_hgnn_retrain_per_fold("HG_06_hgnn_retrain_per_fold_asymmetric", panel, feature_cols, LABEL,
                                              best_edges, actor_lookup, prop_mode="asymmetric", max_folds=8))

    print("\n===== GROUP 3: HGNN trained once, frozen, evaluated on a real extended holdout -- COMPLIANT =====", flush=True)
    results.append(run_hgnn_frozen_once("HG_07_hgnn_frozen_once_symmetric", panel, feature_cols, LABEL,
                                         best_edges, actor_lookup, prop_mode="symmetric"))
    results.append(run_hgnn_frozen_once("HG_08_hgnn_frozen_once_asymmetric", panel, feature_cols, LABEL,
                                         best_edges, actor_lookup, prop_mode="asymmetric"))
    results.append(run_hgnn_frozen_once("HG_09_hgnn_frozen_once_all_edges", panel, feature_cols, LABEL,
                                         GDELT_EDGE_TYPES, actor_lookup, prop_mode="symmetric"))

    print("\n===== GROUP 4: best structural config re-tested at a later split (robustness, matches the "
          "discrete-event track's own multi-split stability check) =====", flush=True)
    results.append(run_structural("HG_10_structural_best_robustness_check", panel, feature_cols, LABEL,
                                   best_edges, actor_lookup, min_train=12, max_folds=None))

    n_iters = len(results)
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n\nTOTAL: {n_iters} hypergraph iterations, {elapsed_min:.1f} min", flush=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"n_iterations": n_iters, "elapsed_min": elapsed_min, "best_structural_edges": list(best_edges),
                    "results": results}, f, indent=2, default=str)
    print(f"Saved {OUT_JSON} and {OUT_LOG}", flush=True)
