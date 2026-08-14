# -*- coding: utf-8 -*-
"""
One additional real iteration (HG_11), appended to the same results/log files
hypergraph_compliance_search.py wrote. Two reasons this one more run earns its
place rather than just padding a count to 20:

1. It mirrors ICL_08 (one-time frozen calibration on top of the best-so-far
   COMPLIANT config) so both search tracks end on a directly comparable
   capstone: does a single, one-time calibration step -- fit once, same as
   the model weights it's calibrating, never refit on new events -- recover
   any of the real precision the frozen-once HGNN gave up relative to the
   NOT-COMPLIANT retrain-per-fold version (HG_05/06)?
2. It's a real test of whether calibration itself is compliant for a graph
   model the same way it already was shown to be for the ICL mechanism:
   IsotonicRegression.fit() is called exactly once, on the select window's
   own frozen-model outputs, then .transform() (no further fitting) is
   applied to the holdout -- so it inherits the frozen-once HGNN's
   compliance rather than reopening the question.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hypergraphs_research"))
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

import large_panel as lp
from hypergraph_model import build_actor_lookup
from hg_research_lib import HypergraphNNVariant

OUT_JSON = "results_v2/hypergraph_optimization_search_results.json"
OUT_LOG = "results_v2/hypergraph_optimization_search_log.jsonl"
BEST_EDGES = ("country", "region_week", "global_week")  # HG_03 / HG_10 winner


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


if __name__ == "__main__":
    t0 = time.time()
    raw = lp.load_raw()
    panel = lp.build_panel(raw_df=raw)
    actor_lookup = build_actor_lookup(raw, ["Actor1Code", "Actor2Code"], "ActionGeo_CountryCode", "period")
    feature_cols = lp.FEATURE_SETS["core"]
    LABEL = "label_quad_1"

    weeks = sorted(panel["week"].unique())
    split_date = weeks[int(len(weeks) * 0.6)]
    select_df = panel[panel["week"] < split_date].dropna(subset=feature_cols + [LABEL])
    holdout_df = panel[panel["week"] >= split_date]

    # carve the select window itself into a calibration-fit slice (first 80%) and a
    # calibration-check slice (last 20%) so IsotonicRegression is never fit on data
    # it is then evaluated against -- same discipline as the rest of this project.
    sel_weeks = sorted(select_df["week"].unique())
    cal_split = sel_weeks[int(len(sel_weeks) * 0.8)]
    fit_df = select_df[select_df["week"] < cal_split]
    cal_df = select_df[select_df["week"] >= cal_split]

    m = HypergraphNNVariant(feature_cols=feature_cols, edge_types=BEST_EDGES, prop_mode="symmetric", epochs=100)
    m.set_actor_lookup(actor_lookup)
    m.fit(fit_df, LABEL)  # the ONE real gradient-training call

    p_cal_raw = np.clip(np.asarray(m.predict_proba(cal_df, LABEL)), 1e-6, 1 - 1e-6)
    y_cal = cal_df[LABEL].astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, y_cal)  # the ONE real calibration-fitting call

    p_hold_raw = np.clip(np.asarray(m.predict_proba(holdout_df, LABEL)), 1e-6, 1 - 1e-6)
    valid = ~pd.isna(holdout_df[LABEL].values)
    y = holdout_df[LABEL].values[valid].astype(int)
    p_hold_calibrated = np.clip(iso.transform(p_hold_raw[valid]), 1e-6, 1 - 1e-6)

    metrics = summarize(y, p_hold_calibrated)
    compliance = {"verdict": "COMPLIANT", "reason": "Two real fitting calls happen, both exactly once, both "
                  "before the holdout is ever scored: HypergraphNNVariant.fit() sets Theta1/Theta2 once on the "
                  "fit slice, and IsotonicRegression.fit() sets the calibration curve once on a separate "
                  "calibration slice of held-back select-window data. Every holdout prediction after that is "
                  "two frozen forward transforms (frozen graph weights, then the frozen calibration curve) -- no "
                  "parameter of either step is ever refit in response to a new event. This is the same pattern "
                  "already used for ICL_08's one-time calibration; shown here to hold for a graph-native "
                  "architecture too."}
    out = {"tag": "HG_11_frozen_once_with_calibration", "edge_types": list(BEST_EDGES), "prop_mode": "symmetric",
           "split_date": str(split_date), "cal_split_date": str(cal_split), "compliance": compliance,
           "metrics": metrics, "elapsed_sec": round(time.time() - t0, 1)}

    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, default=str) + "\n")

    with open(OUT_JSON, "r", encoding="utf-8") as f:
        blob = json.load(f)
    blob["results"].append(out)
    blob["n_iterations"] = len(blob["results"])
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str)

    print(f"[HG_11_frozen_once_with_calibration] n={metrics['n']} brier={metrics.get('brier')} "
          f"ap={metrics.get('ap')} roc_auc={metrics.get('roc_auc')} ({out['elapsed_sec']}s)")
    print(f"Appended to {OUT_JSON} and {OUT_LOG}; total iterations now {blob['n_iterations']}")
