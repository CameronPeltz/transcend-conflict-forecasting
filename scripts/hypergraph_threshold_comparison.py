# -*- coding: utf-8 -*-
"""
Real precision/recall/specificity for the two best COMPLIANT hypergraph
configs from this search (HG_03 structural, HG_11 frozen-once+calibrated),
using the SAME frozen-threshold discipline Track A/B's own headline numbers
already use: pick a threshold on an earlier real window, freeze it, apply it
UNCHANGED to a later real holdout. Nothing here is threshold-tuned on the
data it's then scored against.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hypergraphs_research"))
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

import large_panel as lp
from hypergraph_model import build_actor_lookup
from hg_research_lib import build_incidence_v2, HypergraphNNVariant, xgi_structural_features

OUT_JSON = "results_v2/hypergraph_threshold_comparison_results.json"
BEST_EDGES = ("country", "region_week", "global_week")


def fixed_meanpool_features(nodes, edge_types, actor_lookup):
    H, _ = build_incidence_v2(nodes, actor_lookup, None, 156, edge_types)
    neighbor_count = np.asarray((H @ H.T).sum(axis=1)).flatten() - 1
    return pd.DataFrame({"country": nodes["country"].values, "week": nodes["week"].values,
                          "hg_meanpool_neighbor_count": np.clip(neighbor_count, 0, None)})


def metrics_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"threshold": float(thr), "n": int(len(y)), "n_pos": int(y.sum()), "n_flagged": int(pred.sum()),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def select_frozen_threshold(y_select, p_select, target_precision=0.80):
    """Same rule Track A/B's headline numbers use: the lowest threshold that reaches
    the target precision on the SELECT window; if none reaches it, take the threshold
    that gets closest (highest precision achievable), disclosed either way."""
    order = np.argsort(-p_select)
    candidates = np.unique(p_select)[::-1]
    best_thr, best_prec, best_gap = None, -1, 1e9
    for thr in candidates:
        m = metrics_at_threshold(y_select, p_select, thr)
        if m["n_flagged"] == 0:
            continue
        if m["precision"] >= target_precision and (best_thr is None or thr < best_thr):
            best_thr = thr
        gap = abs(m["precision"] - target_precision)
        if gap < best_gap:
            best_gap = gap; best_prec = m["precision"]; fallback_thr = thr
    if best_thr is not None:
        return float(best_thr), True
    return float(fallback_thr), False


def run_hg03_structural_frozen_threshold(panel, feature_cols, label_col, actor_lookup):
    t0 = time.time()
    nodes_all = panel[["country", "week"]].reset_index(drop=True)
    struct = xgi_structural_features(nodes_all, actor_lookup, None, 156, BEST_EDGES)
    meanpool = fixed_meanpool_features(nodes_all, BEST_EDGES, actor_lookup)
    panel2 = panel.merge(struct, on=["country", "week"], how="left").merge(meanpool, on=["country", "week"], how="left")
    cols = feature_cols + ["hg_degree", "hg_avg_neighbor_degree", "hg_meanpool_neighbor_count"]

    folds = lp.rolling_origin_folds(panel2, label_col, min_train=8)
    rows = []
    for cutoff, train, test in folds:
        X_tr = train[cols].fillna(0).values; y_tr = train[label_col].astype(int).values
        X_te = test[cols].fillna(0).values
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        for w, yv, pv in zip(test["week"].values, test[label_col].astype(int).values, p):
            rows.append({"week": w, "y": int(yv), "p": float(pv)})
    df = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)

    split_idx = int(len(df) * 0.6)
    select, holdout = df.iloc[:split_idx], df.iloc[split_idx:]
    thr, reached_target = select_frozen_threshold(select["y"].values, select["p"].values)
    result = metrics_at_threshold(holdout["y"].values, holdout["p"].values, thr)
    result.update({"tag": "HG_03_structural_frozen_threshold", "select_n": len(select), "holdout_n": len(holdout),
                    "reached_80pct_target_on_select": reached_target,
                    "select_split_week": str(select["week"].max()), "elapsed_sec": round(time.time() - t0, 1)})
    print(f"[HG_03 frozen-threshold] thr={thr:.4f} (reached 80% target on select: {reached_target}) "
          f"holdout n={result['n']} pos={result['n_pos']} flagged={result['n_flagged']} "
          f"precision={result['precision']:.3f} recall={result['recall']:.3f} "
          f"specificity={result['specificity']:.3f} accuracy={result['accuracy']:.3f} ({result['elapsed_sec']}s)")
    return result


def run_hg11_frozen_calibrated_threshold(panel, feature_cols, label_col, actor_lookup):
    t0 = time.time()
    weeks = sorted(panel["week"].unique())
    split_date = weeks[int(len(weeks) * 0.6)]
    select_df = panel[panel["week"] < split_date].dropna(subset=feature_cols + [label_col])
    holdout_df = panel[panel["week"] >= split_date]

    sel_weeks = sorted(select_df["week"].unique())
    cal_split = sel_weeks[int(len(sel_weeks) * 0.8)]
    fit_df = select_df[select_df["week"] < cal_split]
    cal_df = select_df[select_df["week"] >= cal_split]

    m = HypergraphNNVariant(feature_cols=feature_cols, edge_types=BEST_EDGES, prop_mode="symmetric", epochs=100)
    m.set_actor_lookup(actor_lookup)
    m.fit(fit_df, label_col)

    p_cal_raw = np.clip(np.asarray(m.predict_proba(cal_df, label_col)), 1e-6, 1 - 1e-6)
    y_cal = cal_df[label_col].astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, y_cal)

    p_hold_raw = np.clip(np.asarray(m.predict_proba(holdout_df, label_col)), 1e-6, 1 - 1e-6)
    valid = ~pd.isna(holdout_df[label_col].values)
    hold = pd.DataFrame({"week": holdout_df["week"].values[valid],
                          "y": holdout_df[label_col].values[valid].astype(int),
                          "p": np.clip(iso.transform(p_hold_raw[valid]), 1e-6, 1 - 1e-6)}).sort_values("week").reset_index(drop=True)

    # threshold selected on the EARLY half of the holdout (never touched by fit() or iso.fit()),
    # frozen, applied to the LATE half -- a third independent chronological slice.
    split_idx = int(len(hold) * 0.5)
    thr_select, thr_eval = hold.iloc[:split_idx], hold.iloc[split_idx:]
    thr, reached_target = select_frozen_threshold(thr_select["y"].values, thr_select["p"].values)
    result = metrics_at_threshold(thr_eval["y"].values, thr_eval["p"].values, thr)
    result.update({"tag": "HG_11_frozen_calibrated_threshold", "select_n": len(thr_select), "holdout_n": len(thr_eval),
                    "reached_80pct_target_on_select": reached_target, "elapsed_sec": round(time.time() - t0, 1)})
    print(f"[HG_11 frozen-threshold] thr={thr:.4f} (reached 80% target on select: {reached_target}) "
          f"eval n={result['n']} pos={result['n_pos']} flagged={result['n_flagged']} "
          f"precision={result['precision']:.3f} recall={result['recall']:.3f} "
          f"specificity={result['specificity']:.3f} accuracy={result['accuracy']:.3f} ({result['elapsed_sec']}s)")
    return result


if __name__ == "__main__":
    raw = lp.load_raw()
    panel = lp.build_panel(raw_df=raw)
    actor_lookup = build_actor_lookup(raw, ["Actor1Code", "Actor2Code"], "ActionGeo_CountryCode", "period")
    feature_cols = lp.FEATURE_SETS["core"]
    LABEL = "label_quad_1"

    r1 = run_hg03_structural_frozen_threshold(panel, feature_cols, LABEL, actor_lookup)
    r2 = run_hg11_frozen_calibrated_threshold(panel, feature_cols, LABEL, actor_lookup)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": [r1, r2]}, f, indent=2, default=str)
    print(f"Saved {OUT_JSON}")
