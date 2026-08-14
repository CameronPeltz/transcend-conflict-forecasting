# -*- coding: utf-8 -*-
"""
Real precision-recall tradeoff curve for HG_03 (structural hypergraph features
+ GBM), same frozen select/holdout split as hypergraph_threshold_comparison.py,
to answer a direct question: the 80%-precision-target threshold gave 1.7%
recall -- how much would recall improve at more realistic precision targets,
using the SAME frozen threshold discipline (selected on the select window,
applied unchanged to the holdout)?
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hypergraphs_research"))
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

import large_panel as lp
from hypergraph_model import build_actor_lookup
from hg_research_lib import build_incidence_v2, xgi_structural_features

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
    return {"threshold": float(thr), "n_flagged": int(pred.sum()),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


if __name__ == "__main__":
    t0 = time.time()
    raw = lp.load_raw()
    panel = lp.build_panel(raw_df=raw)
    actor_lookup = build_actor_lookup(raw, ["Actor1Code", "Actor2Code"], "ActionGeo_CountryCode", "period")
    feature_cols = lp.FEATURE_SETS["core"]
    LABEL = "label_quad_1"

    nodes_all = panel[["country", "week"]].reset_index(drop=True)
    struct = xgi_structural_features(nodes_all, actor_lookup, None, 156, BEST_EDGES)
    meanpool = fixed_meanpool_features(nodes_all, BEST_EDGES, actor_lookup)
    panel2 = panel.merge(struct, on=["country", "week"], how="left").merge(meanpool, on=["country", "week"], how="left")
    cols = feature_cols + ["hg_degree", "hg_avg_neighbor_degree", "hg_meanpool_neighbor_count"]

    folds = lp.rolling_origin_folds(panel2, LABEL, min_train=8)
    rows = []
    for cutoff, train, test in folds:
        X_tr = train[cols].fillna(0).values; y_tr = train[LABEL].astype(int).values
        X_te = test[cols].fillna(0).values
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        for w, yv, pv in zip(test["week"].values, test[LABEL].astype(int).values, p):
            rows.append({"week": w, "y": int(yv), "p": float(pv)})
    df = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)

    split_idx = int(len(df) * 0.6)
    select, holdout = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"select n={len(select)} pos={int(select['y'].sum())}; holdout n={len(holdout)} pos={int(holdout['y'].sum())}")

    # real score distribution on the select window -- what does the GBM's own
    # confidence actually look like, before any threshold is chosen?
    print("\nSelect-window real score distribution (percentiles):")
    for q in [0.5, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]:
        print(f"  p{int(q*100)}: {np.quantile(select['p'].values, q):.4f}")
    print(f"  max score among real select positives: {select.loc[select['y']==1, 'p'].max():.4f}")
    print(f"  mean score, select positives vs negatives: {select.loc[select['y']==1,'p'].mean():.4f} vs {select.loc[select['y']==0,'p'].mean():.4f}")

    # sweep real precision targets, freeze threshold on select, evaluate on holdout
    print("\nFrozen-threshold sweep (selected on real select window, applied unchanged to real holdout):")
    candidates = np.unique(select["p"].values)[::-1]
    results = []
    for target in [0.80, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15]:
        best_thr, best_gap, fallback = None, 1e9, candidates[-1]
        for thr in candidates:
            m = metrics_at_threshold(select["y"].values, select["p"].values, thr)
            if m["n_flagged"] == 0:
                continue
            if m["precision"] >= target and (best_thr is None or thr < best_thr):
                best_thr = thr
            gap = abs(m["precision"] - target)
            if gap < best_gap:
                best_gap = gap; fallback = thr
        thr = best_thr if best_thr is not None else fallback
        reached = best_thr is not None
        r = metrics_at_threshold(holdout["y"].values, holdout["p"].values, thr)
        r["target_precision"] = target
        r["reached_on_select"] = reached
        results.append(r)
        print(f"  target precision {target:.2f} (reached on select: {reached}) -> thr={thr:.4f} | "
              f"HOLDOUT precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"flagged={r['n_flagged']} tp={r['tp']} fp={r['fp']} fn={r['fn']} ({time.time()-t0:.0f}s)")

    with open("results_v2/hg03_pr_curve_results.json", "w", encoding="utf-8") as f:
        json.dump({"select_n": len(select), "select_pos": int(select["y"].sum()),
                    "holdout_n": len(holdout), "holdout_pos": int(holdout["y"].sum()),
                    "sweep": results}, f, indent=2, default=str)
    print("\nSaved results_v2/hg03_pr_curve_results.json")
