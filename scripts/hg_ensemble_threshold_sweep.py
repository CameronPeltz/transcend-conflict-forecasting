# -*- coding: utf-8 -*-
"""
Real frozen-threshold precision/recall sweep for the ENS_01 hypergraph
ensemble (structural-frozen-once-calibrated + HGNN-frozen-once-calibrated,
simple average), same select/holdout discipline as hg03_pr_curve_analysis.py,
to see whether ensembling -- not just threshold choice -- moves the real
precision/recall tradeoff versus any single compliant hypergraph component.
"""
import json
import numpy as np
import pandas as pd

def metrics_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"threshold": float(thr), "n_flagged": int(pred.sum()),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

if __name__ == "__main__":
    df = pd.read_json("results_v2/hg_ensemble_raw_scores.json", orient="records")
    assert (df["y_struct"] == df["y_hgnn"]).all()
    df["y"] = df["y_struct"]
    df["p_avg"] = (df["p_struct"] + df["p_hgnn"]) / 2
    df = df.sort_values("week").reset_index(drop=True)

    # this holdout (1191 rows) IS the same holdout hg03_pr_curve_analysis.py's earlier stages already selected
    # a threshold on separately -- for the ensemble, select threshold on the earlier 60% of THIS holdout,
    # evaluate on the later 40%, since the ensemble itself was only defined after both components were frozen.
    split_idx = int(len(df) * 0.6)
    select, holdout = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"select n={len(select)} pos={int(select['y'].sum())}; holdout n={len(holdout)} pos={int(holdout['y'].sum())}")

    candidates = np.unique(select["p_avg"].values)[::-1]
    results = []
    for target in [0.80, 0.50, 0.30, 0.25, 0.20, 0.15]:
        best_thr, best_gap, fallback = None, 1e9, candidates[-1]
        for thr in candidates:
            m = metrics_at_threshold(select["y"].values, select["p_avg"].values, thr)
            if m["n_flagged"] == 0:
                continue
            if m["precision"] >= target and (best_thr is None or thr < best_thr):
                best_thr = thr
            gap = abs(m["precision"] - target)
            if gap < best_gap:
                best_gap = gap; fallback = thr
        thr = best_thr if best_thr is not None else fallback
        reached = best_thr is not None
        r = metrics_at_threshold(holdout["y"].values, holdout["p_avg"].values, thr)
        r["target_precision"] = target; r["reached_on_select"] = reached
        results.append(r)
        print(f"  target {target:.2f} (reached: {reached}) thr={thr:.4f} -> HOLDOUT precision={r['precision']:.3f} "
              f"recall={r['recall']:.3f} flagged={r['n_flagged']} tp={r['tp']} fp={r['fp']}")

    with open("results_v2/hg_ensemble_pr_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump({"select_n": len(select), "holdout_n": len(holdout), "sweep": results}, f, indent=2, default=str)
    print("Saved results_v2/hg_ensemble_pr_sweep_results.json")
