# -*- coding: utf-8 -*-
"""
Real frozen-threshold precision/recall for the CURRENT most promising
compliant approaches (post round-3): structural-frozen-once-calibrated
alone, HGNN-frozen-once-calibrated alone (both already-computed per-row
scores, no new compute), the ENS_01 average ensemble (already computed),
and ICL_11 (chain-of-thought + self-consistency, round-3's best-balanced
ICL config, using its already-computed real per-row predictions -- no new
LLM calls). Same select-then-freeze discipline used everywhere else.
"""
import json
import numpy as np
import pandas as pd

def metrics_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"threshold": float(thr), "n": int(len(y)), "n_pos": int(y.sum()), "n_flagged": int(pred.sum()),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def select_frozen_threshold(y_select, p_select, target_precision=0.80):
    candidates = np.unique(p_select)[::-1]
    best_thr, best_gap, fallback = None, 1e9, candidates[-1]
    for thr in candidates:
        m = metrics_at_threshold(y_select, p_select, thr)
        if m["n_flagged"] == 0:
            continue
        if m["precision"] >= target_precision and (best_thr is None or thr < best_thr):
            best_thr = thr
        gap = abs(m["precision"] - target_precision)
        if gap < best_gap:
            best_gap = gap; fallback = thr
    thr = best_thr if best_thr is not None else fallback
    return float(thr), best_thr is not None

results = {}

# --- HG_13 structural alone, HG_21 HGNN alone (same select/holdout split ENS_01 used) ---
hg = pd.read_json("results_v2/hg_ensemble_raw_scores.json", orient="records").sort_values("week").reset_index(drop=True)
split_idx = int(len(hg) * 0.6)
select, holdout = hg.iloc[:split_idx], hg.iloc[split_idx:]

for label, col in [("HG_13_structural_alone", "p_struct"), ("HG_21_hgnn_alone", "p_hgnn")]:
    thr, reached = select_frozen_threshold(select["y_struct"].values, select[col].values)
    r = metrics_at_threshold(holdout["y_struct"].values, holdout[col].values, thr)
    r["reached_80pct_on_select"] = reached
    results[label] = r
    print(f"[{label}] thr={thr:.4f} (reached 80%: {reached}) n={r['n']} pos={r['n_pos']} flagged={r['n_flagged']} "
          f"precision={r['precision']:.3f} recall={r['recall']:.3f} specificity={r['specificity']:.3f} accuracy={r['accuracy']:.3f}")

# --- ENS_01 average, same split (re-derive for a directly side-by-side number) ---
hg["p_avg"] = (hg["p_struct"] + hg["p_hgnn"]) / 2
select, holdout = hg.iloc[:split_idx], hg.iloc[split_idx:]
thr, reached = select_frozen_threshold(select["y_struct"].values, select["p_avg"].values)
r = metrics_at_threshold(holdout["y_struct"].values, holdout["p_avg"].values, thr)
r["reached_80pct_on_select"] = reached
results["ENS_01_average"] = r
print(f"[ENS_01_average] thr={thr:.4f} (reached 80%: {reached}) n={r['n']} pos={r['n_pos']} flagged={r['n_flagged']} "
      f"precision={r['precision']:.3f} recall={r['recall']:.3f} specificity={r['specificity']:.3f} accuracy={r['accuracy']:.3f}")

# --- ICL_11 (round-3 combo config), real per-row predictions already on disk, no new LLM calls ---
icl = pd.read_json("results_v2/icl_round3_ICL_11_cot_plus_self_consistency_raw.json", orient="records").sort_values("week").reset_index(drop=True)
cal_split = int(len(icl) * 0.7)
select, holdout = icl.iloc[:cal_split], icl.iloc[cal_split:]
thr, reached = select_frozen_threshold(select["y"].values, select["p"].values)
r = metrics_at_threshold(holdout["y"].values, holdout["p"].values, thr)
r["reached_80pct_on_select"] = reached
results["ICL_11_combo"] = r
print(f"[ICL_11_combo] thr={thr:.4f} (reached 80%: {reached}) n={r['n']} pos={r['n_pos']} flagged={r['n_flagged']} "
      f"precision={r['precision']:.3f} recall={r['recall']:.3f} specificity={r['specificity']:.3f} accuracy={r['accuracy']:.3f}")

with open("results_v2/round3_threshold_summary_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved results_v2/round3_threshold_summary_results.json")
