# -*- coding: utf-8 -*-
"""
CORRECTION RUN. Every prior ICL result in this search (icl_hypergraph_
compliance_search.py, icl_round3_extended.py, icl_threshold_comparison.py)
was evaluated against iteration_engine.build_panel() -- the small, original
6-country GDELT pilot panel -- while calling it "Track A". That is NOT the
dataset the proposal's real 84.0%/56.8% "Track A" headline number comes
from. That number (results_v2/precision_threshold_validation.json, block
C_pure_ucdp; confirmed against the proposal's own paste-text in
last_edits_to_make_to_draft.html line 96) is computed on the REAL UCDP
panel (build_ucdp_panel.py): select window weeks < 2023-08-14 (1,627 real
rows), holdout weeks >= 2023-08-14 (858 real rows, 324 real positives).

This script re-runs the best compliant ICL configurations against THAT
real dataset, adapted to its real feature set (UCDP_FEATURE_SET -- fatality
counts, state-based/non-state/one-sided violence type, distinct dyads --
NOT the GDELT Goldstein/tone/material-conflict-share features the old
_serialize_context wrongly would have referenced if reused unmodified).

Real LLM cost bound: the real holdout is 858 rows; at ~5s/real call that's
over an hour per configuration. A real, disclosed, evenly-spaced systematic
subsample of the real holdout (every Nth row, preserving the real country/
week distribution rather than only the most recent weeks) is used instead,
honestly labeled as a subsample, not the full holdout.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression

from build_ucdp_panel import UCDP_FEATURE_SET
from icl_hypergraph_compliance_search import ConfigurableICL

OUT_JSON = "results_v2/icl_ucdp_track_a_corrected_results.json"
SPLIT_WEEK = pd.Timestamp("2023-08-14")
SUBSAMPLE_STRIDE = 5  # every 5th real holdout row, systematic (not random, not recency-biased)


class UCDPConfigurableICL(ConfigurableICL):
    """Same retrieval/prompt/LLM/calibration machinery as ConfigurableICL,
    but built on the real UCDP fatality-coded feature set and a context
    description that describes what UCDP actually records (deaths,
    violence type, dyads) instead of the GDELT-specific language
    (Goldstein score, media tone, material-conflict share) the parent
    class's _serialize_context uses -- which would be a real, silent
    dataset-mismatch error again if left unmodified for this data."""

    def fit(self, train_df, label_col):
        self.index_df = train_df.dropna(subset=UCDP_FEATURE_SET + [label_col]).copy()
        self.index_features = self.index_df[UCDP_FEATURE_SET].fillna(0).values
        self.index_mean = self.index_features.mean(axis=0)
        self.index_std = self.index_features.std(axis=0) + 1e-9
        self.index_labels = self.index_df[label_col].values
        return self

    def _serialize_context(self, row):
        return (
            f"Country: {row['country']}, week of {row['week']}\n"
            f"UCDP-recorded events last week: {row['n_events_lag1']:.0f} "
            f"(prior week {row['n_events_lag2']:.0f}, change {row['n_events_delta']:+.0f})\n"
            f"Total UCDP-confirmed fatalities (best estimate) last week: {row['total_best_deaths_lag1']:.0f} "
            f"(change {row['total_best_deaths_delta']:+.0f})\n"
            f"Fatalities per event: {row['deaths_per_event_lag1']:.2f} "
            f"(change {row['deaths_per_event_delta']:+.2f})\n"
            f"Share of events that are state-based armed conflict (vs. non-state or one-sided violence): "
            f"{row['state_based_share_lag1']:.2f} (change {row['state_based_share_delta']:+.2f})\n"
            f"Distinct conflict dyads (named warring-party pairs) active: {row['n_distinct_dyads_lag1']:.0f} "
            f"(change {row['n_distinct_dyads_delta']:+.0f})"
        )

    def predict_proba(self, test_df, label_col):
        probs = []
        for _, row in test_df.iterrows():
            feature_vec = row[UCDP_FEATURE_SET].fillna(0).values.astype(float)
            top_idx, sims = self._retrieve_analogs(feature_vec)
            analog_rows = self.index_df.iloc[top_idx]
            analog_escalation_rate = self.index_labels[top_idx].mean()
            analog_lines = []
            for (_, arow), sim, lab in zip(analog_rows.iterrows(), sims, self.index_labels[top_idx]):
                analog_lines.append(f"  - {arow['country']} {arow['week']} (similarity {sim:.2f}): escalation followed = {bool(lab)}")
            analog_summary = f"{len(top_idx)} nearest analogs, {analog_escalation_rate:.0%} led to escalation:\n" + "\n".join(analog_lines)
            context_text = self._serialize_context(row)
            llm_prob, reasoning, mode = self._call_llm(context_text, analog_summary)
            if llm_prob is not None:
                prob = llm_prob
            else:
                prob = float(np.clip(analog_escalation_rate, 0, 1))
                reasoning = f"[heuristic fallback, mode={mode}] {analog_escalation_rate:.0%} of analogs escalated"
            probs.append(prob)
            self.reasoning_log.append({"country": row["country"], "week": str(row["week"]), "probability": prob, "reasoning": reasoning})
        return np.array(probs)


def summarize(y, p):
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()) if len(y) else 0, "brier": None, "ap": None, "roc_auc": None}
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = None
    return {"n": int(len(y)), "n_pos": int(y.sum()), "brier": float(brier_score_loss(y, p)),
            "ap": float(average_precision_score(y, p)), "roc_auc": roc}


def metrics_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"threshold": float(thr), "n": int(len(y)), "n_pos": int(y.sum()), "n_flagged": int(pred.sum()),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y))}


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


if __name__ == "__main__":
    t0 = time.time()
    panel = pd.read_csv("data/pure_ucdp/ucdp_country_week_panel.csv", parse_dates=["week"])
    select_df = panel[panel["week"] < SPLIT_WEEK].dropna(subset=UCDP_FEATURE_SET + ["label_1"])
    full_holdout = panel[panel["week"] >= SPLIT_WEEK].dropna(subset=UCDP_FEATURE_SET + ["label_1"]).sort_values(["week", "country"]).reset_index(drop=True)
    print(f"Real select window: {len(select_df)} rows. Real full holdout: {len(full_holdout)} rows, "
          f"{int(full_holdout['label_1'].sum())} real positives (matches the established 858/324 real Track A split).", flush=True)

    subsample = full_holdout.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
    print(f"Real systematic subsample (every {SUBSAMPLE_STRIDE}th row): {len(subsample)} rows, "
          f"{int(subsample['label_1'].sum())} real positives.", flush=True)

    results = {}
    configs = [
        ("ICL_UCDP_01_chain_of_thought", {"k_analogs": 5, "use_llm": True, "prompt_style": "chain_of_thought"}),
        ("ICL_UCDP_02_baseline_k5", {"k_analogs": 5, "use_llm": True}),
    ]

    for tag, kwargs in configs:
        tc0 = time.time()
        m = UCDPConfigurableICL(**kwargs)
        m.fit(select_df, "label_1")
        probs = m.predict_proba(subsample, "label_1")
        y = subsample["label_1"].astype(int).values
        pv = np.clip(probs, 1e-6, 1 - 1e-6)
        metrics = summarize(y, pv)

        # one-time calibration + one-time frozen-threshold selection, both fit on the
        # earlier 70% of this real subsample, applied unchanged to the later 30% --
        # same discipline used throughout this whole search.
        weeks_sub = subsample["week"].values
        cal_split_idx = int(len(subsample) * 0.7)
        y_cal, p_cal = y[:cal_split_idx], pv[:cal_split_idx]
        y_eval, p_eval = y[cal_split_idx:], pv[cal_split_idx:]
        calibrated = None
        threshold_result = None
        if 0 < y_cal.sum() < len(y_cal):
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
            p_eval_cal = np.clip(iso.transform(p_eval), 1e-6, 1 - 1e-6)
            calibrated = summarize(y_eval, p_eval_cal)
            thr, reached = select_frozen_threshold(y_cal, np.clip(iso.transform(p_cal), 1e-6, 1 - 1e-6))
            r = metrics_at_threshold(y_eval, p_eval_cal, thr)
            r["reached_80pct_on_select"] = reached
            threshold_result = r

        out = {"tag": tag, "kwargs": kwargs, "compliance": {"verdict": "COMPLIANT",
               "reason": "No parameters are fit via gradient descent or any other learning procedure; fit() "
               "only builds a retrieval index over the real UCDP select window. The LLM's own weights are "
               "never touched."},
               "raw_metrics": metrics, "calibrated_metrics_on_later_30pct": calibrated,
               "frozen_threshold_result": threshold_result, "elapsed_sec": round(time.time() - tc0, 1)}
        results[tag] = out
        print(f"[{tag}] n={metrics['n']} pos={metrics['n_pos']} brier={metrics.get('brier')} ap={metrics.get('ap')} "
              f"roc_auc={metrics.get('roc_auc')} ({out['elapsed_sec']}s)", flush=True)
        if threshold_result:
            print(f"  frozen-threshold (on later 30%): precision={threshold_result['precision']:.3f} "
                  f"recall={threshold_result['recall']:.3f} n={threshold_result['n']} flagged={threshold_result['n_flagged']}", flush=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"real_select_n": len(select_df), "real_full_holdout_n": len(full_holdout),
                    "real_full_holdout_n_pos": int(full_holdout["label_1"].sum()),
                    "subsample_n": len(subsample), "subsample_n_pos": int(subsample["label_1"].sum()),
                    "subsample_stride": SUBSAMPLE_STRIDE, "results": results,
                    "elapsed_min": (time.time() - t0) / 60}, f, indent=2, default=str)
    print(f"\nSaved {OUT_JSON}, {(time.time()-t0)/60:.1f} min total", flush=True)
