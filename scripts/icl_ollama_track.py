"""
The dedicated real-local-LLM ICL forecaster track (models.py's
ICLForecaster with use_llm=True, hitting the free, open-source, local
Ollama llama3.1:8b model already installed -- no API key, no per-call
cost). Kept separate from grand_search_v2.py's 1000+-iteration sweep
because each prediction costs several real seconds of local CPU
inference (~10s warm; see the timing check this project ran before
building this) -- fine for a smaller, deliberate comparison, wrong for
a 1000+-iteration sweep with tens of folds each.

What this script actually does:
  1. Runs the real ICLForecaster (use_llm=True) against Track A (the
     original small, most information-dense panel -- fewer folds means
     this stays tractable) at the 1-week and 2-week horizons, and
     records every real reasoning trace the model produces.
  2. Runs the same forecaster with use_llm=False (heuristic fallback)
     on the identical folds, so the comparison is apples-to-apples: the
     only thing that changes is whether a real LLM call replaces the
     fixed formula.
  3. Also runs it against Track B (large GDELT) restricted to the most
     recent ~20 weeks (again for tractability) and Track C (UCDP) same
     restriction, so the real-LLM comparison has at least one data
     point per track, not just Track A.
  4. Writes every real reasoning trace to results_v2/icl_ollama_log.json
     for the final write-up's case studies -- this is where "what did
     the best model actually say and why" comes from for the real-LLM
     configuration, not a fabricated example.
"""
import sys
sys.path.insert(0, "scripts")
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

import iteration_engine as ie
import large_panel as lp
import build_ucdp_panel as up
from external_features import attach_external_features
from graph_features import build_graph_features
from models import ICLForecaster, FEATURES as ICL_FEATURES


def run_track_a(use_llm, max_folds=14):
    p = ie.build_panel(granularity="W", n_lags=2, label_z=1.0, horizons=(1, 2))
    ext = attach_external_features(p)
    raw = ie.load_raw()
    g = build_graph_features(raw, p[["country", "week"]])
    panel = ext.merge(g, on=["country", "week"], how="left")

    results = {}
    for label_col in ["label_quad_1", "label_quad_2"]:
        folds = ie.rolling_origin_folds(panel, label_col, min_train=6)
        folds = folds[-max_folds:]
        all_p, all_y, traces = [], [], []
        for cutoff, train, test in folds:
            test_v = test.dropna(subset=[label_col])
            if len(test_v) == 0:
                continue
            m = ICLForecaster(k_analogs=5, use_llm=use_llm)
            m.fit(train, label_col)
            probs = m.predict_proba(test_v, label_col)
            all_p.extend(probs.tolist())
            all_y.extend(test_v[label_col].astype(int).tolist())
            traces.extend(m.reasoning_log)
        y, pv = np.array(all_y), np.clip(np.array(all_p), 1e-6, 1 - 1e-6)
        metrics = summarize(y, pv)
        results[label_col] = {"metrics": metrics, "traces": traces}
        print(f"  track A {label_col} use_llm={use_llm}: n={len(y)} pos={int(y.sum())} "
              f"brier={metrics.get('brier')} ap={metrics.get('ap')}")
    return results


def run_track_generic(track_name, panel, feature_cols, rof_fn, label_col, use_llm, max_folds=10):
    all_p, all_y, traces = [], [], []
    folds = rof_fn(panel, label_col)
    folds = folds[-max_folds:]
    for cutoff, train, test in folds:
        test_v = test.dropna(subset=[label_col])
        if len(test_v) == 0:
            continue
        m = ICLForecaster(k_analogs=5, use_llm=use_llm)
        # ICLForecaster's FEATURES constant is fixed (models.py); reindex the
        # incoming frames to present those exact columns regardless of track,
        # filling any track-specific gaps with 0 so the same class works
        # unmodified across tracks (disclosed simplification: it still reasons
        # only over the generic lag/delta shape, not track-specific columns).
        train2 = _reindex_for_icl(train, feature_cols)
        test2 = _reindex_for_icl(test_v, feature_cols)
        m.fit(train2, label_col)
        probs = m.predict_proba(test2, label_col)
        all_p.extend(probs.tolist())
        all_y.extend(test_v[label_col].astype(int).tolist())
        traces.extend(m.reasoning_log)
    y, pv = np.array(all_y), np.clip(np.array(all_p), 1e-6, 1 - 1e-6)
    metrics = summarize(y, pv)
    print(f"  track {track_name} use_llm={use_llm}: n={len(y)} pos={int(y.sum())} "
          f"brier={metrics.get('brier')} ap={metrics.get('ap')}")
    return {"metrics": metrics, "traces": traces}


def _reindex_for_icl(df, feature_cols):
    out = df.copy()
    for c in ICL_FEATURES:
        if c not in out.columns:
            src = feature_cols[ICL_FEATURES.index(c) % len(feature_cols)] if feature_cols else None
            out[c] = out[src] if src and src in out.columns else 0.0
    return out


def summarize(y, p):
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()) if len(y) else 0, "brier": None, "ap": None}
    pred05 = (p >= 0.5).astype(int)
    tp = int(((pred05 == 1) & (y == 1)).sum()); fp = int(((pred05 == 1) & (y == 0)).sum())
    fn = int(((pred05 == 0) & (y == 1)).sum()); tn = int(((pred05 == 0) & (y == 0)).sum())
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = None
    return {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "brier": float(brier_score_loss(y, p)), "ap": float(average_precision_score(y, p)),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y)),
        "roc_auc": roc,
    }


STATUS_PATH = "results_v2/STATUS_icl_ollama.txt"


def write_status(phase, t0):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        f.write(f"icl_ollama_track - live status\nupdated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"phase: {phase}\nelapsed: {(time.time()-t0)/60:.1f} min\n")


RESULTS_PATH = "results_v2/icl_ollama_log.json"


def load_existing():
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_partial(out):
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)


def main():
    t0 = time.time()
    out = load_existing()  # resume support: real LLM calls are ~10s each, too
    # expensive to redo from scratch after an interruption -- each phase below
    # is saved to disk as soon as it finishes, and already-finished phases
    # (present as keys in the loaded file) are skipped rather than re-run.
    write_status("starting" if not out else f"resuming ({len(out)} phases already done)", t0)

    if "A_fallback" not in out:
        print("Track A, heuristic fallback (fast, baseline for comparison)...")
        write_status("Track A fallback (fast)", t0)
        out["A_fallback"] = run_track_a(use_llm=False, max_folds=14)
        save_partial(out)

    if "A_real_llm" not in out:
        print("Track A, REAL local Ollama LLM calls (slow, ~10s/call)...")
        write_status("Track A real LLM (slow, ~10s/call, real reasoning traces)", t0)
        out["A_real_llm"] = run_track_a(use_llm=True, max_folds=14)
        save_partial(out)

    if "B_real_llm" not in out:
        print("Track B (large GDELT), real LLM, most recent weeks only...")
        write_status("Track B real LLM", t0)
        raw_b = lp.load_raw()
        panel_b = lp.build_panel(raw_df=raw_b)
        out["B_real_llm"] = run_track_generic("B", panel_b, lp.FEATURE_SETS["core"], lp.rolling_origin_folds,
                                               "label_quad_1", use_llm=True, max_folds=8)
        save_partial(out)

    print("Track C (UCDP pure), real LLM, most recent weeks only...")
    if "C_real_llm" not in out:
        write_status("Track C real LLM", t0)
        panel_c, _ = up.build_panel()
        def c_rof(panel, label_col):
            weeks = sorted(panel["week"].unique())
            weeks = [w for w in weeks if w >= pd.Timestamp("2023-01-01")]
            folds = []
            for i in range(8, len(weeks)):
                cutoff = weeks[i]
                train = panel[panel["week"] < cutoff].dropna(subset=[label_col])
                test = panel[panel["week"] == cutoff]
                if len(train) and len(test) and not test[label_col].isna().all():
                    folds.append((cutoff, train, test))
            return folds
        out["C_real_llm"] = run_track_generic("C", panel_c, up.UCDP_FEATURE_SET, c_rof,
                                               "label_1", use_llm=True, max_folds=8)
        save_partial(out)

    write_status("DONE", t0)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
