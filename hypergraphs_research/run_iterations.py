"""
Hypergraph research round: new construction methods layered on top of
the existing production HGNN (scripts/hypergraph_model.py, already 24
iterations logged in results_v2/grand_search_v2_log.jsonl across all 3
tracks -- track A best AP 0.378, track B best AP 0.288, track C best AP
0.688). This script does NOT repeat those; it runs genuinely new
hypergraph constructions (see hg_research_lib.py and
theme_hypergraph.py docstrings for exactly what's new in each) across
the same 3 never-merged tracks (A = small original GDELT, B = large
self-scraped GDELT, C = pure UCDP), reusing grand_search_v2's harness
(panels, folds, metrics, gbm estimator) so numbers are directly
comparable to everything already logged.

Sections, each a real, disclosed "iteration" (or small batch of them):
  1. Hyperedge-type ablation      -- which of the 4-5 hyperedge types
                                      actually carries signal, per track
  2. Propagation-operator variant -- Feng et al. 2019 (symmetric) vs
                                      Bai et al. 2021 (asymmetric,
                                      PyTorch Geometric's formula)
  3. XGI structural features + GBM -- hypergraph-as-feature-engineering
                                      instead of end-to-end HGNN training
  4. Text hypergraph embeddings    -- theme-country-week multi-way
                                      hypergraph vs the existing pairwise
                                      PPMI+SVD theme graph, track B only
                                      (the track with real GKG text data)

Logs every run's full metric panel (accuracy, precision, recall,
specificity, F1, Brier, AP, ROC-AUC, log-loss, MCC) to
hypergraphs_research/hg_iterations_log.jsonl, one line per run,
flushed immediately -- same discipline as grand_search_v2.
"""
import sys, os, json, time
sys.path.insert(0, "scripts")
sys.path.insert(0, "hypergraphs_research")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import grand_search_v2 as gs2
import build_ucdp_panel as up
import hg_research_lib as hl
import theme_hypergraph as th

LOG_PATH = "hypergraphs_research/hg_iterations_log.jsonl"
STATUS_PATH = "hypergraphs_research/STATUS.txt"

TRACK_LABEL_DEFAULT = {t: gs2.TRACKS[t]["label_default"] for t in gs2.TRACKS}


def write_status(msg):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


_log_f = None


def log_result(r):
    global _log_f
    if _log_f is None:
        _log_f = open(LOG_PATH, "a", encoding="utf-8")
    _log_f.write(json.dumps(r, default=str) + "\n")
    _log_f.flush()
    print(f"  -> AP={r.get('ap')}, Brier={r.get('brier')}, Recall={r.get('recall')}, n={r.get('n')}, n_pos={r.get('n_pos')}")


# ---------------------------------------------------------------- conflict_lookup (track C only, new)


def get_conflict_lookup_c():
    raw = pd.read_csv("data/pure_ucdp/GEDEvent_v25_1.csv",
                       usecols=["country", "date_start", "conflict_name"], parse_dates=["date_start"])
    raw = raw[raw["country"].isin(up.UCDP_NAME_TO_CODE.keys())].copy()
    raw["country_code"] = raw["country"].map(up.UCDP_NAME_TO_CODE)
    raw["week"] = raw["date_start"].dt.to_period("W-SUN").dt.start_time
    return hl.build_conflict_lookup(raw, "conflict_name", "country_code", "week")


# ---------------------------------------------------------------- section 1+2: HGNN-variant runner


def run_hgnn_variant(track, label_col, blocks, edge_types, prop_mode, hidden_dim=16,
                      actor_lookup=None, conflict_lookup=None, max_train_rows=3000):
    panel = gs2.TRACKS[track]["get_panel"]()
    fold_fn = gs2._strided(gs2.TRACKS[track]["rof"], gs2.GRAPH_MODEL_FOLD_STRIDE)
    feature_cols = [c for c in gs2.resolve_feature_cols(track, blocks) if c in panel.columns]
    min_train = gs2.TRACKS[track]["min_train_default"]

    def pred_fn(train, test):
        m = hl.HypergraphNNVariant(feature_cols, hidden_dim=hidden_dim, epochs=150,
                                    edge_types=edge_types, prop_mode=prop_mode,
                                    max_train_rows=max_train_rows)
        m.set_actor_lookup(actor_lookup)
        m.set_conflict_lookup(conflict_lookup)
        m.fit(train, label_col)
        return m.predict_proba(test, label_col)

    r = gs2.run_backtest_expanded(panel, label_col, pred_fn, fold_fn, min_train)
    r["n_features"] = len(feature_cols)
    return r


def section_edge_ablation():
    print("\n=== SECTION 1: hyperedge-type ablation ===")
    ablation_steps = [
        ("country_only", ("country",)),
        ("plus_region_week", ("country", "region_week")),
        ("plus_global_week", ("country", "region_week", "global_week")),
        ("plus_actor", ("country", "region_week", "global_week", "actor")),
    ]
    for track in gs2.TRACKS:
        label_col = TRACK_LABEL_DEFAULT[track]
        blocks = gs2.TRACK_FAST_BLOCKS[track][:2]
        actor_lookup = gs2.get_actor_lookup(track[0])
        conflict_lookup = get_conflict_lookup_c() if track == "C_pure_ucdp" else None
        steps = list(ablation_steps)
        if track == "C_pure_ucdp":
            steps.append(("plus_conflict", ("country", "region_week", "global_week", "actor", "conflict")))
        for step_name, edge_types in steps:
            name = f"[{track}] edge_ablation: {step_name}"
            print(name)
            t0 = time.time()
            r = run_hgnn_variant(track, label_col, blocks, edge_types, "symmetric",
                                  actor_lookup=actor_lookup, conflict_lookup=conflict_lookup)
            r.update(name=name, category="edge_ablation", track=track, label_col=label_col,
                      edge_types=list(edge_types), prop_mode="symmetric", elapsed_sec=round(time.time() - t0, 1))
            log_result(r)
            write_status(f"section 1: {name} done")


def section_propagation_mode():
    print("\n=== SECTION 2: propagation-operator variant (Feng et al. vs Bai et al.) ===")
    for track in ["B_large_scraped_gdelt", "C_pure_ucdp"]:
        label_col = TRACK_LABEL_DEFAULT[track]
        blocks = gs2.TRACK_FAST_BLOCKS[track][:2]
        actor_lookup = gs2.get_actor_lookup(track[0])
        conflict_lookup = get_conflict_lookup_c() if track == "C_pure_ucdp" else None
        edge_types = ALL5 if track == "C_pure_ucdp" else ("country", "region_week", "global_week", "actor")
        for mode in ["symmetric", "asymmetric"]:
            name = f"[{track}] propagation_mode: {mode}"
            print(name)
            t0 = time.time()
            r = run_hgnn_variant(track, label_col, blocks, edge_types, mode,
                                  actor_lookup=actor_lookup, conflict_lookup=conflict_lookup)
            r.update(name=name, category="propagation_mode", track=track, label_col=label_col,
                      edge_types=list(edge_types), prop_mode=mode, elapsed_sec=round(time.time() - t0, 1))
            log_result(r)
            write_status(f"section 2: {name} done")


ALL5 = ("country", "region_week", "global_week", "actor", "conflict")


# ---------------------------------------------------------------- section 3: xgi structural features + gbm


def section_xgi_structural():
    print("\n=== SECTION 3: xgi structural features -> GBM (feature-engineering alternative) ===")
    for track in gs2.TRACKS:
        panel = gs2.TRACKS[track]["get_panel"]()
        fold_fn = gs2.TRACKS[track]["rof"]  # cheap model -- use the track's own (already strided-for-length) fold_fn, no extra graph stride needed
        label_col = TRACK_LABEL_DEFAULT[track]
        min_train = gs2.TRACKS[track]["min_train_default"]
        tab_blocks = gs2.TRACK_FAST_BLOCKS[track][:2]
        tab_cols = [c for c in gs2.resolve_feature_cols(track, tab_blocks) if c in panel.columns]
        actor_lookup = gs2.get_actor_lookup(track[0])
        conflict_lookup = get_conflict_lookup_c() if track == "C_pure_ucdp" else None
        edge_types = ALL5 if track == "C_pure_ucdp" else ("country", "region_week", "global_week", "actor")
        cat_cols = ["country"]

        for variant in ["structural_plus_tabular", "structural_only"]:
            feature_cols = (tab_cols + hl.HG_STRUCTURAL_FEATURES) if variant == "structural_plus_tabular" else list(hl.HG_STRUCTURAL_FEATURES)

            def pred_fn(train, test, feature_cols=feature_cols, cat_cols=cat_cols):
                nodes_cols = ["country", "week"] + (["region"] if "region" in train.columns else [])
                combined = pd.concat([train, test], ignore_index=True)
                nodes = combined[nodes_cols].reset_index(drop=True)
                struct = hl.xgi_structural_features(nodes, actor_lookup, conflict_lookup,
                                                     edge_types=edge_types)
                combined2 = combined.reset_index(drop=True).merge(
                    struct, on=["country", "week"], how="left", suffixes=("", "_struct"))
                tr2 = combined2.iloc[: len(train)].copy()
                te2 = combined2.iloc[len(train):].copy()
                return gs2.fit_predict_tabular("gbm", {}, tr2, te2, feature_cols, label_col, cat_cols, None)

            name = f"[{track}] xgi_structural: {variant}"
            print(name)
            t0 = time.time()
            r = gs2.run_backtest_expanded(panel, label_col, pred_fn, fold_fn, min_train)
            r["n_features"] = len(feature_cols)
            r.update(name=name, category="xgi_structural", track=track, label_col=label_col,
                      variant=variant, elapsed_sec=round(time.time() - t0, 1))
            log_result(r)
            write_status(f"section 3: {name} done")


# ---------------------------------------------------------------- section 4: text hypergraph, track B only


def section_text_hypergraph():
    print("\n=== SECTION 4: text hypergraph (theme x country-week) vs pairwise PPMI+SVD, track B ===")
    track = "B_large_scraped_gdelt"
    panel = gs2.TRACKS[track]["get_panel"]()
    fold_fn = gs2.TRACKS[track]["rof"]
    label_col = TRACK_LABEL_DEFAULT[track]
    min_train = gs2.TRACKS[track]["min_train_default"]
    base_blocks = gs2.TRACK_FAST_BLOCKS[track][:2]
    base_cols = [c for c in gs2.resolve_feature_cols(track, base_blocks) if c in panel.columns]

    print("building hypergraph theme embeddings...")
    _, hg_theme_df, meta = th.build_theme_hypergraph_embeddings("data/scraped_large/gkg_theme_freq_countryweek.csv")
    print("theme hypergraph meta:", meta)
    panel_with_hg = panel.merge(hg_theme_df, on=["country", "week"], how="left")

    existing_embed_cols = [c for c in gs2.GKG_EMBED_FEATURE_SET if c in panel.columns]

    variants = [
        ("anchor_no_text", panel, base_cols, "no theme/text features at all"),
        ("existing_pairwise_ppmi_svd", panel, base_cols + existing_embed_cols,
         "production pairwise theme-cooccurrence PPMI+SVD embedding (graph_nlp_features.py)"),
        ("new_text_hypergraph", panel_with_hg, base_cols + th.HG_THEME_EMBED_FEATURES,
         "new: theme x country-week multi-way hypergraph, TF-IDF-weighted SVD of the incidence matrix"),
    ]
    for variant_name, use_panel, feature_cols, desc in variants:
        feature_cols = [c for c in feature_cols if c in use_panel.columns]
        name = f"[{track}] text_hypergraph: {variant_name}"
        print(name, "|", desc)
        t0 = time.time()

        def pred_fn(train, test, feature_cols=feature_cols):
            return gs2.fit_predict_tabular("gbm", {}, train, test, feature_cols, label_col, ["country"], None)

        r = gs2.run_backtest_expanded(use_panel, label_col, pred_fn, fold_fn, min_train)
        r["n_features"] = len(feature_cols)
        r.update(name=name, category="text_hypergraph", track=track, label_col=label_col,
                  variant=variant_name, description=desc, elapsed_sec=round(time.time() - t0, 1))
        log_result(r)
        write_status(f"section 4: {name} done")


def main():
    os.makedirs("hypergraphs_research", exist_ok=True)
    t_start = time.time()
    write_status("starting hypergraph research iterations")
    section_edge_ablation()
    section_propagation_mode()
    section_xgi_structural()
    section_text_hypergraph()
    write_status(f"DONE in {(time.time()-t_start)/60:.1f} min")
    print(f"\nAll sections complete in {(time.time()-t_start)/60:.1f} min. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
