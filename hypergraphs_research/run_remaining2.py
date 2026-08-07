"""Resume helper #2: Track A's xgi_structural (both variants) already
logged. This runs only Track B + C xgi_structural, then the text
hypergraph section -- appending to the same log. Split into its own
script (rather than re-running all of section_xgi_structural) because
the tool's background-command timeout keeps landing mid-run under
heavy CPU contention from this machine's other concurrent jobs
(grand_search_v2.py, icl_ollama_track.py, both the user's own,
unrelated, higher-priority runs) -- smaller increments make actual
forward progress each time even if a given call gets cut off."""
import sys, time
sys.path.insert(0, "scripts")
sys.path.insert(0, "hypergraphs_research")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import grand_search_v2 as gs2
import hg_research_lib as hl
import run_iterations as ri

for track in ["B_large_scraped_gdelt", "C_pure_ucdp"]:
    panel = gs2.TRACKS[track]["get_panel"]()
    fold_fn = gs2.TRACKS[track]["rof"]
    label_col = ri.TRACK_LABEL_DEFAULT[track]
    min_train = gs2.TRACKS[track]["min_train_default"]
    tab_blocks = gs2.TRACK_FAST_BLOCKS[track][:2]
    tab_cols = [c for c in gs2.resolve_feature_cols(track, tab_blocks) if c in panel.columns]
    actor_lookup = gs2.get_actor_lookup(track[0])
    conflict_lookup = ri.get_conflict_lookup_c() if track == "C_pure_ucdp" else None
    edge_types = ri.ALL5 if track == "C_pure_ucdp" else ("country", "region_week", "global_week", "actor")
    cat_cols = ["country"]

    for variant in ["structural_plus_tabular", "structural_only"]:
        feature_cols = (tab_cols + hl.HG_STRUCTURAL_FEATURES) if variant == "structural_plus_tabular" else list(hl.HG_STRUCTURAL_FEATURES)

        def pred_fn(train, test, feature_cols=feature_cols, cat_cols=cat_cols):
            nodes_cols = ["country", "week"] + (["region"] if "region" in train.columns else [])
            combined = pd.concat([train, test], ignore_index=True)
            nodes = combined[nodes_cols].reset_index(drop=True)
            struct = hl.xgi_structural_features(nodes, actor_lookup, conflict_lookup, edge_types=edge_types)
            combined2 = combined.reset_index(drop=True).merge(struct, on=["country", "week"], how="left", suffixes=("", "_struct"))
            tr2 = combined2.iloc[: len(train)].copy()
            te2 = combined2.iloc[len(train):].copy()
            return gs2.fit_predict_tabular("gbm", {}, tr2, te2, feature_cols, label_col, cat_cols, None)

        name = f"[{track}] xgi_structural: {variant}"
        print(name, flush=True)
        t0 = time.time()
        r = gs2.run_backtest_expanded(panel, label_col, pred_fn, fold_fn, min_train)
        r["n_features"] = len(feature_cols)
        r.update(name=name, category="xgi_structural", track=track, label_col=label_col,
                  variant=variant, elapsed_sec=round(time.time() - t0, 1))
        ri.log_result(r)
        ri.write_status(f"section 3 (resume2): {name} done")

ri.section_text_hypergraph()
print("\nall remaining sections complete.", flush=True)
