"""Resume helper #3. Diagnosed why resume #2 made zero progress on
Track B within its time budget: gs2.get_actor_lookup('B') calls
large_panel.load_raw(), which re-reads the 5,857,451-row raw GDELT CSV
from scratch in this fresh process (the in-memory cache from earlier
processes is gone) -- confirmed by isolated profiling (a 30-40s timeout
was hit just building the incidence matrix + degree sums, i.e. before
any xgi call even ran). That cost is compounded right now by real,
heavy CPU contention from this machine's other concurrent jobs
(grand_search_v2.py's own iteration rate has dropped to ~0.44/s, well
below its earlier ~0.6+/s -- a real, visible symptom of contention, not
a guess).

Disclosed simplification for the Track B structural-feature experiment
only: drop the 'actor' edge type (the one requiring that expensive raw
reload) and use country/region_week/global_week instead. Track C's
lookups are cheap (UCDP's raw CSV is ~463K rows, already fast in every
prior test this session) so it keeps all 5 edge types unchanged.
A stride is also applied to both tracks' fold functions here, on top of
whatever striding the track's own rof already applies, purely for
tractability under today's contention -- a real, disclosed coarsening
of the evaluation, not a change to how any individual fold is trained
or scored.
"""
import sys, time
sys.path.insert(0, "scripts")
sys.path.insert(0, "hypergraphs_research")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import grand_search_v2 as gs2
import hg_research_lib as hl
# grand_search_v2's own import re-inserts "scripts" at sys.path[0] (its own
# top-level sys.path.insert(0, "scripts")), which -- since scripts/ ALSO has
# an unrelated pre-existing run_iterations.py from the original v1 grand
# search -- would otherwise shadow hypergraphs_research/run_iterations.py.
# Re-assert priority right before importing it (confirmed root cause: an
# earlier run of this script silently imported and executed the WRONG
# run_iterations, wasting ~30s regenerating scripts/run_iterations.py's own
# deterministic data/iteration_results.json output -- harmless, but real).
sys.path.insert(0, "hypergraphs_research")
import run_iterations as ri
assert hasattr(ri, "ALL5"), "wrong run_iterations module imported (shadowed by scripts/run_iterations.py)"

EXTRA_STRIDE = 2


def strided(fold_fn, stride):
    def wrapped(panel, label_col, min_train=8):
        folds = fold_fn(panel, label_col, min_train)
        return folds[::stride] if stride > 1 else folds
    return wrapped


TRACK_STRUCT_EDGE_TYPES = {
    "B_large_scraped_gdelt": ("country", "region_week", "global_week"),  # actor dropped here -- see docstring
    "C_pure_ucdp": ri.ALL5,
}

for track in ["B_large_scraped_gdelt", "C_pure_ucdp"]:
    t_track0 = time.time()
    panel = gs2.TRACKS[track]["get_panel"]()
    print(f"[{track}] panel ready in {time.time()-t_track0:.1f}s, shape={panel.shape}", flush=True)
    fold_fn = strided(gs2.TRACKS[track]["rof"], EXTRA_STRIDE)
    label_col = ri.TRACK_LABEL_DEFAULT[track]
    min_train = gs2.TRACKS[track]["min_train_default"]
    tab_blocks = gs2.TRACK_FAST_BLOCKS[track][:2]
    tab_cols = [c for c in gs2.resolve_feature_cols(track, tab_blocks) if c in panel.columns]
    edge_types = TRACK_STRUCT_EDGE_TYPES[track]

    t_lookup0 = time.time()
    actor_lookup = gs2.get_actor_lookup(track[0]) if "actor" in edge_types else None
    conflict_lookup = ri.get_conflict_lookup_c() if (track == "C_pure_ucdp" and "conflict" in edge_types) else None
    print(f"[{track}] lookups ready in {time.time()-t_lookup0:.1f}s", flush=True)
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

        name = f"[{track}] xgi_structural: {variant} (edge_types={','.join(edge_types)}, fold_stride={EXTRA_STRIDE}x)"
        print(name, flush=True)
        t0 = time.time()
        r = gs2.run_backtest_expanded(panel, label_col, pred_fn, fold_fn, min_train)
        r["n_features"] = len(feature_cols)
        r.update(name=name, category="xgi_structural", track=track, label_col=label_col,
                  variant=variant, edge_types=list(edge_types), fold_stride=EXTRA_STRIDE,
                  elapsed_sec=round(time.time() - t0, 1))
        ri.log_result(r)
        ri.write_status(f"section 3 (resume3): {name} done")

ri.section_text_hypergraph()
print("\nall remaining sections complete.", flush=True)
