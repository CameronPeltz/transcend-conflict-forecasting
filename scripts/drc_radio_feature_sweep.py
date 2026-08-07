"""
An iteration sweep over real radio-feature engineering choices --
which keyword groups, which relevance-score filter, which lag depth --
logged the same way as this project's earlier 1,150-iteration grand
search, for consistency and so results are comparable in format.

Stated once, clearly, so it isn't buried: with only ~4-5 real
overlapping weeks between the radio data and a checkable GDELT label,
NONE of these iterations can produce a statistically significant
result on their own. This sweep exists to see which feature-engineering
choices are at least DIRECTIONALLY consistent (same sign of effect
across variants) versus noisy/inconsistent -- a real, useful
engineering signal even at this sample size, not a substitute for the
much larger sample a real precision claim would need.
"""
import sys
sys.path.insert(0, "scripts")
import json
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import build_drc_panel as bp
import rootwise_features as rw
from drc_radio_integration_test import EVENT_FEATURES, rolling_backtest, build_panel

def _grp(actor=False, location=False, violence=False, diplomacy=False):
    return dict(actor_terms=rw.ACTOR_TERMS if actor else [],
                location_terms=rw.LOCATION_TERMS if location else [],
                violence_terms=rw.VIOLENCE_TERMS if violence else [],
                diplomacy_terms=rw.DIPLOMACY_TERMS if diplomacy else [])

# every single term-group plus every pairwise and full combination --
# a real, complete ablation of which real keyword categories carry signal
FEATURE_GROUPS = {
    "actor_only": _grp(actor=True),
    "location_only": _grp(location=True),
    "violence_only": _grp(violence=True),
    "diplomacy_only": _grp(diplomacy=True),
    "actor_location": _grp(actor=True, location=True),
    "actor_violence": _grp(actor=True, violence=True),
    "actor_diplomacy": _grp(actor=True, diplomacy=True),
    "location_violence": _grp(location=True, violence=True),
    "location_diplomacy": _grp(location=True, diplomacy=True),
    "violence_diplomacy": _grp(violence=True, diplomacy=True),
    "actor_location_violence": _grp(actor=True, location=True, violence=True),
    "actor_location_diplomacy": _grp(actor=True, location=True, diplomacy=True),
    "actor_violence_diplomacy": _grp(actor=True, violence=True, diplomacy=True),
    "location_violence_diplomacy": _grp(location=True, violence=True, diplomacy=True),
    "all_combined": _grp(actor=True, location=True, violence=True, diplomacy=True),
}
RELEVANCE_FILTERS = [None, 2]  # min_relevance=1 is a no-op: 1 is the real data's own floor value
MIN_TRAIN_VALUES = [20]  # fixed: for the 5 fixed radio-covered weeks (near the end of a 157-week
# series), training data already vastly exceeds any of {12,20,30} by then -- confirmed by direct
# comparison (identical results across min_train for the same feature config), so sweeping it here
# only re-runs the same computation 3x for no additional information. Kept as a single value.


def main():
    raw_gdelt = bp.load_gdelt_drc_raw()
    event_panel = bp.build_gdelt_drc_panel(raw_gdelt)
    rootwise_raw = rw.load_rootwise_raw()

    log_path = "data/drc/radio_sweep_log.jsonl"
    t0 = time.time()
    idx = 0
    # cache baseline (event-only) preds per min_train since they don't
    # depend on the radio feature group/relevance filter being swept
    baseline_cache = {}

    with open(log_path, "w") as f:
        for min_train in MIN_TRAIN_VALUES:
            baseline_cache[min_train] = rolling_backtest(event_panel, EVENT_FEATURES, min_train=min_train)

        for group_name, group_kwargs in FEATURE_GROUPS.items():
            for min_rel in RELEVANCE_FILTERS:
                kwargs = dict(group_kwargs)
                if min_rel is not None:
                    kwargs["min_relevance"] = min_rel
                radio_feat = rw.build_weekly_features(rootwise_raw, **kwargs)
                panel = event_panel.merge(radio_feat.drop(columns=["country"]), on="week", how="left")

                radio_cols = [c for c in radio_feat.columns if c not in ("country", "week")]
                lag_delta_cols = [f"{c}_lag1" for c in radio_cols if f"{c}_lag1" in panel.columns] + \
                                  [f"{c}_delta" for c in radio_cols if f"{c}_delta" in panel.columns]
                all_features = EVENT_FEATURES + lag_delta_cols
                radio_weeks = panel[panel["radio_n_clips"].notna()]["week"].tolist()

                for min_train in MIN_TRAIN_VALUES:
                    idx += 1
                    baseline_preds = baseline_cache[min_train]
                    radio_preds = rolling_backtest(panel, all_features, min_train=min_train)

                    overlap = sorted(set(baseline_preds.week) & set(radio_preds.week) & set(radio_weeks))
                    if not overlap:
                        row = {"iter": idx, "name": f"{group_name} | min_relevance={min_rel} | min_train={min_train}",
                               "feature_group": group_name, "min_relevance": min_rel, "min_train": min_train,
                               "n_overlap_weeks": 0, "note": "no overlapping weeks at this min_train", "elapsed_sec": time.time() - t0}
                        f.write(json.dumps(row, default=str) + "\n"); f.flush()
                        continue
                    b = baseline_preds[baseline_preds.week.isin(overlap)].sort_values("week")
                    r = radio_preds[radio_preds.week.isin(overlap)].sort_values("week")
                    yb, pb = b.actual.values, np.clip(b.prob.values, 1e-6, 1 - 1e-6)
                    yr, pr = r.actual.values, np.clip(r.prob.values, 1e-6, 1 - 1e-6)
                    mse_b = float(((pb - yb) ** 2).mean())
                    mse_r = float(((pr - yr) ** 2).mean())

                    row = {
                        "iter": idx, "name": f"{group_name} | min_relevance={min_rel} | min_train={min_train}",
                        "feature_group": group_name, "min_relevance": min_rel, "min_train": min_train,
                        "n_overlap_weeks": len(overlap),
                        "baseline_mse": mse_b, "radio_augmented_mse": mse_r,
                        "improved": mse_r < mse_b,
                        "delta": mse_b - mse_r,
                        "elapsed_sec": time.time() - t0,
                    }
                    f.write(json.dumps(row, default=str) + "\n")
                    f.flush()
                    print(f"[{idx}] {row['name']}: baseline_mse={mse_b:.4f} radio_mse={mse_r:.4f} "
                          f"{'IMPROVED' if row['improved'] else 'did not improve'} (n={len(overlap)} weeks)")

    print(f"\nDONE, {idx} real feature-engineering variants tested in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
