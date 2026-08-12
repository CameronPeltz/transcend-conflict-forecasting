"""
Answers the direct question: does feeding the country-week escalation
classifier's real probability output into the grid-cell/discrete-event
model actually improve precision on the real DARPA-specified task,
versus the grid-cell approach alone?

Joins build_countryweek_signal.py's real, walk-forward, never-look-
ahead country-week escalation probabilities onto the grid-cell dataset
by (country, week == issue_date), then re-runs the same rolling-origin
evaluation used throughout this folder on four configs: the round-2
baseline (cell+spatial), +countryweek alone, +ACLED alone (already
known from round 2, included here for a single side-by-side table),
and everything combined.
"""
import json
import numpy as np
import pandas as pd

from train_discrete_event_model_v2 import rolling_predictions, compute_metrics

DATA_PATH = "../data/discrete_event_candidates_v2.csv"
SIGNAL_PATH = "../data/countryweek_escalation_signal.csv"

BASE = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
        "days_since_last_event", "neighbor_count_30d"]
ACLED = ["acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth"]
COUNTRYWEEK = ["countryweek_escalation_prob"]

FEATURE_SETS = {
    "cell_plus_spatial (round-2 baseline)": BASE,
    "cell_plus_spatial_plus_countryweek": BASE + COUNTRYWEEK,
    "cell_plus_spatial_plus_acled": BASE + ACLED,
    "cell_plus_spatial_plus_countryweek_plus_acled": BASE + COUNTRYWEEK + ACLED,
}
HORIZONS = ["10day", "14day"]

if __name__ == "__main__":
    print("Loading grid-cell candidates and country-week signal...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])
    signal = pd.read_csv(SIGNAL_PATH, parse_dates=["week"])

    before = len(df)
    df = df.merge(signal, left_on=["country", "issue_date"], right_on=["country", "week"], how="left")
    matched = df["countryweek_escalation_prob"].notna().sum()
    print(f"{matched} / {before} grid-cell rows matched to a real country-week escalation probability "
          f"({matched/before*100:.1f}%) -- the rest fall outside the country-week signal's own fold "
          f"coverage (its first {52} weeks are pure warm-up, same as the grid-cell model's own).", flush=True)
    df["countryweek_escalation_prob"] = df["countryweek_escalation_prob"].fillna(df["countryweek_escalation_prob"].median())

    results = []
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        print(f"\n=== {horizon} ===", flush=True)
        for fs_name, cols in FEATURE_SETS.items():
            idx, y_true, y_score = rolling_predictions(df, cols, label_col)
            m = compute_metrics(y_true, y_score)
            m["horizon"], m["feature_set"] = horizon, fs_name
            results.append(m)
            print(f"  {fs_name:52s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"AP={m['average_precision']:.3f} AUC={m['roc_auc']:.3f} FP_rate={m['false_positive_rate']:.3f}",
                  flush=True)

    with open("../results/combined_signal_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved ../results/combined_signal_results.json")
