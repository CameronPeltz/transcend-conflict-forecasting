"""
Re-runs the same rolling-origin folds as backtest_harness.py, but this
time logs every model's probability for every single country-week
test point (not just pooled metrics), plus the real underlying
feature values -- so real, specific cases can be picked out and
explained rather than described in the abstract.
"""
import json
import pandas as pd

from backtest_harness import rolling_origin_folds
from models import MODEL_REGISTRY, FEATURES


def build_ledger(panel, label_col):
    folds = rolling_origin_folds(panel, label_col)
    ledger_rows = []

    for cutoff, train, test in folds:
        test_valid = test.dropna(subset=[label_col])
        if len(test_valid) == 0:
            continue

        fitted = {}
        icl_model = None
        for name, cls in MODEL_REGISTRY.items():
            m = cls()
            m.fit(train, label_col)
            fitted[name] = m
            if name == "icl_temporal_graph":
                icl_model = m

        preds_by_model = {name: m.predict_proba(test_valid, label_col) for name, m in fitted.items()}

        for i, (_, row) in enumerate(test_valid.iterrows()):
            entry = {
                "week": str(row["week"]), "country": row["country"], "region": row.get("region", ""),
                "label": int(row[label_col]),
                "n_events": row["n_events"], "material_conflict_share": round(row["material_conflict_share"], 3),
                "mean_goldstein": round(row["mean_goldstein"], 2), "mean_tone": round(row["mean_tone"], 2),
                "distinct_actors": row["distinct_actors"],
            }
            for f in ["n_events_delta", "material_conflict_share_delta", "mean_goldstein_delta",
                      "distinct_actors_delta", "mean_tone_delta"]:
                entry[f] = round(row[f], 3) if pd.notna(row[f]) else None
            for name in MODEL_REGISTRY:
                entry[f"prob_{name}"] = round(float(preds_by_model[name][i]), 3)
            if icl_model is not None and len(icl_model.reasoning_log) >= len(test_valid):
                entry["icl_reasoning"] = icl_model.reasoning_log[-len(test_valid) + i]["reasoning"]
            ledger_rows.append(entry)

    return pd.DataFrame(ledger_rows)


def main():
    panel = pd.read_csv("data/country_week_panel_v2.csv", parse_dates=["week"])

    ledger_1w = build_ledger(panel, "label_1w_ahead")
    ledger_1w.to_csv("data/prediction_ledger_1w.csv", index=False)
    print(f"wrote data/prediction_ledger_1w.csv ({len(ledger_1w)} rows)")

    ledger_2w = build_ledger(panel, "label_2w_ahead")
    ledger_2w.to_csv("data/prediction_ledger_2w.csv", index=False)
    print(f"wrote data/prediction_ledger_2w.csv ({len(ledger_2w)} rows)")

    # surface a few genuinely interesting cases automatically
    for name, ledger in [("1-week", ledger_1w), ("2-week", ledger_2w)]:
        print(f"\n=== {name}: real positives, sorted by model disagreement ===")
        pos = ledger[ledger.label == 1].copy()
        model_cols = [c for c in ledger.columns if c.startswith("prob_")]
        pos["spread"] = pos[model_cols].max(axis=1) - pos[model_cols].min(axis=1)
        print(pos.sort_values("spread", ascending=False)[["country", "week"] + model_cols + ["spread"]].head(8).to_string())

        print(f"\n=== {name}: false negatives everyone missed (label=1, all probs low) ===")
        missed = pos[(pos[model_cols] < 0.3).all(axis=1)]
        print(missed[["country", "week"] + model_cols].head(5).to_string())


if __name__ == "__main__":
    main()
