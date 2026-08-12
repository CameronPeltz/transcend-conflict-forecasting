"""
Round 3, lever 7: does routing ambiguous/discontinuous cells through the
project's existing ICL/LLM reasoning layer (Ollama, llama3.1 -- the same
free, local, no-per-call-cost model used for Criterion 1 elsewhere in
this project) catch false alarms the numeric ensemble misses?

Diagnosis this targets: round 2's frozen-threshold holdout failure was
traced to the August 2021 Afghanistan/Taliban regime change -- cells
that were "hot" all summer kept scoring high right through the
government's collapse, because the ensemble only sees event counts, not
what kind of event is happening or why. This script pulls real holdout
cases (true hits, false alarms, and near-threshold ambiguous calls) from
the round-3 10-day combined-feature frozen-threshold run, and for each
one prompts the LLM with the same real trend/momentum context available
to the ensemble, asking it to reason about whether recent activity is
still escalating or has already peaked/turned -- then checks whether
its calls, on this real sample, would have been more or less accurate
than the ensemble's raw threshold decision.

This is a scoped, real-output test on a real sample (not the full
228,851-row dataset) for the same reason Criterion 1's ICL validation
elsewhere in this project was scoped: live local-LLM inference costs
several real seconds per call.
"""
import json
import urllib.request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "../data/discrete_event_candidates_v3.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

ROUND2_BEST = ["cell_count_30d", "cell_count_60d", "cell_count_90d", "cell_count_365d",
               "days_since_last_event", "neighbor_count_30d",
               "acled_civ_targeting_events_prevmonth", "acled_civ_targeting_fatalities_prevmonth"]
COMBINED = ROUND2_BEST + ["cell_count_30d_delta", "cell_count_90d_delta", "momentum_ratio_30_90",
                          "neighbor_count_30d_ring2"]

N_PER_BUCKET = 10


def fit_predict_ensemble(train_X, train_y, test_X):
    pos = max(1, train_y.sum())
    neg = max(1, len(train_y) - pos)
    gbm = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                 random_state=0, class_weight="balanced", n_jobs=-1)
    scaler = StandardScaler()
    train_Xs, test_Xs = scaler.fit_transform(train_X), scaler.transform(test_X)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)
    gbm.fit(train_X, train_y); rf.fit(train_X, train_y); logreg.fit(train_Xs, train_y)
    p = (gbm.predict_proba(test_X)[:, 1] + rf.predict_proba(test_X)[:, 1] + logreg.predict_proba(test_Xs)[:, 1]) / 3.0
    return p


def get_holdout_predictions(df, feature_cols, label_col, target_precision=0.80):
    """Reproduces the round-3 frozen-threshold split/train/threshold exactly,
    but returns the per-row holdout index/score/y instead of only summary
    metrics, so real cases can be pulled for the LLM to reason over."""
    issue_dates_sorted = sorted(df["issue_date"].unique())
    split_idx = int(len(issue_dates_sorted) * 0.6)
    split_date = issue_dates_sorted[split_idx]
    select_df = df[df["issue_date"] < split_date]
    holdout_df = df[df["issue_date"] >= split_date]

    # rolling-origin select-window scores, to pick the frozen threshold
    min_train, n_folds = 52, 10
    sel_dates = sorted(select_df["issue_date"].unique())
    remaining = len(sel_dates) - min_train
    step = max(1, remaining // n_folds)
    all_true, all_score = [], []
    for k in range(min(n_folds, remaining)):
        cutoff_idx = min_train + k * step
        if cutoff_idx >= len(sel_dates) - 1:
            break
        test_dates = sel_dates[cutoff_idx: min(cutoff_idx + step, len(sel_dates))]
        train = select_df[select_df["issue_date"] < sel_dates[cutoff_idx]]
        test = select_df[select_df["issue_date"].isin(test_dates)]
        if len(train) < 200 or test[label_col].sum() == 0:
            continue
        s = fit_predict_ensemble(train[feature_cols].fillna(0).values, train[label_col].values,
                                  test[feature_cols].fillna(0).values)
        all_true.append(test[label_col].values); all_score.append(s)
    sel_true, sel_score = np.concatenate(all_true), np.concatenate(all_score)
    order = np.argsort(-sel_score)
    sorted_true, sorted_score = sel_true[order], sel_score[order]
    chosen_threshold = None
    for k in range(10, len(sorted_true)):
        if sorted_true[:k].sum() / k < target_precision:
            chosen_threshold = float(sorted_score[k - 1]); break
    if chosen_threshold is None:
        chosen_threshold = float(sorted_score[-1])

    train_X = select_df[feature_cols].fillna(0).values
    train_y = select_df[label_col].values
    holdout_X = holdout_df[feature_cols].fillna(0).values
    holdout_score = fit_predict_ensemble(train_X, train_y, holdout_X)
    holdout_y = holdout_df[label_col].values
    return holdout_df.reset_index(drop=True), holdout_score, holdout_y, chosen_threshold


def serialize_context(row, threshold):
    trend = "rising" if row["cell_count_30d_delta"] > 2 else ("falling" if row["cell_count_30d_delta"] < -2 else "flat")
    return (
        f"Grid cell in {row['country']}, forecast issued {row['issue_date'].date()}.\n"
        f"Events in the last 30 days: {row['cell_count_30d']:.0f} "
        f"(vs {row['cell_count_30d_prior']:.0f} in the preceding 30 days; trend is {trend}, "
        f"delta {row['cell_count_30d_delta']:+.0f}).\n"
        f"Events in the last 90 days: {row['cell_count_90d']:.0f} "
        f"(vs {row['cell_count_90d_prior']:.0f} in the preceding 90 days).\n"
        f"Events in the last 365 days: {row['cell_count_365d']:.0f}.\n"
        f"Days since the most recent event in this cell: {row['days_since_last_event']:.0f}.\n"
        f"Events in immediately neighboring cells, last 30 days: {row['neighbor_count_30d']:.0f}.\n"
        f"Events in the wider surrounding ring of cells, last 30 days: {row['neighbor_count_30d_ring2']:.0f}.\n"
        f"Numeric ensemble model's probability estimate: {row['ensemble_score']:.3f} "
        f"(alert threshold for this task is {threshold:.3f})."
    )


def call_llm(context_text):
    prompt = (
        "You are a conflict early-warning analyst reviewing one grid cell's recent activity. "
        "A numeric model has already scored this cell using event counts alone. Weigh whether the "
        "recent trend (rising, flat, or falling activity) suggests this is still escalating, or "
        "settling down / a one-off spike rather than sustained conflict likely to produce a new "
        "event soon. Reply on exactly two lines:\n"
        "PROBABILITY: <a single number between 0 and 1>\n"
        "REASON: <one short sentence>\n\n"
        f"{context_text}"
    )
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"num_predict": 60, "temperature": 0.0},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=150) as resp:
        body = json.loads(resp.read().decode())
    raw = body.get("response", "")
    prob = None
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("PROBABILITY"):
            try:
                prob = float(line.split(":", 1)[1].strip().split()[0])
            except Exception:
                pass
    return prob, raw.strip()


def main():
    print("Loading round-3 dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["issue_date"])

    print("Reproducing the round-3 10-day / combined-features / 80%-target frozen-threshold run "
          "to get real per-row holdout predictions...", flush=True)
    holdout_df, holdout_score, holdout_y, threshold = get_holdout_predictions(
        df, COMBINED, "label_10day", target_precision=0.80)
    holdout_df = holdout_df.copy()
    holdout_df["ensemble_score"] = holdout_score
    holdout_df["actual"] = holdout_y
    print(f"Frozen threshold: {threshold:.3f}. Holdout n={len(holdout_df)}.", flush=True)

    predicted_pos = holdout_df["ensemble_score"] >= threshold
    true_hits = holdout_df[predicted_pos & (holdout_df["actual"] == 1)].sort_values("ensemble_score", ascending=False)
    false_alarms = holdout_df[predicted_pos & (holdout_df["actual"] == 0)].sort_values("ensemble_score", ascending=False)
    near_threshold = holdout_df[(holdout_df["ensemble_score"] >= threshold - 0.1) &
                                 (holdout_df["ensemble_score"] < threshold)].sort_values("ensemble_score", ascending=False)

    print(f"Real holdout composition at threshold: {len(true_hits)} true hits, "
          f"{len(false_alarms)} false alarms, {len(near_threshold)} near-threshold (just below cutoff).", flush=True)

    sample = pd.concat([
        true_hits.head(N_PER_BUCKET),
        false_alarms.head(N_PER_BUCKET),
        near_threshold.head(N_PER_BUCKET),
    ]).reset_index(drop=True)

    print(f"Querying the local LLM for {len(sample)} real cases (several seconds each)...", flush=True)
    records = []
    for i, row in sample.iterrows():
        context = serialize_context(row, threshold)
        llm_prob, reasoning = call_llm(context)
        ensemble_call = "alert" if row["ensemble_score"] >= threshold else "no-alert"
        llm_call = None if llm_prob is None else ("alert" if llm_prob >= threshold else "no-alert")
        rec = {
            "issue_date": str(row["issue_date"].date()), "priogrid_gid": int(row["priogrid_gid"]),
            "country": row["country"], "actual": int(row["actual"]),
            "ensemble_score": round(float(row["ensemble_score"]), 3), "ensemble_call": ensemble_call,
            "llm_probability": llm_prob, "llm_call": llm_call, "llm_reasoning": reasoning,
            "cell_count_30d": float(row["cell_count_30d"]), "cell_count_30d_prior": float(row["cell_count_30d_prior"]),
            "cell_count_30d_delta": float(row["cell_count_30d_delta"]),
        }
        records.append(rec)
        print(f"  [{i+1}/{len(sample)}] {row['country']} {row['issue_date'].date()} "
              f"actual={rec['actual']} ensemble={ensemble_call}({row['ensemble_score']:.3f}) "
              f"llm={llm_call}({llm_prob})", flush=True)

    def acc(recs, key):
        valid = [r for r in recs if r[key] is not None]
        if not valid:
            return None
        correct = sum(1 for r in valid if (r[key] == "alert") == (r["actual"] == 1))
        return correct / len(valid)

    summary = {
        "threshold": threshold,
        "n_cases": len(records),
        "ensemble_accuracy_on_sample": acc(records, "ensemble_call"),
        "llm_accuracy_on_sample": acc(records, "llm_call"),
        "n_false_alarms_tested": int((false_alarms.head(N_PER_BUCKET)).shape[0]),
        "n_false_alarms_llm_corrected": sum(
            1 for r in records if r["actual"] == 0 and r["ensemble_call"] == "alert" and r["llm_call"] == "no-alert"),
        "n_true_hits_tested": int((true_hits.head(N_PER_BUCKET)).shape[0]),
        "n_true_hits_llm_lost": sum(
            1 for r in records if r["actual"] == 1 and r["ensemble_call"] == "alert" and r["llm_call"] == "no-alert"),
        "cases": records,
    }
    with open("../results/icl_hybrid_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved ../results/icl_hybrid_results.json")
    print(f"Ensemble accuracy on sample: {summary['ensemble_accuracy_on_sample']}")
    print(f"LLM accuracy on sample: {summary['llm_accuracy_on_sample']}")
    print(f"False alarms LLM corrected: {summary['n_false_alarms_llm_corrected']}/{summary['n_false_alarms_tested']}")
    print(f"True hits LLM lost: {summary['n_true_hits_llm_lost']}/{summary['n_true_hits_tested']}")


if __name__ == "__main__":
    main()
