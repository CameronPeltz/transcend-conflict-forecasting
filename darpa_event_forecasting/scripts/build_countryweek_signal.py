"""
Tests a genuinely different question from anything tried so far: does
feeding the country-week escalation classifier's own real, out-of-fold
probability output ("there will likely be some escalation in this
country this week") into the grid-cell/discrete-event model as a
feature actually improve precision on the real DARPA-specified task?

The two approaches have never been combined before in this project --
Criterion 1/2's country-week classifier and this folder's grid-cell
event forecaster have been developed and evaluated in isolation. This
tests whether the coarser, higher-precision country-week signal
carries real information the fine-grained cell/spatial/ACLED features
don't already have.

Country-week label and feature set mirror scripts/build_ucdp_panel.py
(the real logic behind Criterion 2's validated 84% precision figure)
exactly -- same trailing-baseline z-score escalation definition, same
feature set, same ensemble -- rebuilt here on UCDP v26.1 with full
country names (rather than the short codes build_ucdp_panel.py uses)
so it joins directly onto the grid-cell dataset's own `country` field.

Walk-forward discipline: for each of the SAME rolling folds the
grid-cell sweep uses, the country-week ensemble is trained only on
country-week rows strictly before that fold's cutoff, then used to
score every country-week in that fold's test window -- real,
never-look-ahead, out-of-fold probabilities, not a single model fit
on everything and reused.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RAW_PATH = "../data/pure_ucdp_v26/GEDEvent_v26_1.csv"
COUNTRIES_19 = [
    "Afghanistan", "Myanmar (Burma)", "Pakistan", "Tajikistan", "Kyrgyzstan", "Uzbekistan",
    "Sudan", "Ethiopia", "Somalia", "South Sudan", "Kenya", "Eritrea",
    "Colombia", "Venezuela", "Ecuador", "Peru", "Bolivia", "Haiti", "Nicaragua",
]
WEEK_FREQ = "W-SUN"  # period freq; .start_time gives the Monday each period starts on
LABEL_Z = 1.0

UCDP_FEATURE_SET = [
    "n_events_lag1", "n_events_lag2", "n_events_delta",
    "total_best_deaths_lag1", "total_best_deaths_lag2", "total_best_deaths_delta",
    "deaths_per_event_lag1", "deaths_per_event_delta",
    "state_based_share_lag1", "state_based_share_delta",
    "n_distinct_dyads_lag1", "n_distinct_dyads_delta",
]


def build_country_week_panel(n_lags=2, label_z=LABEL_Z):
    """Same logic as scripts/build_ucdp_panel.py in the parent project,
    rebuilt on v26.1 with full country names (no short-code mapping)."""
    cols = ["id", "type_of_violence", "conflict_name", "country", "date_start",
            "best"]
    df = pd.read_csv(RAW_PATH, usecols=cols, parse_dates=["date_start"])
    df = df[df["country"].isin(COUNTRIES_19)].copy()
    df["week"] = df["date_start"].dt.to_period(WEEK_FREQ).dt.start_time

    rows = []
    for country, sub in df.groupby("country"):
        g = sub.groupby("week")
        weekly = g.agg(
            n_events=("id", "count"), total_best_deaths=("best", "sum"),
            n_state_based=("type_of_violence", lambda s: (s == 1).sum()),
            n_distinct_dyads=("conflict_name", "nunique"),
        ).reset_index()
        weekly["country"] = country
        rows.append(weekly)
    panel = pd.concat(rows, ignore_index=True).sort_values(["country", "week"]).reset_index(drop=True)

    filled = []
    for country, sub in panel.groupby("country"):
        sub = sub.copy()
        full_weeks = pd.period_range(start=sub["week"].min(), end=sub["week"].max(), freq=WEEK_FREQ).start_time
        sub = sub.set_index("week").reindex(full_weeks).rename_axis("week").reset_index()
        sub["country"] = country
        for c in ["n_events", "total_best_deaths", "n_state_based", "n_distinct_dyads"]:
            sub[c] = sub[c].fillna(0)
        filled.append(sub)
    panel = pd.concat(filled, ignore_index=True).sort_values(["country", "week"]).reset_index(drop=True)

    panel["deaths_per_event"] = panel["total_best_deaths"] / panel["n_events"].clip(lower=1)
    panel["state_based_share"] = panel["n_state_based"] / panel["n_events"].clip(lower=1)

    def add_baseline(col):
        panel[f"baseline_mean_{col}"] = panel.groupby("country")[col].transform(
            lambda s: s.expanding(min_periods=4).mean().shift(1))
        panel[f"baseline_std_{col}"] = panel.groupby("country")[col].transform(
            lambda s: s.expanding(min_periods=4).std().shift(1))

    for col in ["total_best_deaths", "n_events"]:
        add_baseline(col)

    z_deaths = (panel["total_best_deaths"] - panel["baseline_mean_total_best_deaths"]) / panel["baseline_std_total_best_deaths"].replace(0, np.nan)
    z_events = (panel["n_events"] - panel["baseline_mean_n_events"]) / panel["baseline_std_n_events"].replace(0, np.nan)
    panel["escalation"] = ((z_deaths > label_z) | (z_events > label_z)).astype("Int64")
    panel["label_1"] = panel.groupby("country")["escalation"].shift(-1)

    lag_cols = ["n_events", "total_best_deaths", "deaths_per_event", "state_based_share", "n_distinct_dyads"]
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            panel[f"{col}_lag{lag}"] = panel.groupby("country")[col].shift(lag)
        panel[f"{col}_delta"] = panel[col] - panel[f"{col}_lag1"]

    # standardize country naming to match the grid-cell dataset's own field
    panel["country"] = panel["country"].replace({"Myanmar (Burma)": "Myanmar"})
    return panel


def fit_predict_ensemble(train_X, train_y, test_X):
    pos = max(1, train_y.sum())
    neg = max(1, len(train_y) - pos)
    gbm = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                 random_state=0, class_weight="balanced", n_jobs=-1)
    scaler = StandardScaler()
    train_Xs, test_Xs = scaler.fit_transform(train_X), scaler.transform(test_X)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)
    gbm.fit(train_X, train_y)
    rf.fit(train_X, train_y)
    logreg.fit(train_Xs, train_y)
    p = (gbm.predict_proba(test_X)[:, 1] + rf.predict_proba(test_X)[:, 1] + logreg.predict_proba(test_Xs)[:, 1]) / 3.0
    return p


def rolling_folds(issue_dates_sorted, min_train, n_folds):
    remaining = len(issue_dates_sorted) - min_train
    if remaining < n_folds:
        n_folds = remaining
    step = max(1, remaining // n_folds)
    folds = []
    for k in range(n_folds):
        cutoff_idx = min_train + k * step
        if cutoff_idx >= len(issue_dates_sorted) - 1:
            break
        test_idx_end = min(cutoff_idx + step, len(issue_dates_sorted))
        folds.append((issue_dates_sorted[cutoff_idx], issue_dates_sorted[cutoff_idx:test_idx_end]))
    return folds


def build_walkforward_signal(grid_issue_dates_sorted, min_train=52, n_folds=10):
    """Real, walk-forward, never-look-ahead country-week escalation
    probability for every (country, week) the grid-cell dataset needs --
    using the identical fold boundaries the grid-cell sweep itself uses."""
    print("Building UCDP v26.1 country-week panel (mirrors Criterion 2's real logic)...", flush=True)
    panel = build_country_week_panel()
    print(f"{len(panel)} real country-week rows, {panel['label_1'].sum()} real escalation labels", flush=True)

    folds = rolling_folds(grid_issue_dates_sorted, min_train, n_folds)
    results = []
    for train_cutoff, test_dates in folds:
        train = panel[(panel["week"] < train_cutoff) & panel["label_1"].notna()].dropna(subset=UCDP_FEATURE_SET)
        test = panel[panel["week"].isin(test_dates)].copy()
        if len(train) < 50 or len(test) == 0:
            continue
        test_X = test[UCDP_FEATURE_SET].fillna(0).values
        scores = fit_predict_ensemble(train[UCDP_FEATURE_SET].fillna(0).values, train["label_1"].astype(int).values, test_X)
        test["countryweek_escalation_prob"] = scores
        results.append(test[["country", "week", "countryweek_escalation_prob"]])
        print(f"  fold train_cutoff={train_cutoff.date()}: {len(train)} train rows, "
              f"{len(test)} scored country-weeks", flush=True)

    out = pd.concat(results, ignore_index=True)
    out.to_csv("../data/countryweek_escalation_signal.csv", index=False)
    print(f"\nSaved {len(out)} real walk-forward (country, week) escalation probabilities.")
    return out


if __name__ == "__main__":
    grid = pd.read_csv("../data/discrete_event_candidates_v2.csv", usecols=["issue_date"], parse_dates=["issue_date"])
    issue_dates_sorted = sorted(grid["issue_date"].unique())
    build_walkforward_signal(issue_dates_sorted)
