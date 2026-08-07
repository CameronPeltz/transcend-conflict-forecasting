"""
Builds the "pure" comparison track from UCDP GED v25.1 (Uppsala Conflict
Data Program, Georeferenced Event Dataset) -- real, fatality-coded,
hand-curated conflict events, 1989-2024, 124 countries, downloaded with
no authentication from ucdp.uu.se/downloads/ged/ged251-csv.zip.

This is deliberately the real academic dataset that most Kaggle
"armed conflict" datasets are themselves mirrors or subsets of. It is
kept in its own file (data/pure_ucdp/ucdp_country_week_panel.csv),
never merged with any GDELT-derived table, so it can always be
evaluated in isolation as the "purer set" per the user's explicit
request -- distinct from both the original small GDELT panel and the
new large self-scraped GDELT panel.

Label: same escalation operationalization philosophy as the GDELT
tracks for comparability (trailing country-specific baseline, not an
absolute threshold), but built from UCDP's own `best` fatality estimate
per event rather than a Goldstein/quad-class proxy -- an actual
fatality-coded label, which the reference document's tab 05 flagged as
the honest thing to do once a dataset like this was available.
"""
import numpy as np
import pandas as pd

# UCDP `country` field name -> our FIPS-style short code, for the 19
# countries used in the large GDELT track, so the two tracks can be
# compared on the same national scope even though they're never merged.
UCDP_NAME_TO_CODE = {
    "Sudan": "SU", "Ethiopia": "ET", "Somalia": "SO", "South Sudan": "OD",
    "Kenya": "KE", "Eritrea": "ER",
    "Afghanistan": "AF", "Myanmar (Burma)": "BM", "Pakistan": "PK",
    "Tajikistan": "TI", "Kyrgyzstan": "KG", "Uzbekistan": "UZ",
    "Colombia": "CO", "Venezuela": "VE", "Ecuador": "EC", "Peru": "PE",
    "Bolivia": "BL",
    "Haiti": "HA", "Nicaragua": "NU",
}

WEEK_FREQ = "W-SUN"


def load_ucdp_raw(path="data/pure_ucdp/GEDEvent_v25_1.csv"):
    cols = ["id", "type_of_violence", "conflict_name", "side_a", "side_b",
            "country", "region", "date_start", "date_end",
            "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown", "best",
            "latitude", "longitude"]
    df = pd.read_csv(path, usecols=cols, parse_dates=["date_start", "date_end"])
    return df


def build_global_summary(df):
    """A real, undiluted look at the full 124-country dataset, not just
    our 19-country subset -- reported once in the write-up for scale
    context even though the panel below is scoped to the 19 comparison
    countries."""
    return {
        "n_events_global": int(len(df)),
        "n_countries_global": int(df["country"].nunique()),
        "date_min": str(df["date_start"].min().date()),
        "date_max": str(df["date_start"].max().date()),
        "total_best_deaths_global": int(df["best"].sum()),
    }


def build_panel(n_lags=2, label_z=1.0, horizons=(1, 2, 4, 8)):
    df = load_ucdp_raw()
    df = df[df["country"].isin(UCDP_NAME_TO_CODE.keys())].copy()
    df["code"] = df["country"].map(UCDP_NAME_TO_CODE)
    df["week"] = df["date_start"].dt.to_period(WEEK_FREQ).dt.start_time

    rows = []
    for code, sub in df.groupby("code"):
        g = sub.groupby("week")
        weekly = g.agg(
            n_events=("id", "count"),
            total_best_deaths=("best", "sum"),
            n_state_based=("type_of_violence", lambda s: (s == 1).sum()),
            n_nonstate=("type_of_violence", lambda s: (s == 2).sum()),
            n_onesided=("type_of_violence", lambda s: (s == 3).sum()),
            n_distinct_dyads=("conflict_name", "nunique"),
        ).reset_index().rename(columns={"period": "week"})
        weekly["code"] = code
        rows.append(weekly)

    panel = pd.concat(rows, ignore_index=True).sort_values(["code", "week"]).reset_index(drop=True)
    panel = panel.rename(columns={"code": "country"})

    # fill the full weekly calendar per country (UCDP weeks with zero
    # events are real signal -- a silent week is not missing data) --
    # bounded by that country's own first/last observed UCDP week
    filled = []
    for country, sub in panel.groupby("country"):
        sub = sub.copy()
        # NB: pd.date_range(freq="W-SUN") anchors on Sundays, but
        # .dt.to_period("W-SUN").dt.start_time (used above) returns the
        # Monday each period *starts* on -- a 6-day-offset mismatch that
        # silently reindexes every row to NaN if you use date_range here.
        # period_range + start_time reproduces the exact same Monday set.
        full_weeks = pd.period_range(start=sub["week"].min(), end=sub["week"].max(), freq=WEEK_FREQ).start_time
        sub = sub.set_index("week").reindex(full_weeks).rename_axis("week").reset_index()
        sub["country"] = country
        for c in ["n_events", "total_best_deaths", "n_state_based", "n_nonstate", "n_onesided", "n_distinct_dyads"]:
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

    # fatality-coded escalation: a real spike in UCDP-confirmed deaths
    # above the country's own trailing baseline, OR event-count spike
    panel["escalation"] = ((z_deaths > label_z) | (z_events > label_z)).astype("Int64")

    for h in horizons:
        panel[f"label_{h}"] = panel.groupby("country")["escalation"].shift(-h)

    lag_cols = ["n_events", "total_best_deaths", "deaths_per_event", "state_based_share", "n_distinct_dyads"]
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            panel[f"{col}_lag{lag}"] = panel.groupby("country")[col].shift(lag)
        panel[f"{col}_delta"] = panel[col] - panel[f"{col}_lag1"]

    return panel, df


UCDP_FEATURE_SET = [
    "n_events_lag1", "n_events_lag2", "n_events_delta",
    "total_best_deaths_lag1", "total_best_deaths_lag2", "total_best_deaths_delta",
    "deaths_per_event_lag1", "deaths_per_event_delta",
    "state_based_share_lag1", "state_based_share_delta",
    "n_distinct_dyads_lag1", "n_distinct_dyads_delta",
]


if __name__ == "__main__":
    panel, raw = build_panel()
    panel.to_csv("data/pure_ucdp/ucdp_country_week_panel.csv", index=False)
    summary = build_global_summary(load_ucdp_raw())  # true 124-country global stats, unfiltered
    summary["n_country_weeks_19country_panel"] = int(len(panel))
    summary["n_positive_label_1"] = int(panel["label_1"].sum(skipna=True))
    summary["countries_in_panel"] = sorted(panel["country"].unique().tolist())
    import json
    with open("data/pure_ucdp/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
