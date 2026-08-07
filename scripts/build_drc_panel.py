"""
Real DRC (Democratic Republic of the Congo) country-week panel, built
the same way as the existing 19-country UCDP and GDELT tracks, so the
existing model classes work on it unmodified. DRC was not in either
original track -- added specifically to test the real Rootwise DRC
radio-transcript data (task: does adding a real novel radio signal
improve forecasting for a country the rest of this project has never
touched).

Two sub-panels, matching the project's existing UCDP_FEATURE_SET /
GDELT FEATURE_SETS shapes exactly:
  - UCDP: real fatality-coded events, UCDP's own name "DR Congo (Zaire)"
  - GDELT: real coded events, FIPS code "CG" (confirmed against
    gdeltproject.org's own FIPS.country.txt -- NOT "CF", which is the
    Republic of the Congo / Congo-Brazzaville, a different country)
"""
import numpy as np
import pandas as pd

WEEK_FREQ = "W-SUN"
UCDP_COUNTRY_NAME = "DR Congo (Zaire)"
GDELT_FIPS = "CG"


def build_ucdp_drc_panel(n_lags=2, label_z=1.0, horizons=(1, 2, 4, 8)):
    cols = ["id", "type_of_violence", "conflict_name", "side_a", "side_b",
            "country", "date_start", "date_end",
            "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown", "best"]
    df = pd.read_csv("data/pure_ucdp/GEDEvent_v25_1.csv", usecols=cols, parse_dates=["date_start"])
    df = df[df["country"] == UCDP_COUNTRY_NAME].copy()
    df["week"] = df["date_start"].dt.to_period(WEEK_FREQ).dt.start_time

    weekly = df.groupby("week").agg(
        n_events=("id", "count"),
        total_best_deaths=("best", "sum"),
        n_state_based=("type_of_violence", lambda s: (s == 1).sum()),
        n_nonstate=("type_of_violence", lambda s: (s == 2).sum()),
        n_onesided=("type_of_violence", lambda s: (s == 3).sum()),
        n_distinct_dyads=("conflict_name", "nunique"),
    ).reset_index()

    full_weeks = pd.period_range(start=weekly["week"].min(), end=weekly["week"].max(), freq=WEEK_FREQ).start_time
    weekly = weekly.set_index("week").reindex(full_weeks).rename_axis("week").reset_index()
    for c in ["n_events", "total_best_deaths", "n_state_based", "n_nonstate", "n_onesided", "n_distinct_dyads"]:
        weekly[c] = weekly[c].fillna(0)
    weekly["country"] = "CG"

    weekly["deaths_per_event"] = weekly["total_best_deaths"] / weekly["n_events"].clip(lower=1)
    weekly["state_based_share"] = weekly["n_state_based"] / weekly["n_events"].clip(lower=1)

    def add_baseline(col):
        weekly[f"baseline_mean_{col}"] = weekly[col].expanding(min_periods=4).mean().shift(1)
        weekly[f"baseline_std_{col}"] = weekly[col].expanding(min_periods=4).std().shift(1)

    for col in ["total_best_deaths", "n_events"]:
        add_baseline(col)

    z_deaths = (weekly["total_best_deaths"] - weekly["baseline_mean_total_best_deaths"]) / weekly["baseline_std_total_best_deaths"].replace(0, np.nan)
    z_events = (weekly["n_events"] - weekly["baseline_mean_n_events"]) / weekly["baseline_std_n_events"].replace(0, np.nan)
    weekly["escalation"] = ((z_deaths > label_z) | (z_events > label_z)).astype("Int64")

    for h in horizons:
        weekly[f"label_{h}"] = weekly["escalation"].shift(-h)

    lag_cols = ["n_events", "total_best_deaths", "deaths_per_event", "state_based_share", "n_distinct_dyads"]
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            weekly[f"{col}_lag{lag}"] = weekly[col].shift(lag)
        weekly[f"{col}_delta"] = weekly[col] - weekly[f"{col}_lag1"]

    return weekly


def load_gdelt_drc_raw(path="data/drc/gdelt_drc_raw.csv"):
    df = pd.read_csv(path, dtype={"ActionGeo_CountryCode": str, "EventCode": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["ActionGeo_CountryCode"] == GDELT_FIPS]
    # same disclosed cleanup as every other GDELT track in this project:
    # a small number of rows carry a retrospective/corrected SQLDATE far
    # outside the real requested window (56 rows here dated 2015-2022,
    # against a 3-year-back-from-2026-08-07 request) -- filtered rather
    # than silently left in to inflate apparent history depth
    df = df[(df["date"] >= "2023-08-07") & (df["date"] <= "2026-08-07")]
    df["QuadClass"] = pd.to_numeric(df["QuadClass"], errors="coerce")
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")
    df["EventCode"] = df["EventCode"].astype(str)
    df["root"] = df["EventCode"].str[:2]
    return df


SEVERE_ROOTS = {"18", "19", "20"}
COERCE_PROTEST_ROOTS = {"14", "17"}


def build_gdelt_drc_panel(raw_df, n_lags=2, label_z=1.0, horizons=(1, 2, 4, 8)):
    df = raw_df.copy()
    df["period"] = df["date"].dt.to_period(WEEK_FREQ).dt.start_time
    g = df.groupby("period")
    weekly = g.agg(
        n_events=("GlobalEventID", "count"),
        n_material_conflict=("QuadClass", lambda s: (s == 4).sum()),
        n_severe_root=("root", lambda s: s.isin(SEVERE_ROOTS).sum()),
        n_coerce_protest_root=("root", lambda s: s.isin(COERCE_PROTEST_ROOTS).sum()),
        mean_goldstein=("GoldsteinScale", "mean"),
        mean_tone=("AvgTone", "mean"),
        tone_neg_share=("AvgTone", lambda s: (s < 0).mean()),
    ).reset_index().rename(columns={"period": "week"})
    actor_counts = df.groupby("period").apply(
        lambda d: pd.concat([d["Actor1Code"], d["Actor2Code"]]).dropna().nunique(), include_groups=False)
    weekly["distinct_actors"] = weekly["week"].map(actor_counts).fillna(0)
    weekly["country"] = "CG"

    weekly["material_conflict_share"] = weekly["n_material_conflict"] / weekly["n_events"].clip(lower=1)
    weekly["severe_root_share"] = weekly["n_severe_root"] / weekly["n_events"].clip(lower=1)
    weekly["coerce_protest_share"] = weekly["n_coerce_protest_root"] / weekly["n_events"].clip(lower=1)

    def add_baseline(col):
        weekly[f"baseline_mean_{col}"] = weekly[col].expanding(min_periods=4).mean().shift(1)
        weekly[f"baseline_std_{col}"] = weekly[col].expanding(min_periods=4).std().shift(1)

    for col in ["material_conflict_share", "severe_root_share", "mean_goldstein"]:
        add_baseline(col)

    z_quad = (weekly["material_conflict_share"] - weekly["baseline_mean_material_conflict_share"]) / weekly["baseline_std_material_conflict_share"].replace(0, np.nan)
    goldstein_drop = weekly["baseline_mean_mean_goldstein"] - weekly["mean_goldstein"]
    weekly["escalation_quad"] = ((z_quad > label_z) | (goldstein_drop > 2.0)).astype("Int64")

    for h in horizons:
        weekly[f"label_quad_{h}"] = weekly["escalation_quad"].shift(-h)

    lag_cols = ["n_events", "material_conflict_share", "severe_root_share", "coerce_protest_share",
                "mean_goldstein", "distinct_actors", "mean_tone", "tone_neg_share"]
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            weekly[f"{col}_lag{lag}"] = weekly[col].shift(lag)
        weekly[f"{col}_delta"] = weekly[col] - weekly[f"{col}_lag1"]

    return weekly


if __name__ == "__main__":
    ucdp_panel = build_ucdp_drc_panel()
    ucdp_panel.to_csv("data/drc/ucdp_drc_panel.csv", index=False)
    print(f"UCDP DRC panel: {len(ucdp_panel)} weeks, {ucdp_panel['week'].min()} to {ucdp_panel['week'].max()}, "
          f"{ucdp_panel['label_1'].sum(skipna=True):.0f} positive (1w horizon)")
