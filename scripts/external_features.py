"""
Attaches the four new real external data sources (NASA POWER climate,
WFP food prices, World Bank structural indicators, GDELT GKG thematic/
tone signal) plus a small hand-curated real election calendar onto the
existing GDELT-derived country-week panel from iteration_engine.py.

Every value here traces back to a real download in data/ -- nothing
simulated. Week keys are aligned to the same W-SUN period-start (Monday)
convention iteration_engine.build_panel() already uses, so joins are
exact-match on (country, week), not nearest-date approximations.
"""
import numpy as np
import pandas as pd

WEEK_FREQ = "W-SUN"


def _to_week(dates):
    return pd.to_datetime(dates).dt.to_period(WEEK_FREQ).dt.start_time


def load_climate_weekly():
    df = pd.read_csv("data/climate_daily.csv", parse_dates=["date"])
    df["week"] = _to_week(df["date"])
    weekly = df.groupby(["country", "week"]).agg(
        precip_mm_sum=("precip_mm", "sum"),
        temp_c_mean=("temp_c", "mean"),
    ).reset_index()

    weekly = weekly.sort_values(["country", "week"])
    weekly["baseline_precip_mean"] = weekly.groupby("country")["precip_mm_sum"].transform(
        lambda s: s.expanding(min_periods=4).mean().shift(1))
    weekly["baseline_precip_std"] = weekly.groupby("country")["precip_mm_sum"].transform(
        lambda s: s.expanding(min_periods=4).std().shift(1))
    weekly["precip_anomaly_z"] = (weekly["precip_mm_sum"] - weekly["baseline_precip_mean"]) / weekly["baseline_precip_std"].replace(0, np.nan)

    for col in ["precip_mm_sum", "temp_c_mean", "precip_anomaly_z"]:
        weekly[f"{col}_lag1"] = weekly.groupby("country")[col].shift(1)
        weekly[f"{col}_delta"] = weekly[col] - weekly[f"{col}_lag1"]

    return weekly


def load_food_prices_weekly():
    df = pd.read_csv("data/food_prices_recent.csv", parse_dates=["date"])
    df = df.dropna(subset=["usdprice"])
    df["week"] = _to_week(df["date"])

    # per (country, commodity) trailing baseline, so a country's basket
    # composition shifting over time doesn't get mistaken for real
    # price movement -- each commodity is judged against its own history
    grp = df.groupby(["country", "commodity", "week"])["usdprice"].mean().reset_index()
    grp = grp.sort_values(["country", "commodity", "week"])
    grp["baseline"] = grp.groupby(["country", "commodity"])["usdprice"].transform(
        lambda s: s.expanding(min_periods=3).mean().shift(1))
    grp["pct_dev"] = (grp["usdprice"] - grp["baseline"]) / grp["baseline"].replace(0, np.nan)

    monthly = grp.groupby(["country", "week"]).agg(
        food_price_pct_dev=("pct_dev", "mean"),
        food_n_commodities=("commodity", "nunique"),
    ).reset_index()

    # WFP reports real prices on a real monthly cadence (verified: ~15th of
    # each month), not weekly -- forward-filling the last known reading
    # onto the intervening weeks is the honest way to use a monthly series
    # in a weekly panel (the reading "persists" until the next real update),
    # not a fabrication of new data points
    # NOTE: date_range(freq="W-SUN") emits actual Sundays, but the "week"
    # column elsewhere is the Monday period-start from to_period("W-SUN")
    # -- freq="W-MON" is what actually reproduces those same Mondays
    all_weeks = pd.date_range(monthly["week"].min(), monthly["week"].max(), freq="W-MON")
    frames = []
    for country, sub in monthly.groupby("country"):
        sub = sub.set_index("week").reindex(all_weeks)
        sub["country"] = country
        sub["food_price_pct_dev"] = sub["food_price_pct_dev"].ffill()
        frames.append(sub.reset_index().rename(columns={"index": "week"}))
    weekly = pd.concat(frames, ignore_index=True).sort_values(["country", "week"])

    weekly["food_price_pct_dev_lag1"] = weekly.groupby("country")["food_price_pct_dev"].shift(1)
    weekly["food_price_pct_dev_delta"] = weekly["food_price_pct_dev"] - weekly["food_price_pct_dev_lag1"]

    return weekly[["country", "week", "food_price_pct_dev", "food_price_pct_dev_lag1", "food_price_pct_dev_delta"]]


def load_worldbank_static():
    df = pd.read_csv("data/worldbank_structural.csv")
    keep = ["country", "inflation_cpi_pct", "gdp_growth_pct", "unemployment_pct", "gini_index"]
    return df[keep]


def load_election_proximity(weeks_by_country):
    cal = pd.read_csv("data/election_calendar.csv", parse_dates=["event_date"])
    rows = []
    for country, weeks in weeks_by_country.items():
        events = cal[cal["country"] == country]["event_date"]
        for w in weeks:
            if len(events) == 0:
                days_to_nearest = np.nan
            else:
                deltas = (events - w).dt.days
                days_to_nearest = deltas.iloc[(deltas.abs()).argmin()]
            rows.append({"country": country, "week": w, "days_to_nearest_election": days_to_nearest})
    out = pd.DataFrame(rows)
    # countries with no scheduled election in this window (AF, SU, VE):
    # real absence, filled with a large sentinel distance rather than 0
    out["days_to_nearest_election"] = out["days_to_nearest_election"].fillna(9999)
    out["election_within_4wk"] = (out["days_to_nearest_election"].abs() <= 28).astype(int)
    return out


def load_gkg_weekly():
    df = pd.read_csv("data/gkg_weekly_country.csv", parse_dates=["date"])
    df["week"] = _to_week(df["date"])
    df = df.sort_values(["country", "week"])
    df["gkg_fragility_theme_share_lag1"] = df.groupby("country")["gkg_fragility_theme_share"].shift(1)
    df["gkg_fragility_theme_share_delta"] = df["gkg_fragility_theme_share"] - df["gkg_fragility_theme_share_lag1"]
    df["gkg_mean_tone_lag1"] = df.groupby("country")["gkg_mean_tone"].shift(1)
    df["gkg_mean_tone_delta"] = df["gkg_mean_tone"] - df["gkg_mean_tone_lag1"]
    keep = ["country", "week", "gkg_n_docs", "gkg_fragility_theme_share", "gkg_fragility_theme_share_lag1",
            "gkg_fragility_theme_share_delta", "gkg_mean_tone", "gkg_mean_tone_lag1", "gkg_mean_tone_delta"]
    return df[keep]


def attach_external_features(panel):
    """panel: the existing GDELT country-week panel from iteration_engine.build_panel().
    Returns a copy with all real external feature columns left-joined on (country, week)."""
    out = panel.copy()

    climate = load_climate_weekly()
    out = out.merge(climate, on=["country", "week"], how="left")

    food = load_food_prices_weekly()
    out = out.merge(food, on=["country", "week"], how="left")

    wb = load_worldbank_static()
    out = out.merge(wb, on="country", how="left")

    weeks_by_country = out.groupby("country")["week"].apply(list).to_dict()
    elect = load_election_proximity(weeks_by_country)
    out = out.merge(elect, on=["country", "week"], how="left")

    gkg = load_gkg_weekly()
    out = out.merge(gkg, on=["country", "week"], how="left")

    return out


EXTERNAL_FEATURE_SETS = {
    "climate_only": ["precip_mm_sum_lag1", "precip_mm_sum_delta", "precip_anomaly_z", "temp_c_mean_lag1", "temp_c_mean_delta"],
    "food_price_only": ["food_price_pct_dev_lag1", "food_price_pct_dev_delta"],
    "structural_only": ["inflation_cpi_pct", "gdp_growth_pct", "unemployment_pct", "gini_index"],
    "election_only": ["days_to_nearest_election", "election_within_4wk"],
    "gkg_only": ["gkg_fragility_theme_share_lag1", "gkg_fragility_theme_share_delta", "gkg_mean_tone_lag1", "gkg_mean_tone_delta"],
    "all_external": ["precip_mm_sum_lag1", "precip_mm_sum_delta", "precip_anomaly_z", "temp_c_mean_lag1",
                      "food_price_pct_dev_lag1", "food_price_pct_dev_delta",
                      "inflation_cpi_pct", "gdp_growth_pct", "unemployment_pct", "gini_index",
                      "days_to_nearest_election", "election_within_4wk",
                      "gkg_fragility_theme_share_lag1", "gkg_fragility_theme_share_delta",
                      "gkg_mean_tone_lag1", "gkg_mean_tone_delta"],
}
