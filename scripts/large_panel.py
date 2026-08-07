"""
Panel builder for the "way bigger" self-scraped GDELT track (task 2):
19 countries across DARPA's 3 named regions plus 2 disclosed extras,
3 years of daily events (vs. the original 6-country/180-day panel in
iteration_engine.py). Mirrors iteration_engine.build_panel()'s logic
exactly (same label definition, same lag/delta feature construction)
so results are comparable across tracks -- only the underlying raw
event source and country/region list changed.

Kept in its own module and its own output file
(data/scraped_large/large_country_week_panel.csv) so this track is
never silently merged with the original small GDELT panel or with the
UCDP pure track -- exactly the "keep the datasets distinct" requirement.
"""
import numpy as np
import pandas as pd

TARGET_COUNTRIES = ["SU", "ET", "SO", "OD", "KE", "ER",
                     "AF", "BM", "PK", "TI", "KG", "UZ",
                     "CO", "VE", "EC", "PE", "BL",
                     "HA", "NU"]

COUNTRY_REGION = {
    "SU": "East/NE Africa", "ET": "East/NE Africa", "SO": "East/NE Africa",
    "OD": "East/NE Africa", "KE": "East/NE Africa", "ER": "East/NE Africa",
    "AF": "Central/SE Asia", "BM": "Central/SE Asia", "PK": "Central/SE Asia",
    "TI": "Central/SE Asia", "KG": "Central/SE Asia", "UZ": "Central/SE Asia",
    "CO": "South America", "VE": "South America", "EC": "South America",
    "PE": "South America", "BL": "South America",
    "HA": "Extra (disclosed)", "NU": "Extra (disclosed)",
}

SEVERE_ROOTS = {"18", "19", "20"}
COERCE_PROTEST_ROOTS = {"14", "17"}


def load_raw(path="data/scraped_large/gdelt_large_raw.csv"):
    df = pd.read_csv(path, dtype={"ActionGeo_CountryCode": str, "EventCode": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["ActionGeo_CountryCode"].isin(TARGET_COUNTRIES)]
    # same disclosed cleanup discipline as the original: drop the handful
    # of rows GDELT tags with a SQLDATE outside our real requested window
    df = df[(df["date"] >= "2023-08-06") & (df["date"] <= "2026-08-05")]
    df["QuadClass"] = pd.to_numeric(df["QuadClass"], errors="coerce")
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce")
    df["root"] = df["EventCode"].astype(str).str[:2]
    return df


def build_panel(granularity="W", n_lags=2, label_z=1.0, horizons=(1, 2, 4, 8), raw_df=None):
    df = raw_df if raw_df is not None else load_raw()
    freq = "W-SUN" if granularity == "W" else "2W-SUN"
    df["period"] = df["date"].dt.to_period(freq).dt.start_time

    rows = []
    for country in TARGET_COUNTRIES:
        sub = df[df.ActionGeo_CountryCode == country]
        if len(sub) == 0:
            continue
        g = sub.groupby("period")
        weekly = g.agg(
            n_events=("GlobalEventID", "count"),
            n_material_conflict=("QuadClass", lambda s: (s == 4).sum()),
            n_verbal_conflict=("QuadClass", lambda s: (s == 3).sum()),
            n_severe_root=("root", lambda s: s.isin(SEVERE_ROOTS).sum()),
            n_coerce_protest_root=("root", lambda s: s.isin(COERCE_PROTEST_ROOTS).sum()),
            mean_goldstein=("GoldsteinScale", "mean"),
            mean_tone=("AvgTone", "mean"),
            tone_neg_share=("AvgTone", lambda s: (s < 0).mean()),
            total_mentions=("NumMentions", "sum"),
        ).reset_index().rename(columns={"period": "week"})
        actor_counts = sub.groupby("period").apply(
            lambda d: pd.concat([d["Actor1Code"], d["Actor2Code"]]).dropna().nunique(), include_groups=False)
        weekly["distinct_actors"] = weekly["week"].map(actor_counts).fillna(0)
        weekly["country"] = country
        weekly["region"] = COUNTRY_REGION[country]
        rows.append(weekly)

    panel = pd.concat(rows, ignore_index=True).sort_values(["country", "week"]).reset_index(drop=True)
    panel["material_conflict_share"] = panel["n_material_conflict"] / panel["n_events"].clip(lower=1)
    panel["severe_root_share"] = panel["n_severe_root"] / panel["n_events"].clip(lower=1)
    panel["coerce_protest_share"] = panel["n_coerce_protest_root"] / panel["n_events"].clip(lower=1)

    def add_baseline(col):
        panel[f"baseline_mean_{col}"] = panel.groupby("country")[col].transform(
            lambda s: s.expanding(min_periods=4).mean().shift(1))
        panel[f"baseline_std_{col}"] = panel.groupby("country")[col].transform(
            lambda s: s.expanding(min_periods=4).std().shift(1))

    for col in ["material_conflict_share", "severe_root_share", "mean_goldstein"]:
        add_baseline(col)

    z_quad = (panel["material_conflict_share"] - panel["baseline_mean_material_conflict_share"]) / panel["baseline_std_material_conflict_share"].replace(0, np.nan)
    z_root = (panel["severe_root_share"] - panel["baseline_mean_severe_root_share"]) / panel["baseline_std_severe_root_share"].replace(0, np.nan)
    goldstein_drop = panel["baseline_mean_mean_goldstein"] - panel["mean_goldstein"]

    panel["escalation_quad"] = ((z_quad > label_z) | (goldstein_drop > 2.0)).astype("Int64")
    panel["escalation_root"] = ((z_root > label_z) | (goldstein_drop > 2.0)).astype("Int64")

    for h in horizons:
        panel[f"label_quad_{h}"] = panel.groupby("country")["escalation_quad"].shift(-h)
        panel[f"label_root_{h}"] = panel.groupby("country")["escalation_root"].shift(-h)

    lag_cols = ["n_events", "material_conflict_share", "severe_root_share", "coerce_protest_share",
                "mean_goldstein", "distinct_actors", "mean_tone", "tone_neg_share"]
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            panel[f"{col}_lag{lag}"] = panel.groupby("country")[col].shift(lag)
        panel[f"{col}_delta"] = panel[col] - panel[f"{col}_lag1"]

    return panel


FEATURE_SETS = {
    "core": ["n_events_lag1", "n_events_lag2", "n_events_delta",
             "material_conflict_share_lag1", "material_conflict_share_lag2", "material_conflict_share_delta",
             "mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
             "distinct_actors_lag1", "distinct_actors_lag2", "distinct_actors_delta",
             "mean_tone_lag1", "mean_tone_delta"],
    "root_taxonomy": ["n_events_lag1", "n_events_lag2", "n_events_delta",
                       "severe_root_share_lag1", "severe_root_share_lag2", "severe_root_share_delta",
                       "coerce_protest_share_lag1", "coerce_protest_share_lag2", "coerce_protest_share_delta",
                       "mean_goldstein_lag1", "mean_goldstein_delta",
                       "distinct_actors_lag1", "distinct_actors_delta",
                       "mean_tone_lag1", "mean_tone_delta"],
    "tone_only": ["mean_tone_lag1", "mean_tone_lag2", "mean_tone_delta", "tone_neg_share_lag1", "tone_neg_share_delta"],
    "goldstein_only": ["mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
                        "material_conflict_share_lag1", "material_conflict_share_delta"],
    "volume_only": ["n_events_lag1", "n_events_lag2", "n_events_delta"],
    "kitchen_sink": ["n_events_lag1", "n_events_lag2", "n_events_delta",
                      "material_conflict_share_lag1", "material_conflict_share_lag2", "material_conflict_share_delta",
                      "severe_root_share_lag1", "severe_root_share_delta",
                      "coerce_protest_share_lag1", "coerce_protest_share_delta",
                      "mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
                      "distinct_actors_lag1", "distinct_actors_lag2", "distinct_actors_delta",
                      "mean_tone_lag1", "mean_tone_delta", "tone_neg_share_lag1", "tone_neg_share_delta"],
}


def rolling_origin_folds(panel, label_col, min_train=8):
    weeks = sorted(panel["week"].unique())
    folds = []
    for i in range(min_train, len(weeks)):
        cutoff = weeks[i]
        train = panel[panel["week"] < cutoff].dropna(subset=[label_col])
        test = panel[panel["week"] == cutoff]
        if len(train) == 0 or len(test) == 0 or test[label_col].isna().all():
            continue
        folds.append((cutoff, train, test))
    return folds


if __name__ == "__main__":
    raw = load_raw()
    print(f"raw events: {len(raw)}, countries: {raw['ActionGeo_CountryCode'].nunique()}, "
          f"date range {raw['date'].min().date()}..{raw['date'].max().date()}")
    panel = build_panel(raw_df=raw)
    panel.to_csv("data/scraped_large/large_country_week_panel.csv", index=False)
    print(f"panel: {len(panel)} country-weeks, {panel['label_quad_1'].sum(skipna=True):.0f} positive (1w horizon)")
    print(panel.groupby("country").size())
